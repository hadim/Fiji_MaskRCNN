package sc.fiji.maskrcnn;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FilenameUtils;
import org.scijava.ItemIO;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.io.http.HTTPLocation;
import org.scijava.io.location.FileLocation;
import org.scijava.io.location.Location;
import org.scijava.log.LogService;
import org.scijava.module.Module;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.tensorflow.Tensor;
import org.yaml.snakeyaml.Yaml;

import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imagej.table.DefaultGenericTable;
import net.imagej.table.DoubleColumn;
import net.imagej.table.GenericColumn;
import net.imagej.table.GenericTable;
import net.imagej.table.IntColumn;
import net.imagej.tensorflow.TensorFlowService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.RealType;

@Plugin(type = Command.class, menuPath = "Plugins>Detection>Mask RCNN Detector", headless = true)
public class ObjectsDetector implements Command {

	static private Map<String, String> AVAILABLE_MODELS = new HashMap<>();
	static {
		AVAILABLE_MODELS.put("Microtubule",
				"https://github.com/hadim/Fiji_MaskRCNN/releases/download/Fiji_MaskRCNN-0.3.3/tf_model_coco_512_new.zip");
	}

	@Parameter
	private ImageJ ij;

	@Parameter
	private LogService log;

	@Parameter
	private DatasetService ds;

	@Parameter
	private StatusService ss;

	@Parameter(required = false)
	private String modelURL = null;

	@Parameter(required = false)
	private String modelPath = null;

	@Parameter
	private Dataset inputDataset;

	@Parameter(choices = { "Microtubule" }, required = false)
	private String modelNameToUse = null;

	@Parameter(type = ItemIO.OUTPUT)
	private List<Roi> roisList;

	@Parameter(type = ItemIO.OUTPUT)
	private GenericTable table;

	@Parameter(type = ItemIO.OUTPUT)
	private Dataset masksImage;

	@Parameter
	protected TensorFlowService tfService;

	@Parameter
	protected OpService ops;

	@Parameter
	private CustomDownloadService cds;

	private Location modelLocation;

	// This name is only used for caching the model ZIP file on disk.
	private String modelnameCache;

	private Map<String, Object> parameters;

	@Override
	public void run() {

		try {

			// Get model location
			if (modelURL == null) {
				if (modelPath == null) {
					if (modelNameToUse == null) {
						throw new Exception("modelURL, modelPath or modelNameToUse, needs to be provided.");
					} else {
						this.modelLocation = new FileLocation(AVAILABLE_MODELS.get(modelNameToUse));
					}

					throw new Exception("modelURL, modelPath or modelToUse, needs to be provided.");
				} else {
					this.modelLocation = new FileLocation(modelPath);
				}
			} else {
				this.modelLocation = new HTTPLocation(modelURL);
			}

			this.checkInput();
			this.runPrediction();

		} catch (Exception e) {
			log.error(e);
		}
	}

	private void runPrediction() {

		this.modelnameCache = FilenameUtils.getBaseName(modelLocation.getURI().toString());

		// Load the ZIP model file to access the parameters.
		this.loadParameters();

		// How many images to process ?
		long nImages;
		if (this.inputDataset.numDimensions() == 3) {
			nImages = this.inputDataset.dimension(2);
		} else {
			nImages = 1;
		}
		Dataset twoDImage;

		double startTime;
		double stopTime;
		double elapsedTime;

		// Preprocess the image.
		log.info("Preprocessing image...");
		ss.showStatus(0, 100, "Preprocessing image.");
		startTime = System.currentTimeMillis();

		Module preprocessModule;
		List<String> classLabels = new ArrayList<>();

		Map<String, List<Tensor<?>>> preprocessingOutputsMap = new HashMap<>();
		preprocessingOutputsMap.put("moldedImage", new ArrayList<>());
		preprocessingOutputsMap.put("imageMetadata", new ArrayList<>());
		preprocessingOutputsMap.put("windows", new ArrayList<>());
		preprocessingOutputsMap.put("anchors", new ArrayList<>());
		preprocessingOutputsMap.put("originalImageShape", new ArrayList<>());
		preprocessingOutputsMap.put("imageShape", new ArrayList<>());

		for (int i = 0; i < nImages; i++) {
			// Get a 2D image and run it.
			twoDImage = this.getStack(i);
			preprocessModule = this.preprocessSingleImage(twoDImage);

			// Gather outputs in a Map.
			for (Map.Entry<String, List<Tensor<?>>> entry : preprocessingOutputsMap.entrySet()) {
				entry.getValue().add((Tensor<?>) preprocessModule.getOutput(entry.getKey()));
			}
			// Gather non-Tensor outputs.
			classLabels = (List<String>) preprocessModule.getOutput("classLabels");
		}

		stopTime = System.currentTimeMillis();
		elapsedTime = stopTime - startTime;
		log.info("Preprocessing done. It tooks " + elapsedTime / 1000 + " s.");

		// Detect objects.
		log.info("Running detection...");
		ss.showStatus(33, 100, "Running detection.");
		startTime = System.currentTimeMillis();

		Module detectionModule;

		Map<String, List<Tensor<?>>> detectionOutputsMap = new HashMap<>();
		detectionOutputsMap.put("detections", new ArrayList<>());
		detectionOutputsMap.put("mrcnn_mask", new ArrayList<>());
		detectionOutputsMap.put("mrcnn_class", new ArrayList<>());
		detectionOutputsMap.put("mrcnn_bbox", new ArrayList<>());
		detectionOutputsMap.put("rois", new ArrayList<>());

		for (int i = 0; i < nImages; i++) {
			// Get a 2D image and run it.
			twoDImage = this.getStack(i);
			detectionModule = this.detectSingleImage(preprocessingOutputsMap.get("moldedImage").get(i),
					preprocessingOutputsMap.get("imageMetadata").get(i), preprocessingOutputsMap.get("anchors").get(i));

			// Gather outputs in a Map.
			for (Map.Entry<String, List<Tensor<?>>> entry : detectionOutputsMap.entrySet()) {
				entry.getValue().add((Tensor<?>) detectionModule.getOutput(entry.getKey()));
			}
		}

		stopTime = System.currentTimeMillis();
		elapsedTime = stopTime - startTime;
		log.info("Detection done. It tooks " + elapsedTime / 1000 + " s.");

		// Postprocess results.
		log.info("Postprocessing results...");
		ss.showStatus(66, 100, "Postprocessing results.");
		startTime = System.currentTimeMillis();

		Module postprocessModule;

		Map<String, List<Tensor<?>>> postprocessOutputsMap = new HashMap<>();
		postprocessOutputsMap.put("rois", new ArrayList<>());
		postprocessOutputsMap.put("class_ids", new ArrayList<>());
		postprocessOutputsMap.put("scores", new ArrayList<>());
		postprocessOutputsMap.put("mrcnn_bbox", new ArrayList<>());
		postprocessOutputsMap.put("masks", new ArrayList<>());

		for (int i = 0; i < nImages; i++) {
			// Get a 2D image and run it.
			twoDImage = this.getStack(i);
			postprocessModule = this.postprocessSingleImage(detectionOutputsMap.get("detections").get(i),
					detectionOutputsMap.get("mrcnn_mask").get(i),
					preprocessingOutputsMap.get("originalImageShape").get(i),
					preprocessingOutputsMap.get("imageShape").get(i), preprocessingOutputsMap.get("windows").get(i));

			// Gather outputs in a Map.
			for (Map.Entry<String, List<Tensor<?>>> entry : postprocessOutputsMap.entrySet()) {
				entry.getValue().add((Tensor<?>) postprocessModule.getOutput(entry.getKey()));
			}
		}

		stopTime = System.currentTimeMillis();
		elapsedTime = stopTime - startTime;
		log.info("Postprocessing done. It tooks " + elapsedTime / 1000 + " s.");

		int nDetectedObjects = postprocessOutputsMap.get("scores").stream().mapToInt(tensor -> (int) tensor.shape()[0])
				.sum();

		// Format and return outputs.
		if (nDetectedObjects == 0) {
			this.roisList = new ArrayList<Roi>();
			this.table = new DefaultGenericTable();
			this.masksImage = null;
		} else {
			this.roisList = this.fillRoiManager(postprocessOutputsMap.get("rois"), postprocessOutputsMap.get("scores"),
					postprocessOutputsMap.get("class_ids"));
			this.table = this.fillTable(postprocessOutputsMap.get("rois"), postprocessOutputsMap.get("scores"),
					postprocessOutputsMap.get("class_ids"), classLabels);
			this.masksImage = this.createMaskImage(postprocessOutputsMap.get("masks"));
		}

		log.info(nDetectedObjects + " objects detected.");

		ss.showStatus(100, 100, "Done.");
		log.info("Detection done");
	}

	private Dataset getStack(int position) {
		if (this.inputDataset.numDimensions() == 3) {
			return ds.create((RandomAccessibleInterval) ops.transform().hyperSliceView(this.inputDataset, 2, position));
		} else {
			return this.inputDataset;
		}
	}

	private Module preprocessSingleImage(Dataset data) {
		Map<String, Object> inputs = new HashMap<>();
		inputs.put("modelLocation", this.modelLocation);
		inputs.put("modelName", this.modelnameCache);
		inputs.put("inputDataset", data);
		Module module = ij.module().waitFor(ij.command().run(PreprocessImage.class, true, inputs));
		return module;
	}

	private Module detectSingleImage(Tensor<?> moldedImage, Tensor<?> imageMetadata, Tensor<?> anchors) {
		Map<String, Object> inputs = new HashMap<>();
		inputs.put("modelLocation", this.modelLocation);
		inputs.put("modelName", this.modelnameCache);
		inputs.put("moldedImage", moldedImage);
		inputs.put("imageMetadata", imageMetadata);
		inputs.put("anchors", anchors);
		Module module = ij.module().waitFor(ij.command().run(Detector.class, true, inputs));
		return module;
	}

	private Module postprocessSingleImage(Tensor<?> detections, Tensor<?> mrcnn_mask, Tensor<?> originalImageShape,
			Tensor<?> imageShape, Tensor<?> windows) {
		Map<String, Object> inputs = new HashMap<>();
		inputs.put("modelLocation", modelLocation);
		inputs.put("modelName", this.modelnameCache);
		inputs.put("detections", detections);
		inputs.put("mrcnnMask", mrcnn_mask);
		inputs.put("originalImageShape", originalImageShape);
		inputs.put("imageShape", imageShape);
		inputs.put("window", windows);
		Module module = ij.module().waitFor(ij.command().run(PostprocessImage.class, true, inputs));
		return module;
	}

	protected List<Roi> fillRoiManager(List<Tensor<?>> rois, List<Tensor<?>> scores, List<Tensor<?>> classIds) {
		// TODO: output masks as polygons ? (need a marching cube-like algorithm.)

		RoiManager rm = RoiManager.getRoiManager();
		rm.reset();

		List<Roi> roisList = new ArrayList<>();
		Roi box = null;
		int x1, y1, x2, y2;
		int id = 0;

		for (int n = 0; n < rois.size(); n++) {
			int nRois = (int) rois.get(n).shape()[0];
			int nCoords = (int) rois.get(n).shape()[1];
			int[][] roisArray = rois.get(n).copyTo(new int[nRois][nCoords]);

			float[] scoresArray = scores.get(n).copyTo(new float[(int) scores.get(n).shape()[0]]);
			int[] classIdsArray = classIds.get(n).copyTo(new int[(int) classIds.get(n).shape()[0]]);

			for (int i = 0; i < roisArray.length; i++) {
				x1 = roisArray[i][0];
				y1 = roisArray[i][1];
				x2 = roisArray[i][2];
				y2 = roisArray[i][3];
				box = new Roi(y1, x1, y2 - y1, x2 - x1);
				box.setPosition(n + 1);
				box.setName("BBox-" + id + "-Score-" + scoresArray[i] + "-ClassID-" + classIdsArray[i] + "-Frame-" + n);
				rm.addRoi(box);
				roisList.add(box);
				id++;
			}
		}

		return roisList;
	}

	protected GenericTable fillTable(List<Tensor<?>> rois, List<Tensor<?>> scores, List<Tensor<?>> classIds,
			List<String> classLabels) {

		GenericTable table = new DefaultGenericTable();
		table.add(new IntColumn("id"));
		table.add(new IntColumn("frame"));
		table.add(new IntColumn("class_id"));
		table.add(new GenericColumn("class_label"));
		table.add(new DoubleColumn("score"));
		table.add(new IntColumn("x"));
		table.add(new IntColumn("y"));
		table.add(new IntColumn("width"));
		table.add(new IntColumn("height"));

		int x1, y1, x2, y2;
		int lastRow = 0;
		int id = 0;

		for (int n = 0; n < rois.size(); n++) {

			int nRois = (int) rois.get(n).shape()[0];
			int nCoords = (int) rois.get(n).shape()[1];
			int[][] roisArray = rois.get(n).copyTo(new int[nRois][nCoords]);

			float[] scoresArray = scores.get(n).copyTo(new float[(int) scores.get(n).shape()[0]]);
			int[] classIdsArray = classIds.get(n).copyTo(new int[(int) classIds.get(n).shape()[0]]);

			for (int i = 0; i < roisArray.length; i++) {
				x1 = roisArray[i][0];
				y1 = roisArray[i][1];
				x2 = roisArray[i][2];
				y2 = roisArray[i][3];
				table.appendRow();
				lastRow = table.getRowCount() - 1;
				table.set("id", lastRow, id);
				table.set("frame", lastRow, n);
				table.set("class_id", lastRow, classIdsArray[i]);
				table.set("class_label", lastRow, classLabels.get(classIdsArray[i]));
				table.set("score", lastRow, (double) scoresArray[i]);
				table.set("x", lastRow, y1);
				table.set("y", lastRow, x1);
				table.set("width", lastRow, y1 - y2);
				table.set("height", lastRow, x1 - x2);
				id++;
			}
		}
		return table;
	}

	private <T extends RealType<?>> Dataset createMaskImage(List<Tensor<?>> masks) {

		List<Img<T>> maskList = new ArrayList<>();
		for (Tensor<?> mask : masks) {
			maskList.add((Img<T>) net.imagej.tensorflow.Tensors.imgFloat((Tensor<Float>) mask));
		}

		/*
		 * RandomAccessibleInterval<T> im = Views.stack(maskList); ds.create(im);
		 * 
		 * AxisType[] axisTypes = new AxisType[] { Axes.X, Axes.Y, Axes.CHANNEL,
		 * Axes.TIME }; ImgPlus imgPlus = new ImgPlus(ds.create(im), "image",
		 * axisTypes);
		 * 
		 * log.info(imgPlus.numDimensions()); log.info(imgPlus.dimension(0));
		 * log.info(imgPlus.dimension(1)); log.info(imgPlus.dimension(2));
		 * log.info(imgPlus.dimension(3));
		 * 
		 * return ds.create(imgPlus);
		 */
		return null;
	}

	public void checkInput() throws Exception {
		if (this.inputDataset.numDimensions() != 2 && this.inputDataset.numDimensions() != 3) {
			throw new Exception("Input image must have 2 or 3 dimensions.");
		}

		long maxSize = (long) this.parameters.get("image_max_dimension");
		if (this.inputDataset.dimension(0) > maxSize) {
			throw new Exception("Width cannot be greater than " + maxSize + " pixels.");
		}
		if (this.inputDataset.dimension(1) > maxSize) {
			throw new Exception("Height cannot be greater than " + maxSize + " pixels.");
		}
	}

	private void loadParameters() {
		try {
			File parametersFile = cds.loadFile(this.modelLocation, this.modelnameCache, "parameters.yml");

			InputStream input = new FileInputStream(parametersFile);
			Yaml yaml = new Yaml();
			this.parameters = (Map<String, Object>) yaml.load(input);

		} catch (IOException e) {
			log.error("Can't read parameters.yml in the ZIP model file: " + e);
		}
	}

}
