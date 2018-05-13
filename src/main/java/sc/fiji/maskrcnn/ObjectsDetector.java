package sc.fiji.maskrcnn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FilenameUtils;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.io.http.HTTPLocation;
import org.scijava.io.location.FileLocation;
import org.scijava.io.location.Location;
import org.scijava.log.LogService;
import org.scijava.module.Module;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.tensorflow.Tensor;

import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.table.DefaultGenericTable;
import net.imagej.table.DoubleColumn;
import net.imagej.table.GenericColumn;
import net.imagej.table.GenericTable;
import net.imagej.table.IntColumn;
import net.imagej.tensorflow.TensorFlowService;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;

@Plugin(type = Command.class, menuPath = "Plugins>Detection>Mask RCNN Detector", headless = true)
public class ObjectsDetector implements Command {

	static private Map<String, String> AVAILABLE_MODELS = new HashMap<>();
	static {
		AVAILABLE_MODELS.put("Microtubule",
				"https://github.com/hadim/Fiji_MaskRCNN/releases/download/Fiji_MaskRCNN-0.3.0/tf_model_coco_512_new.zip");
	}

	@Parameter
	private ImageJ ij;

	@Parameter
	private LogService log;

	@Parameter
	private DatasetService ds;

	@Parameter(required = false)
	private String modelURL = null;

	@Parameter(required = false)
	private String modelPath = null;

	@Parameter
	private Dataset inputDataset;

	@Parameter(choices = { "Microtubule" }, required = false)
	private String modelNameToUse = null;

	@Parameter(required = false)
	private boolean verbose = false;

	@Parameter(type = ItemIO.OUTPUT)
	private List<Roi> roisList;

	@Parameter(type = ItemIO.OUTPUT)
	private GenericTable table;

	@Parameter(type = ItemIO.OUTPUT)
	private Dataset masksImage;

	@Parameter
	protected TensorFlowService tfService;

	private Location modelLocation;

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

		// This name is only used for caching the model ZIP file on disk.
		String modelName = FilenameUtils.getBaseName(modelLocation.getURI().toString());

		Module module;

		// Preprocess the image.
		log.info("Preprocess image.");
		Map<String, Object> inputs = new HashMap<>();
		inputs.put("modelLocation", modelLocation);
		inputs.put("modelName", modelName);
		inputs.put("inputDataset", inputDataset);
		inputs.put("verbose", verbose);
		module = ij.module().waitFor(ij.command().run(PreprocessImage.class, true, inputs));
		Tensor<?> moldedImage = (Tensor<?>) module.getOutput("moldedImage");
		Tensor<?> imageMetadata = (Tensor<?>) module.getOutput("imageMetadata");
		Tensor<?> windows = (Tensor<?>) module.getOutput("windows");
		Tensor<?> anchors = (Tensor<?>) module.getOutput("anchors");
		Tensor<?> originalImageShape = (Tensor<?>) module.getOutput("originalImageShape");
		Tensor<?> imageShape = (Tensor<?>) module.getOutput("imageShape");
		List<String> classLabels = (List<String>) module.getOutput("classLabels");

		// Detect objects.
		log.info("Run detection.");
		inputs = new HashMap<>();
		inputs.put("modelLocation", modelLocation);
		inputs.put("modelName", modelName);
		inputs.put("moldedImage", moldedImage);
		inputs.put("imageMetadata", imageMetadata);
		inputs.put("anchors", anchors);
		inputs.put("verbose", verbose);
		module = ij.module().waitFor(ij.command().run(Detector.class, true, inputs));
		Tensor<?> detections = (Tensor<?>) module.getOutput("detections");
		Tensor<?> mrcnn_mask = (Tensor<?>) module.getOutput("mrcnn_mask");
		Tensor<?> mrcnn_class = (Tensor<?>) module.getOutput("mrcnn_class");
		Tensor<?> mrcnn_bbox = (Tensor<?>) module.getOutput("mrcnn_bbox");
		Tensor<?> rois = (Tensor<?>) module.getOutput("rois");

		log.info(detections.shape()[0]);

		// Postprocess results.
		log.info("Postprocess results.");
		inputs = new HashMap<>();
		inputs.put("modelLocation", modelLocation);
		inputs.put("modelName", modelName);
		inputs.put("detections", detections);
		inputs.put("mrcnnMask", mrcnn_mask);
		inputs.put("originalImageShape", originalImageShape);
		inputs.put("imageShape", imageShape);
		inputs.put("window", windows);
		inputs.put("verbose", verbose);
		module = ij.module().waitFor(ij.command().run(PostprocessImage.class, true, inputs));
		Tensor<?> finalROIs = (Tensor<?>) module.getOutput("rois");
		Tensor<?> classIds = (Tensor<?>) module.getOutput("class_ids");
		Tensor<?> scores = (Tensor<?>) module.getOutput("scores");
		Tensor<?> masks = (Tensor<?>) module.getOutput("masks");

		if (scores.shape()[0] == 0) {
			this.roisList = new ArrayList<Roi>();
			this.table = new DefaultGenericTable();
			this.masksImage = null;
		} else {
			this.roisList = this.fillRoiManager(finalROIs, scores, classIds);
			this.table = this.fillTable(finalROIs, scores, classIds, classLabels);

			Img<FloatType> im = net.imagej.tensorflow.Tensors.imgFloat((Tensor<Float>) masks);
			this.masksImage = ds.create(im);
		}

		log.info(scores.shape()[0] + " objects detected.");
		log.info("Detection done");
	}

	protected List<Roi> fillRoiManager(Tensor<?> rois, Tensor<?> scores, Tensor<?> classIds) {
		// Add ROI of detected objects to RoiManager
		// TODO: add resizing masks step and output them as polygon.

		RoiManager rm = RoiManager.getRoiManager();
		rm.reset();

		List<Roi> roisList = new ArrayList<>();
		Roi box = null;
		int x1, y1, x2, y2;

		int nRois = (int) rois.shape()[0];
		int nCoords = (int) rois.shape()[1];
		int[][] roisArray = rois.copyTo(new int[nRois][nCoords]);

		float[] scoresArray = scores.copyTo(new float[(int) scores.shape()[0]]);
		int[] classIdsArray = classIds.copyTo(new int[(int) classIds.shape()[0]]);

		for (int i = 0; i < roisArray.length; i++) {
			x1 = roisArray[i][0];
			y1 = roisArray[i][1];
			x2 = roisArray[i][2];
			y2 = roisArray[i][3];
			box = new Roi(y1, x1, y2 - y1, x2 - x1);
			box.setName("BBox-" + i + "-Score-" + scoresArray[i] + "-ClassID-" + classIdsArray[i]);
			rm.addRoi(box);
			roisList.add(box);
		}
		return roisList;
	}

	protected GenericTable fillTable(Tensor<?> rois, Tensor<?> scores, Tensor<?> classIds, List<String> classLabels) {

		GenericTable table = new DefaultGenericTable();
		table.add(new IntColumn("id"));
		table.add(new IntColumn("class_id"));
		table.add(new GenericColumn("class_label"));
		table.add(new DoubleColumn("score"));
		table.add(new IntColumn("x"));
		table.add(new IntColumn("y"));
		table.add(new IntColumn("width"));
		table.add(new IntColumn("height"));

		int x1, y1, x2, y2;

		int nRois = (int) rois.shape()[0];
		int nCoords = (int) rois.shape()[1];
		int[][] roisArray = rois.copyTo(new int[nRois][nCoords]);

		float[] scoresArray = scores.copyTo(new float[(int) scores.shape()[0]]);
		int[] classIdsArray = classIds.copyTo(new int[(int) classIds.shape()[0]]);

		for (int i = 0; i < roisArray.length; i++) {
			x1 = roisArray[i][0];
			y1 = roisArray[i][1];
			x2 = roisArray[i][2];
			y2 = roisArray[i][3];
			table.appendRow();
			table.set("id", i, i);
			table.set("class_id", i, classIdsArray[i]);
			table.set("class_label", i, classLabels.get(classIdsArray[i]));
			table.set("score", i, (double) scoresArray[i]);
			table.set("x", i, y1);
			table.set("y", i, x1);
			table.set("width", i, y1 - y2);
			table.set("height", i, x1 - x2);
		}

		return table;
	}

	public void checkInput() throws Exception {
		if (this.inputDataset.numDimensions() != 2) {
			throw new Exception("Input image must have exactly 2 dimensions (XY).");
		}

		// TODO: should be setup from model parameters.
		if (this.inputDataset.dimension(0) > 512) {
			throw new Exception("Width cannot be greater than 512 pixels.");
		}
		if (this.inputDataset.dimension(1) > 512) {
			throw new Exception("Height cannot be greater than 512 pixels.");
		}
	}

}
