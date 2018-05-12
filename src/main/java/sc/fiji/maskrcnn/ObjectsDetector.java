package sc.fiji.maskrcnn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.io.location.Location;
import org.scijava.log.LogService;
import org.scijava.module.Module;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.tensorflow.Tensor;

import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.table.DefaultGenericTable;
import net.imagej.table.DoubleColumn;
import net.imagej.table.GenericColumn;
import net.imagej.table.GenericTable;
import net.imagej.table.IntColumn;
import net.imagej.tensorflow.TensorFlowService;

@Plugin(type = Command.class, menuPath = "Plugins>Detection>Mask RCNN Detector", headless = true)
public class ObjectsDetector implements Command {

	@Parameter
	private ImageJ ij;

	@Parameter
	private LogService log;

	@Parameter
	private Location modelLocation;

	@Parameter
	private Dataset inputDataset;

	@Parameter(required = false)
	private boolean verbose = false;

	@Parameter(type = ItemIO.OUTPUT)
	private List<Roi> roisList;

	@Parameter(type = ItemIO.OUTPUT)
	private GenericTable table;

	@Parameter
	protected TensorFlowService tfService;

	@Override
	public void run() {

		Module module;

		// Preprocess the image.
		log.info("Preprocess image.");
		Map<String, Object> inputs = new HashMap<>();
		inputs.put("modelLocation", modelLocation);
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
		String modelName = (String) module.getOutput("modelName");

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

		// Postprocess results.
		log.info("Postprocess results.");
		inputs = new HashMap<>();
		inputs.put("modelLocation", modelLocation);
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

		this.roisList = this.fillRoiManager(finalROIs, scores, classIds);
		this.table = this.fillTable(finalROIs, scores, classIds, classLabels);

		log.info("Done");
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

}
