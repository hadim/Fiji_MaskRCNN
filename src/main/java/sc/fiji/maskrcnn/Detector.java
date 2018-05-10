package sc.fiji.maskrcnn;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;

public class Detector extends AbstractPredictor implements Command {

	private static final String MODEL_URL = "/home/hadim/Drive/Data/Neural_Network/Mask-RCNN/Microtubules/saved_model/manual_complete_model_coco_512_new.zip";
	private static final String MODEL_NAME = "microtubules";
	private static final String MODEL_FILENAME = "model.pb";

	// Specific parameters.
	private static final Map<String, Object> DEFAULT_INPUT_NODES = new HashMap<String, Object>() {
		private static final long serialVersionUID = 1L;
		{
			put("input_image", null);
			put("input_image_meta", null);
			put("input_anchors", null);
		}
	};

	private static final List<String> OUTPUT_NODE_NAMES = Arrays.asList("output_detections", "output_mrcnn_class",
			"output_mrcnn_bbox", "output_mrcnn_mask", "output_rois");

	@Parameter
	private Tensor<?> moldedImage;

	@Parameter
	private Tensor<?> imageMetadata;

	@Parameter
	private Tensor<?> anchors;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> detections;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> mrcnn_class;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> mrcnn_bbox;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> mrcnn_mask;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> rois;

	@Override
	public void run() {

		this.loadModel(MODEL_URL, MODEL_NAME, MODEL_FILENAME);

		// Get input nodes as tensor.
		Map<String, Object> inputNodes = new HashMap<>(DEFAULT_INPUT_NODES);

		moldedImage = TensorUtils.expandDimension(moldedImage, 0);
		inputNodes.put("input_image", moldedImage);

		imageMetadata = TensorUtils.convertDoubleToFloat((Tensor<Double>) imageMetadata);
		imageMetadata = TensorUtils.expandDimension(imageMetadata, 0);
		inputNodes.put("input_image_meta", imageMetadata);

		anchors = TensorUtils.convertDoubleToFloat((Tensor<Double>) anchors);
		anchors = TensorUtils.expandDimension(anchors, 0);
		inputNodes.put("input_anchors", anchors);

		// Setup the runner with input and output nodes.
		Runner runner = this.session.runner();
		for (Map.Entry<String, Object> entry : inputNodes.entrySet()) {
			runner = runner.feed(entry.getKey(), (Tensor<?>) entry.getValue());
		}

		for (String outputName : OUTPUT_NODE_NAMES) {
			runner = runner.fetch(outputName);
		}

		// Run the model
		final List<Tensor<?>> outputsList = runner.run();

		// Save results in a dict
		detections = outputsList.get(0);
		mrcnn_class = outputsList.get(1);
		mrcnn_bbox = outputsList.get(2);
		mrcnn_mask = outputsList.get(3);
		rois = outputsList.get(4);

		this.clear();
	}
}
