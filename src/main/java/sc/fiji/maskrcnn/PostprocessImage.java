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

public class PostprocessImage extends AbstractPredictor implements Command {

	private static final String MODEL_URL = "/home/hadim/Drive/Data/Neural_Network/Mask-RCNN/Microtubules/saved_model/tf_model_coco_512_new.zip";
	private static final String MODEL_NAME = "default";
	private static final String MODEL_FILENAME = "postprocessing_graph.pb";

	// Specific parameters.
	private static final Map<String, Object> DEFAULT_INPUT_NODES = new HashMap<String, Object>() {
		private static final long serialVersionUID = 1L;
		{
			put("detections", null);
			put("mrcnn_mask", null);
			put("original_image_shape", null);
			put("image_shape", null);
			put("window", null);
		}
	};

	private static final List<String> OUTPUT_NODE_NAMES = Arrays.asList("rois", "class_ids", "scores", "masks");

	@Parameter
	private Tensor<?> detections;

	@Parameter
	private Tensor<?> mrcnnMask;

	@Parameter
	private Tensor<?> originalImageShape;

	@Parameter
	private Tensor<?> imageShape;

	@Parameter
	private Tensor<?> window;

	@Parameter(required = false)
	private boolean verbose = false;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> rois;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> class_ids;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> scores;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> masks;

	@Override
	public void run() {

		this.loadModel(MODEL_URL, MODEL_NAME, MODEL_FILENAME);

		// Get input nodes as tensor.
		Map<String, Object> inputNodes = new HashMap<>(DEFAULT_INPUT_NODES);

		inputNodes.put("detections", detections);
		inputNodes.put("mrcnn_mask", mrcnnMask);
		inputNodes.put("original_image_shape", originalImageShape);
		inputNodes.put("image_shape", imageShape);
		inputNodes.put("window", window);

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
		rois = outputsList.get(0);
		class_ids = outputsList.get(1);
		scores = outputsList.get(2);
		masks = outputsList.get(3);

		if (verbose) {
			log.info("rois : " + rois);
			log.info("class_ids : " + class_ids);
			log.info("scores : " + scores);
			log.info("masks : " + masks);
		}

		this.clear();
	}

}
