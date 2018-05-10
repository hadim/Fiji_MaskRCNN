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

import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ops.OpService;
import net.imagej.tensorflow.Tensors;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;

public class PreprocessImage extends AbstractPredictor implements Command {

	private static final String MODEL_URL = "/home/hadim/Drive/Data/Neural_Network/Mask-RCNN/Microtubules/saved_model/manual_complete_model_coco_512_new.zip";
	private static final String MODEL_NAME = "default";
	private static final String MODEL_FILENAME = "preprocessing_graph.pb";

	// Ideally, CLASS_IDS would be provided in the ZIP file containing the model.
	private static final List<String> CLASS_IDS = Arrays.asList("background", "microtubule");

	// Specific parameters.
	private static final Map<String, Object> DEFAULT_INPUT_NODES = new HashMap<String, Object>() {
		private static final long serialVersionUID = 1L;
		{
			put("input_image", null);
			put("original_image_height", null);
			put("original_image_width", null);
			put("image_min_dimension", 10);
			put("image_max_dimension", 512);
			put("minimum_scale", 1.0);
			put("mean_pixels", new float[] { 123.7f, 116.8f, 103.9f });
			put("class_ids", null);
			put("backbone_strides", new int[] { 4, 8, 16, 32, 64 });
			put("rpn_anchor_scales", new int[] { 8, 16, 32, 64, 128 });
			put("rpn_anchor_ratios", new double[] { 0.5, 1, 2 });
			put("rpn_anchor_stride", 1);
		}
	};

	private static final List<String> OUTPUT_NODE_NAMES = Arrays.asList("molded_image", "image_metadata", "window",
			"anchors");

	@Parameter
	private Dataset inputDataset;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> moldedImage;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> imageMetadata;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> windows;

	@Parameter(type = ItemIO.OUTPUT)
	private Tensor<?> anchors;

	@Parameter
	private OpService op;

	@Parameter
	private DatasetService ds;

	private Tensor<Float> inputTensorImage = null;

	@Override
	public void run() {

		this.loadModel(MODEL_URL, MODEL_NAME, MODEL_FILENAME);

		// Get input nodes as tensor.
		Map<String, Object> inputNodes = this.preprocessInputs();

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
		moldedImage = outputsList.get(0);
		imageMetadata = outputsList.get(1);
		windows = outputsList.get(2);
		anchors = outputsList.get(3);

/*		// Convert molded image tensor to Dataset
		Tensor<Float> moldedImageTensor = (Tensor<Float>) moldedImage;
		Img moldedImage = Tensors.imgFloat(moldedImageTensor);
		moldedImage = TensorUtils.reverseReorder(TensorUtils.reverse(moldedImage), new int[] { 1, 0, 2 });*/

		this.clear();
	}

	private Map<String, Object> preprocessInputs() {
		// Compute input values
		Map<String, Object> inputNodes = new HashMap<>(DEFAULT_INPUT_NODES);

		this.inputTensorImage = Tensors.tensorFloat(
				(RandomAccessibleInterval<FloatType>) op.run("convert.float32", inputDataset.getImgPlus()));
		inputNodes.put("input_image", this.inputTensorImage);

		inputNodes.put("original_image_height", ((Long) this.inputTensorImage.shape()[0]).intValue());
		inputNodes.put("original_image_width", ((Long) this.inputTensorImage.shape()[1]).intValue());

		int[] classIDs = new int[CLASS_IDS.size()];
		for (int i = 0; i < CLASS_IDS.size(); i++) {
			classIDs[i] = 0;
		}
		inputNodes.put("class_ids", classIDs);

		// Convert inputs to tensors
		inputNodes.put("original_image_height",
				org.tensorflow.Tensors.create((int) inputNodes.get("original_image_height")));
		inputNodes.put("original_image_width",
				org.tensorflow.Tensors.create((int) inputNodes.get("original_image_width")));

		inputNodes.put("image_min_dimension",
				org.tensorflow.Tensors.create((int) inputNodes.get("image_min_dimension")));
		inputNodes.put("image_max_dimension",
				org.tensorflow.Tensors.create((int) inputNodes.get("image_max_dimension")));

		inputNodes.put("minimum_scale", org.tensorflow.Tensors.create((double) inputNodes.get("minimum_scale")));
		inputNodes.put("mean_pixels", org.tensorflow.Tensors.create((float[]) inputNodes.get("mean_pixels")));
		inputNodes.put("class_ids", org.tensorflow.Tensors.create((int[]) inputNodes.get("class_ids")));

		inputNodes.put("backbone_strides", org.tensorflow.Tensors.create((int[]) inputNodes.get("backbone_strides")));
		inputNodes.put("rpn_anchor_scales", org.tensorflow.Tensors.create((int[]) inputNodes.get("rpn_anchor_scales")));
		inputNodes.put("rpn_anchor_ratios",
				org.tensorflow.Tensors.create((double[]) inputNodes.get("rpn_anchor_ratios")));
		inputNodes.put("rpn_anchor_stride", org.tensorflow.Tensors.create((int) inputNodes.get("rpn_anchor_stride")));

		return inputNodes;
	}

}
