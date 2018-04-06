package sc.fiji.maskrcnn;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.scijava.io.location.Location;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;

import net.imagej.Dataset;
import net.imagej.tensorflow.TensorFlowService;

public abstract class DetectorMaskRCNN {

	private static final String INPUT_NODE_IMAGE_NAME = "input_image_1";
	private static final String INPUT_NODE_IMAGE_METADATA_NAME = "input_image_meta_1";
	private static final List<String> OUTPUT_NODE_NAMES = Arrays.asList("output_detections", "output_mrcnn_class",
			"output_mrcnn_bbox", "output_mrcnn_mask", "output_rois", "output_rpn_class", "output_rpn_bbox");

	private static final int DEFAULT_IMAGE_ID = 0;

	@Parameter
	private TensorFlowService tfService;

	@Parameter
	private LogService log;

	private Graph graph;
	private Session session;

	private Tensor<Float> inputImage = null;
	private Tensor<Float> inputImageMetadata = null;

	protected void loadModel(Location modelLocation, String modelName) {
		try {
			log.info("Load the model.");
			this.graph = tfService.loadGraph(modelLocation, modelName, "model.pb");
			this.session = new Session(this.graph);
		} catch (IOException e) {
			log.error(e);
		}
	}

	protected void predict(Dataset dataset, List<String> classIds) {

		log.info("Preprocess image.");
		this.preprocessInput(dataset, classIds);

		log.info(this.inputImageMetadata);
		log.info(this.inputImage);

		// Build the runner with names of input and output nodes.
		log.info("Setting up the prediction.");

		Runner runner = this.session.runner();

		runner = runner.feed(INPUT_NODE_IMAGE_NAME, this.inputImage);
		runner = runner.feed(INPUT_NODE_IMAGE_METADATA_NAME, this.inputImageMetadata);

		for (String outputName : OUTPUT_NODE_NAMES) {
			runner = runner.fetch(outputName);
		}

		// Run the model
		log.info("Running the model.");
		final List<Tensor<?>> outputsList = runner.run();

		log.info("Postprocess predictions results.");
		final Map<String, Tensor<?>> outputs = this.postProcessOutput(outputsList);

		log.info(outputs);

	}

	protected void clear() {
		// Do some cleaning
		this.session.close();
		this.graph.close();
	}

	protected void preprocessInput(Dataset dataset, List<String> classIds) {
		this.inputImage = ImgUtils.loadFromImgLib(dataset);
		this.inputImage = ImgUtils.normalizeImage(this.inputImage);

		int[] originalImageShape = new int[] { 61, 25, 3 };
		int[] imageShape = new int[] { 256, 256, 3 };
		int[] window = new int[] { 61, 25, 125, 200 };
		int scale = 1;

		this.inputImageMetadata = this.getImageMetadata(originalImageShape, imageShape, window, scale, classIds);
	}

	/*
	 * All metadata used by the model are stored in a 1D tensor:
	 * 
	 * - image_id: An integer ID of the image. Useful for debugging (size=1). -
	 * original_image_shape: [H, W, C] before resizing or padding (size=3). -
	 * image_shape: [H, W, C] after resizing and padding (size=3). - window: (y1,
	 * x1, y2, x2) in pixels. The area of the image where the real image is
	 * (excluding the padding) (size=4 (y1, x1, y2, x2) in image coordinates). -
	 * scale: The scaling factor applied to the original image (float32) size=1. -
	 * active_class_ids: List of class_ids available in the dataset from which the
	 * image came. Useful if training on images from multiple datasets where not all
	 * classes are present in all datasets (size=num_classes).
	 * 
	 */
	protected Tensor<Float> getImageMetadata(int[] originalImageShape, int[] imageShape, int[] window, int scale,
			List<String> classIds) {

		int metadataSize = 12 + classIds.size();
		FloatBuffer fb = FloatBuffer.allocate(metadataSize);

		int imageId = DEFAULT_IMAGE_ID; // Not important during prediction.

		fb.put(0, (float) imageId);
		fb.put(1, (float) originalImageShape[0]);
		fb.put(2, (float) originalImageShape[1]);
		fb.put(3, (float) originalImageShape[2]);
		fb.put(4, (float) imageShape[0]);
		fb.put(5, (float) imageShape[1]);
		fb.put(6, (float) imageShape[2]);
		fb.put(7, (float) window[0]);
		fb.put(8, (float) window[1]);
		fb.put(9, (float) window[2]);
		fb.put(10, (float) window[3]);
		fb.put(11, (float) scale);

		for (int i = 0; i < classIds.size(); i++) {
			fb.put(12 + i, (float) i);
		}
		return Tensor.create(new long[] { 1, metadataSize }, fb);
	}

	protected Map<String, Tensor<?>> postProcessOutput(List<Tensor<?>> outputsList) {

		// Convert the list of tensors to a Map
		final Map<String, Tensor<?>> outputs = new HashMap<>();
		for (int i = 0; i < outputsList.size(); i++) {
			outputs.put(OUTPUT_NODE_NAMES.get(i), outputsList.get(i));
		}

		// TODO
		return outputs;
	}

	public Graph getGraph() {
		return graph;
	}

	public Session getSession() {
		return session;
	}

	public Tensor<Float> getInputImage() {
		return inputImage;
	}

	public Tensor<Float> getInputImageMetadata() {
		return inputImageMetadata;
	}

}
