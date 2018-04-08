package sc.fiji.maskrcnn;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.scijava.Context;
import org.scijava.io.location.Location;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.ui.UIService;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;

import net.imagej.Dataset;
import net.imagej.ops.OpService;
import net.imagej.tensorflow.GraphBuilder;
import net.imagej.tensorflow.TensorFlowService;
import net.imagej.tensorflow.Tensors;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;

public abstract class DetectorMaskRCNN {

	private static final String INPUT_NODE_IMAGE_NAME = "input_image_1";
	private static final String INPUT_NODE_IMAGE_METADATA_NAME = "input_image_meta_1";
	private static final List<String> OUTPUT_NODE_NAMES = Arrays.asList("output_detections", "output_mrcnn_class",
			"output_mrcnn_bbox", "output_mrcnn_mask", "output_rois", "output_rpn_class", "output_rpn_bbox");

	private static final int DEFAULT_IMAGE_ID = 0;
	private static final float[] MEAN_PIXEL = { (float) 123.7, (float) 116.8, (float) 103.9 };

	@Parameter
	private Context context;
	
	@Parameter
	private TensorFlowService tfService;

	@Parameter
	private LogService log;

	@Parameter
	private OpService op;

	@Parameter
	private UIService ui;

	private Graph graph;
	private Session session;

	private Tensor<Float> inputImage = null;
	private Tensor<Float> inputPreprocessedImage = null;
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

	protected void predict(Dataset dataset, List<String> classIds, int minDim, int maxDim, boolean pad) {

		log.info("Preprocess image.");
		this.preprocessInput(dataset, classIds, minDim, maxDim, pad);

		// Build the runner with names of input and output nodes.
		log.info("Setting up the prediction.");

		Runner runner = this.session.runner();

		runner = runner.feed(INPUT_NODE_IMAGE_NAME, this.inputPreprocessedImage);
		runner = runner.feed(INPUT_NODE_IMAGE_METADATA_NAME, this.inputImageMetadata);

		for (String outputName : OUTPUT_NODE_NAMES) {
			runner = runner.fetch(outputName);
		}

		// Run the model
		log.info("Running the model.");
		final List<Tensor<?>> outputsList = runner.run();

		log.info("Postprocess predictions results.");
		DetectionResult results = this.postProcessOutput(outputsList);

		log.info(results);
	}

	protected void clear() {
		// Do some cleaning
		this.session.close();
		this.graph.close();
	}

	/*
	 * Here is the multiple steps of preprocessing: - Convert to float32 - Add one
	 * axis at 0: for batch axis and required by the model. - Add one axis at -1. -
	 * Convert to RGB: channel axis at -1. - Subtract intensities to MEAN_PIXEL. -
	 * Resize image if needed. - Pad image to a maxDimxmaxDim square with 0 value.
	 */
	protected void preprocessInput(Dataset dataset, List<String> classIds, int minDim, int maxDim, boolean pad) {

		this.inputImage = Tensors
				.tensorFloat((RandomAccessibleInterval<FloatType>) op.run("convert.float32", dataset.getImgPlus()));

		final Graph g = new Graph();
		final GraphBuilder b = new GraphBuilder(g);

		Output<Float> imageOutput = g.opBuilder("Const", "input").setAttr("dtype", this.inputImage.dataType())
				.setAttr("value", this.inputImage).build().output(0);

		if (imageOutput.shape().numDimensions() == 2) {
			imageOutput = g.opBuilder("ExpandDims", "expand_batch").addInput(imageOutput)
					.addInput(b.constant("batch_axis", 0)).build().output(0);
		}

		// Convert grayscale to RGB
		imageOutput = g.opBuilder("ExpandDims", "expand_channel").addInput(imageOutput)
				.addInput(b.constant("channel_axis", -1)).build().output(0);
		imageOutput = g.opBuilder("Tile", "tile").addInput(imageOutput)
				.addInput(b.constant("multiples", new int[] { 1, 1, 1, 3 })).build().output(0);

		imageOutput = g.opBuilder("Sub", "sub_mean").addInput(imageOutput)
				.addInput(b.constant("mean_pixel", MEAN_PIXEL)).build().output(0);

		int ori_h = (int) imageOutput.shape().size(1);
		int ori_w = (int) imageOutput.shape().size(2);

		int h = ori_h;
		int w = ori_w;

		int[] window = { 0, 0, h, w };
		float scale = 1;
		int[][] padding = { { 0, 0 }, { 0, 0 }, { 0, 0 } };
		int image_max;

		if (minDim != -1) {
			scale = Math.max(1, minDim / Math.min(h, w));
		}

		if (maxDim != -1) {
			image_max = Math.max(h, w);
			if (Math.round(image_max * scale) > maxDim) {
				scale = maxDim / image_max;
			}
		}

		if (scale != 1) {
			int[] new_size = { Math.round(h * scale), Math.round(w * scale) };
			imageOutput = g.opBuilder("ResizeBilinear", "resize").addInput(imageOutput)
					.addInput(b.constant("size", new_size)).build().output(0);
		}

		if (pad) {
			h = (int) imageOutput.shape().size(1);
			w = (int) imageOutput.shape().size(2);

			int top_pad = (maxDim - h) / 2;
			int bottom_pad = maxDim - h - top_pad;
			int left_pad = (maxDim - w) / 2;
			int right_pad = maxDim - w - left_pad;

			padding[0][0] = top_pad;
			padding[0][1] = bottom_pad;
			padding[1][0] = left_pad;
			padding[1][1] = right_pad;
			padding[2][0] = 0;
			padding[2][1] = 0;

			window[0] = top_pad;
			window[1] = left_pad;
			window[2] = h + top_pad;
			window[3] = w + left_pad;

			// Pad image to maxDim x maxDim
			int[] stack = { 0, 0, top_pad, bottom_pad, left_pad, right_pad, 0, 0 };
			Output<Float> opPaddings = g.opBuilder("Reshape", "reshape").addInput(b.constant("stack", stack))
					.addInput(b.constant("extra", new int[] { 4, 2 })).build().output(0);
			imageOutput = g.opBuilder("Pad", "pad").addInput(imageOutput).addInput(opPaddings).build().output(0);
		}

		final Session s = new Session(g);
		this.inputPreprocessedImage = (Tensor<Float>) s.runner().fetch(imageOutput.op().name()).run().get(0);

		int[] originalImageShape = new int[] { ori_h, ori_w, 3 };
		int[] imageShape = new int[] { h, w, 3 };

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
	protected Tensor<Float> getImageMetadata(int[] originalImageShape, int[] imageShape, int[] window, float scale,
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
			fb.put(12 + i, (float) 0);
		}
		return Tensor.create(new long[] { 1, metadataSize }, fb);
	}

	protected DetectionResult postProcessOutput(List<Tensor<?>> outputsList) {

		// Convert the list of tensors to a Map
		final Map<String, Tensor<Float>> outputs = new HashMap<>();
		for (int i = 0; i < outputsList.size(); i++) {
			outputs.put(OUTPUT_NODE_NAMES.get(i), (Tensor<Float>) outputsList.get(i));
		}

		DetectionResult results = new DetectionResult(context);
		results.parseOutput(outputs);
		return results;
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
