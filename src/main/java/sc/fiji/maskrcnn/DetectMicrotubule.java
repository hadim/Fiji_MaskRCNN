package sc.fiji.maskrcnn;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.io.location.FileLocation;
import org.scijava.io.location.Location;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;

import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.tensorflow.TensorFlowService;

@Plugin(type = Command.class, menuPath = "Plugins>Sandbox>Mask RCNN Prediction", headless = true)
public class DetectMicrotubule implements Command {

	private static final String MASK_RCNN_MODEL_URL = "/home/hadim/local/Data/Neural_Networks/Microtubules/tf_model_microtubule20180403T2203.zip";
	private static final String MODEL_NAME = "microtubules";

	private static final String INPUT_NODE_IMAGE = "input_image_1";
	private static final String INPUT_NODE_IMAGE_META = "input_image_meta_1";
	private static final List<String> OUTPUT_NODE_NAMES = Arrays.asList("output_detections", "output_mrcnn_class",
			"output_mrcnn_bbox", "output_mrcnn_mask", "output_rois", "output_rpn_class", "output_rpn_bbox");
	private static final List<String> CLASS_IDS = Arrays.asList("background", "microtubule");
	private static final int DEFAULT_IMAGE_ID = 0;

	@Parameter
	private TensorFlowService tfService;

	@Parameter
	private DatasetService datasetService;

	@Parameter
	private LogService log;

	@Parameter
	private Dataset inputDataset;

	@Parameter(type = ItemIO.OUTPUT)
	private Dataset outputData;

	private Graph graph;
	private Session session;

	private Tensor<Float> inputImage = null;
	private Tensor<Float> inputImageMetadata = null;

	@Override
	public void run() {

		try {
			log.info("Load the model.");
			final Location source = new FileLocation(MASK_RCNN_MODEL_URL);
			this.graph = tfService.loadGraph(source, MODEL_NAME, "model.pb");

			log.info("Preprocess image.");
			this.preprocessInput(inputDataset);

			log.debug(this.inputImageMetadata);
			log.debug(this.inputImage);

			// Build the runner with names of input and output nodes.
			log.info("Setting up the prediction.");
			this.session = new Session(this.graph);
			Runner runner = this.session.runner();

			runner = runner.feed(INPUT_NODE_IMAGE, this.inputImage);
			runner = runner.feed(INPUT_NODE_IMAGE_META, this.inputImageMetadata);

			for (String outputName : OUTPUT_NODE_NAMES) {
				runner = runner.fetch(outputName);
			}

			// Run the model
			log.info("Running the model.");
			final List<Tensor<?>> outputsList = runner.run();

			log.info("Postprocess predictions results.");
			final Map<String, Tensor<?>> outputs = this.postProcessOutput(outputsList);

			log.info(outputs);

			// Do some cleaning
			this.session.close();
			this.graph.close();

		} catch (IOException e) {
			log.error(e);
		}
	}

	private void preprocessInput(Dataset dataset) {
		this.inputImage = ImgUtils.loadFromImgLib(inputDataset);
		this.inputImage = ImgUtils.normalizeImage(this.inputImage);

		int[] originalImageShape = new int[] { 61, 25, 3 };
		int[] imageShape = new int[] { 256, 256, 3 };
		int[] window = new int[] { 61, 25, 125, 200 };
		int scale = 1;

		this.inputImageMetadata = this.getImageMetadata(originalImageShape, imageShape, window, scale);
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
	private Tensor<Float> getImageMetadata(int[] originalImageShape, int[] imageShape, int[] window, int scale) {

		int metadataSize = 12 + CLASS_IDS.size();
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

		for (int i = 0; i < CLASS_IDS.size(); i++) {
			fb.put(12 + i, (float) i);
		}
		return Tensor.create(new long[] { 1, metadataSize }, fb);
	}

	private Map<String, Tensor<?>> postProcessOutput(List<Tensor<?>> outputsList) {

		// Convert the list of tensors to a Map
		final Map<String, Tensor<?>> outputs = new HashMap<>();
		for (int i = 0; i < outputsList.size(); i++) {
			outputs.put(OUTPUT_NODE_NAMES.get(i), outputsList.get(i));
		}

		// TODO
		return outputs;
	}

	public static void main(String[] args) throws IOException {
		final ImageJ ij = new ImageJ();
		ij.launch(args);

		// Open an image and display it.
		final String imagePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/test-tracking.tif";
		final Object dataset = ij.io().open(imagePath);
		ij.ui().show(dataset);

		ij.command().run(DetectMicrotubule.class, true, "inputDataset", dataset);
	}

}
