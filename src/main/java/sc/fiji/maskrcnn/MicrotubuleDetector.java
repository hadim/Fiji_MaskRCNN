package sc.fiji.maskrcnn;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.module.Module;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.tensorflow.Tensor;

import net.imagej.Dataset;
import net.imagej.ImageJ;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>Detect Microtubules", headless = true)
public class MicrotubuleDetector implements Command {

	@Parameter
	private ImageJ ij;

	@Parameter
	private LogService log;

	@Parameter
	private Dataset inputDataset;

	@Override
	public void run() {
		Module module;

		// Preprocess the image.
		log.info("Preprocess image.");
		module = ij.module().waitFor(ij.command().run(PreprocessImage.class, true, "inputDataset", inputDataset));
		Tensor<?> moldedImage = (Tensor<?>) module.getOutput("moldedImage");
		Tensor<?> imageMetadata = (Tensor<?>) module.getOutput("imageMetadata");
		Tensor<?> windows = (Tensor<?>) module.getOutput("windows");
		Tensor<?> anchors = (Tensor<?>) module.getOutput("anchors");

		log.info(anchors);

		// Detect objects.
		log.info("Run detection.");
		Map<String, Object> inputs = new HashMap<>();
		inputs.put("moldedImage", moldedImage);
		inputs.put("imageMetadata", imageMetadata);
		inputs.put("anchors", anchors);
		module = ij.module().waitFor(ij.command().run(Detector.class, true, inputs));
		Tensor<?> detections = (Tensor<?>) module.getOutput("detections");
		Tensor<?> mrcnn_mask = (Tensor<?>) module.getOutput("mrcnn_mask");
		// Tensor<?> mrcnn_class = (Tensor<?>) module.getOutput("mrcnn_class");
		// Tensor<?> mrcnn_bbox = (Tensor<?>) module.getOutput("mrcnn_bbox");
		// Tensor<?> rois = (Tensor<?>) module.getOutput("rois");

		// Postprocess results.
		log.info("Postprocess results.");
		inputs = new HashMap<>();
		inputs.put("detections", detections);
		inputs.put("mrcnnMask", mrcnn_mask);
		inputs.put("originalImageShape", Tensor.create(9));
		inputs.put("imageShape", Tensor.create(5));
		inputs.put("window", windows);
		module = ij.module().waitFor(ij.command().run(PostprocessImage.class, true, inputs));
		Tensor<?> finalROIs = (Tensor<?>) module.getOutput("rois");
		Tensor<?> classIds = (Tensor<?>) module.getOutput("class_ids");
		Tensor<?> scores = (Tensor<?>) module.getOutput("scores");
		Tensor<?> masks = (Tensor<?>) module.getOutput("masks");

		log.info(finalROIs);
		log.info(classIds);
		log.info(scores);
		log.info(masks);
	}

	public static void main(String[] args) throws IOException {
		final ImageJ ij = new ImageJ();
		ij.launch(args);

		// Open an image and display it.
		String imagePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/single-256x256.tif";
		// imagePath =
		// "/home/hadim/Documents/Code/Postdoc/ij/testdata/fake-flat-corrected.tif";
		final Object dataset = ij.io().open(imagePath);
		ij.ui().show(dataset);

		ij.command().run(MicrotubuleDetector.class, true, "inputDataset", dataset);
	}

}
