package sc.fiji.maskrcnn;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;

import org.scijava.io.http.HTTPLocation;
import org.scijava.io.location.Location;

import net.imagej.ImageJ;

public class TestPlugin {

	public static void main(String[] args) throws IOException, URISyntaxException {
		final String modelURL = "https://github.com/hadim/Fiji_MaskRCNN/releases/download/Fiji-MaskRCNN-0.1.0/tf_model_coco_512_new.zip";
		final String modelName = "microtubules";

		Location modelLocation = new HTTPLocation(modelURL);
		// modelLocation = new FileLocation(
		// "/home/hadim/Drive/Data/Neural_Network/Mask-RCNN/Microtubules/saved_model/tf_model_coco_512_new.zip");

		final ImageJ ij = new ImageJ();
		ij.launch(args);

		// Open an image and display it.
		String imagePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/single-256x256.tif";
		// imagePath =
		// "/home/hadim/Documents/Code/Postdoc/ij/testdata/fake-flat-corrected.tif";

		final Object dataset = ij.io().open(imagePath);
		ij.ui().show(dataset);

		Map<String, Object> inputs = new HashMap<>();
		inputs.put("modelLocation", modelLocation);
		inputs.put("inputDataset", dataset);
		inputs.put("verbose", true);
		ij.command().run(ObjectsDetector.class, true, inputs);
	}
}
