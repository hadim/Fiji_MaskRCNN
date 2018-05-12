package sc.fiji.maskrcnn;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import net.imagej.ImageJ;

public class TestPlugin {

	public static void main(String[] args) throws IOException {
		final String modelURL = "/home/hadim/Drive/Data/Neural_Network/Mask-RCNN/Microtubules/saved_model/tf_model_coco_512_new.zip";
		final String modelName = "microtubules";

		final ImageJ ij = new ImageJ();
		ij.launch(args);

		// Open an image and display it.
		String imagePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/single-256x256.tif";
		// imagePath =
		// "/home/hadim/Documents/Code/Postdoc/ij/testdata/fake-flat-corrected.tif";
		
		final Object dataset = ij.io().open(imagePath);
		ij.ui().show(dataset);

		Map<String, Object> inputs = new HashMap<>();
		inputs.put("modelURL", modelURL);
		inputs.put("inputDataset", dataset);
		inputs.put("verbose", true);
		ij.command().run(ObjectsDetector.class, true, inputs);
	}
}
