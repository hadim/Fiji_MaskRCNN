
package sc.fiji.maskrcnn;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;

import net.imagej.ImageJ;

public class TestPlugin {

	public static void main(String[] args) throws IOException, URISyntaxException {

		final String modelURL =
			"https://github.com/hadim/Fiji_MaskRCNN/releases/download/Fiji_MaskRCNN-0.4.0/tf_model_microtubule_coco_512.zip";
		final String modelPath =
			"/home/hadim/Drive/Data/Neural_Network/Mask-RCNN/Microtubules/saved_model/tf_model_microtubule_coco_512.zip";

		final ImageJ ij = new ImageJ();
		ij.launch(args);

		// Open an image and display it.
		String basePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/";
		String imagePath = basePath + "single-256x256.tif";
		// imagePath = basePath + "test-tracking-2-frames.tif";
		// imagePath = basePath + "seed-small-10-frames.tif";
		// imagePath = basePath + "Cell_Colony-1.tif";
		// imagePath = basePath + "FakeTracks.tif";
		// imagePath = basePath + "Cell_Colony.tif";
		imagePath = basePath + "Spindle-1-Frame.tif";
		// imagePath = basePath + "Spindle-1-Frame-Small.tif";

		final Object dataset = ij.io().open(imagePath);
		ij.ui().show(dataset);

		Map<String, Object> inputs = new HashMap<>();
		inputs.put("modelURL", null);
		inputs.put("modelPath", modelPath);
		inputs.put("modelNameToUse", null);
		inputs.put("inputDataset", dataset);
		ij.command().run(ObjectsDetector.class, true, inputs);
	}
}
