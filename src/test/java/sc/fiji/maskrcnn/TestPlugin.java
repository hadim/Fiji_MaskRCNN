package sc.fiji.maskrcnn;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;

import net.imagej.ImageJ;

public class TestPlugin {

	public static void main(String[] args) throws IOException, URISyntaxException {

		final String modelURL = "https://github.com/hadim/Fiji_MaskRCNN/releases/download/Fiji_MaskRCNN-0.3.3/tf_model_coco_512_new.zip";
		final String modelPath = "/home/hadim/Drive/Data/Neural_Network/Mask-RCNN/Microtubules/saved_model/tf_model_coco_512_new.zip";

		final ImageJ ij = new ImageJ();
		ij.launch(args);

		// Open an image and display it.
		String imagePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/";
		imagePath += "single-256x256.tif";
		//imagePath += "Cell_Colony-1.tif";

		final Object dataset = ij.io().open(imagePath);
		ij.ui().show(dataset);

		Map<String, Object> inputs = new HashMap<>();
		inputs.put("modelURL", null);
		inputs.put("modelPath", modelPath);
		inputs.put("modelNameToUse", null);
		inputs.put("inputDataset", dataset);
		inputs.put("verbose", true);
		ij.command().run(ObjectsDetector.class, true, inputs);
	}
}
