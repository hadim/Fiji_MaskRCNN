
package sc.fiji.maskrcnn;

import java.io.IOException;

import net.imagej.ImageJ;

public class TestPluginParameters {

	public static void main(String[] args) throws IOException {

		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		// Open an image and display it.
		String imagePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/";
		// imagePath += "single-256x256.tif";
		imagePath += "test-tracking-2-frames.tif";
		// imagePath += "seed-small-10-frames.tif";
		// imagePath += "Cell_Colony-1.tif";

		final Object dataset = ij.io().open(imagePath);
		ij.ui().show(dataset);

		ij.command().run(ObjectsDetector.class, true);
	}
}
