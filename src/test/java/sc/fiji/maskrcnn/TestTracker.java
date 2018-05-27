
package sc.fiji.maskrcnn;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import net.imagej.ImageJ;

import sc.fiji.maskrcnn.tracking.ObjectTracker;

public class TestTracker {

	public static void main(String[] args) throws IOException {

		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		// Open an image and display it.
		String basePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/";

		String maskPath = basePath + "Masks-of-seed-small-10-frames.tif";
		String tablePath = basePath + "Masks-of-seed-small-10-frames.csv";

		final Object dataset = ij.io().open(maskPath);
		ij.ui().show(dataset);

		Map<String, Object> inputs = new HashMap<>();
		inputs.put("mask", dataset);
		inputs.put("table", null);
		ij.command().run(ObjectTracker.class, true, inputs);

	}
}
