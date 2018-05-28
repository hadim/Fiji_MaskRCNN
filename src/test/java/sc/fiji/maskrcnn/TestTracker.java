
package sc.fiji.maskrcnn;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import net.imagej.ImageJ;
import net.imagej.table.CommonsCSVTableIOPlugin;
import net.imagej.table.GenericTable;

import sc.fiji.maskrcnn.tracking.ObjectTracker;

public class TestTracker {

	public static void main(String[] args) throws IOException {

		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		// Open an image and display it.
		String basePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/";

		String maskPath = basePath + "Masks-of-seed-small-10-frames.tif";
		String tablePath = basePath + "Masks-of-seed-small-10-frames.tif.csv";

		// The writer in TestPlugin does not use the same symbols for column
		// separation. So the csv file needs to be manually converted to use tab
		// instead of ','. That should be fixed with the new scijava-table-io
		// component.
		CommonsCSVTableIOPlugin tableIOPlugin = new CommonsCSVTableIOPlugin();
		GenericTable table = (GenericTable) tableIOPlugin.open(tablePath);

		final Object dataset = ij.io().open(maskPath);
		ij.ui().show(dataset);

		Map<String, Object> inputs = new HashMap<>();
		inputs.put("masks", dataset);
		inputs.put("table", table);
		inputs.put("linkingMaxDistance", 10.0);
		inputs.put("gapClosingMaxDistance", 10.0);
		inputs.put("maxFrameGap", 5);
		inputs.put("fillROIManager", true);
		ij.command().run(ObjectTracker.class, true, inputs);
	}
}
