package sc.fiji.maskrcnn;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.io.location.FileLocation;
import org.scijava.io.location.Location;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import net.imagej.Dataset;
import net.imagej.ImageJ;

@Plugin(type = Command.class, menuPath = "Plugins>Segmentation>Detect Microtubules", headless = true)
public class DetectorMicrotubule extends DetectorMaskRCNN implements Command {

	private static final String MASK_RCNN_MODEL_URL = "/home/hadim/local/Data/Neural_Networks/Microtubules/tf_model_microtubule20180403T2203.zip";
	private static final String MODEL_NAME = "microtubules";
	private static final List<String> CLASS_IDS = Arrays.asList("background", "microtubule");
	
	// Could be parameters
	private static final int MIN_DIM = 10;
	private static final int MAX_DIM = 256;
	private static final boolean PAD = true;
	private static final float[] MEAN_PIXEL = { (float) 123.7, (float) 116.8, (float) 103.9};

	@Parameter
	private Dataset inputDataset;

	@Parameter(type = ItemIO.OUTPUT)
	private Dataset outputData;

	@Override
	public void run() {

		final Location source = new FileLocation(MASK_RCNN_MODEL_URL);
		this.loadModel(source, MODEL_NAME);

		this.predict(inputDataset, CLASS_IDS, MIN_DIM, MAX_DIM, PAD, MEAN_PIXEL);

		this.clear();
	}

	public static void main(String[] args) throws IOException {
		final ImageJ ij = new ImageJ();
		ij.launch(args);

		// Open an image and display it.
		final String imagePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/test-tracking-single.tif";
		final Object dataset = ij.io().open(imagePath);
		ij.ui().show(dataset);

		ij.command().run(DetectorMicrotubule.class, true, "inputDataset", dataset);
	}

}
