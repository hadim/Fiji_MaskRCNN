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

	private static final String MASK_RCNN_MODEL_URL = "/home/hadim/Drive/Data/Neural_Network/Mask-RCNN/Microtubules/tf_model_coco_all.zip";
	private static final String MODEL_NAME = "microtubules";
	private static final List<String> CLASS_IDS = Arrays.asList("background", "microtubule");
	
	// Those parameters are model-specific.
	private static final int MIN_DIM = 10;
	private static final int MAX_DIM = 256;
	private static final boolean PAD = true;

	@Parameter
	private Dataset inputDataset;

	@Parameter(type = ItemIO.OUTPUT)
	private Dataset outputData;

	@Override
	public void run() {

		final Location source = new FileLocation(MASK_RCNN_MODEL_URL);
		this.loadModel(source, MODEL_NAME);

		this.predict(inputDataset, CLASS_IDS, MIN_DIM, MAX_DIM, PAD);

		this.clear();
	}

	public static void main(String[] args) throws IOException {
		final ImageJ ij = new ImageJ();
		ij.launch(args);

		// Open an image and display it.
		String imagePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/single-256x256.tif";
		//imagePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/fake-flat-corrected.tif";
		final Object dataset = ij.io().open(imagePath);
		ij.ui().show(dataset);

		ij.command().run(DetectorMicrotubule.class, true, "inputDataset", dataset);
	}

}
