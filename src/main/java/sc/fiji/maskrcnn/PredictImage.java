package sc.fiji.maskrcnn;

import java.io.IOException;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.io.location.FileLocation;
import org.scijava.io.location.Location;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.tensorflow.Graph;
import org.tensorflow.Tensor;

import net.imagej.Dataset;
import net.imagej.DatasetService;
import net.imagej.ImageJ;
import net.imagej.tensorflow.TensorFlowService;

@Plugin(type = Command.class, menuPath = "Plugins>Sandbox>Mask RCNN Prediction", headless = true)
public class PredictImage implements Command {

	private static final String MASK_RCNN_MODEL_URL = "/home/hadim/local/Data/Neural_Networks/Microtubules/tf_model_microtubule20180403T2203.zip";
	private static final String MODEL_NAME = "microtubules";

	@Parameter
	private TensorFlowService tfService;

	@Parameter
	private DatasetService datasetService;

	@Parameter
	private LogService log;

	@Parameter
	private Dataset inputDataset;

	@Parameter(type = ItemIO.OUTPUT)
	private Dataset outputData;

	@Override
	public void run() {
		final Location source = new FileLocation(MASK_RCNN_MODEL_URL);
		Graph graph;
		try {
			graph = tfService.loadGraph(source, MODEL_NAME, "model.pb");
			log.info(graph);

			final Tensor<Float> inputTensor = ImgUtils.loadFromImgLib(inputDataset);
			final Tensor<Float> image = ImgUtils.normalizeImage(inputTensor);

			log.info(image);

		} catch (IOException e) {
			log.error(e);
		}
	}

	public static void main(String[] args) throws IOException {
		final ImageJ ij = new ImageJ();
		ij.launch(args);

		// Open an image and display it.
		final String imagePath = "/home/hadim/Documents/Code/Postdoc/ij/testdata/test-tracking.tif";
		final Object dataset = ij.io().open(imagePath);
		ij.ui().show(dataset);

		ij.command().run(PredictImage.class, true, "inputDataset", dataset);
	}

}
