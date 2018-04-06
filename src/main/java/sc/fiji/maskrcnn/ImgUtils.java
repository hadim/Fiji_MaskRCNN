package sc.fiji.maskrcnn;

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import net.imagej.Dataset;
import net.imagej.tensorflow.GraphBuilder;
import net.imagej.tensorflow.Tensors;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.converter.RealFloatConverter;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class ImgUtils {
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static Tensor<Float> loadFromImgLib(final Dataset d) {
		return loadFromImgLib((RandomAccessibleInterval) d.getImgPlus());
	}

	public static <T extends RealType<T>> Tensor<Float> loadFromImgLib(final RandomAccessibleInterval<T> image) {
		// NB: Assumes XYC ordering. TensorFlow wants YXC.
		RealFloatConverter<T> converter = new RealFloatConverter<>();
		return Tensors.tensorFloat(Converters.convert(image, converter, new FloatType()), new int[] { 1, 0, 2 });
	}

	public static Tensor<Float> normalizeImage(final Tensor<Float> t) {
		try (Graph g = new Graph()) {
			final GraphBuilder b = new GraphBuilder(g);
			// Some constants specific to the pre-trained model at:
			// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
			//
			// - The model was trained with images scaled to 224x224 pixels.
			// - The colors, represented as R, G, B in 1-byte each were converted to
			// float using (value - Mean)/Scale.
			final int H = 224;
			final int W = 224;
			final float mean = 117f;
			final float scale = 1f;

			// Since the graph is being constructed once per execution here, we can
			// use a constant for the input image. If the graph were to be re-used for
			// multiple input images, a placeholder would have been more appropriate.
			final Output<Float> input = g.opBuilder("Const", "input")//
					.setAttr("dtype", t.dataType())//
					.setAttr("value", t).build().output(0);
			final Output<Float> output = b.div(b.sub(b.resizeBilinear(b.expandDims(//
					input, //
					b.constant("make_batch", 0)), //
					b.constant("size", new int[] { H, W })), //
					b.constant("mean", mean)), //
					b.constant("scale", scale));
			try (Session s = new Session(g)) {
				@SuppressWarnings("unchecked")
				Tensor<Float> result = (Tensor<Float>) s.runner().fetch(output.op().name()).run().get(0);
				return result;
			}
		}
	}
}
