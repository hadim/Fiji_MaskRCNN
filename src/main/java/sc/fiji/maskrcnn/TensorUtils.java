package sc.fiji.maskrcnn;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import net.imagej.tensorflow.GraphBuilder;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.ImgView;
import net.imglib2.type.numeric.RealType;
import net.imglib2.view.Views;

public class TensorUtils {

	public static Tensor<Float> convertDoubleToFloat(Tensor<Double> arrayTensor) {
		DoubleBuffer buff = DoubleBuffer.allocate(arrayTensor.numElements());
		arrayTensor.writeTo(buff);
		double[] arrayDouble = buff.array();
		float[] arrayFloat = new float[arrayDouble.length];
		for (int i = 0; i < arrayDouble.length; i++) {
			arrayFloat[i] = (float) arrayDouble[i];
		}
		return Tensor.create(arrayTensor.shape(), FloatBuffer.wrap(arrayFloat));
	}

	public static Tensor<Double> convertFloatToDouble(Tensor<Float> arrayTensor) {
		FloatBuffer buff = FloatBuffer.allocate(arrayTensor.numElements());
		arrayTensor.writeTo(buff);
		float[] arrayFloat = buff.array();
		double[] arrayDouble = new double[arrayFloat.length];
		for (int i = 0; i < arrayFloat.length; i++) {
			arrayDouble[i] = (double) arrayFloat[i];
		}
		return Tensor.create(arrayTensor.shape(), DoubleBuffer.wrap(arrayDouble));
	}

	public static Tensor<?> expandDimension(Tensor<?> tensor, int dimension) {
		try (Graph g = new Graph()) {
			final GraphBuilder b = new GraphBuilder(g);

			final Output input = b.constant("input", tensor);
			final Output output = g.opBuilder("ExpandDims", "ExpandDims").addInput(input)
					.addInput(b.constant("dimension", dimension)).build().output(0);

			try (Session s = new Session(g)) {
				Tensor<?> result = (Tensor<?>) s.runner().fetch(output.op().name()).run().get(0);
				return result;
			}
		}
	}

	private static <T extends RealType<T>> Img<T> reorder(Img<T> image, int[] dimOrder) {
		RandomAccessibleInterval<T> result = reorder((RandomAccessibleInterval<T>) image, dimOrder);
		return ImgView.wrap(result, image.factory());
	}

	protected static <T extends RealType<T>> RandomAccessibleInterval<T> reorder(RandomAccessibleInterval<T> image,
			int[] dimOrder) {
		RandomAccessibleInterval<T> output = image;

		// Array which contains for each dimension information on which dimension it is
		// right now
		int[] moved = IntStream.range(0, image.numDimensions()).toArray();

		// Loop over all dimensions and move it to the right spot
		for (int i = 0; i < image.numDimensions(); i++) {
			int from = moved[i];
			int to = dimOrder[i];

			// Move the dimension to the right dimension
			output = Views.permute(output, from, to);

			// Now we have to update which dimension was moved where
			moved[i] = to;
			moved = Arrays.stream(moved).map(v -> v == to ? from : v).toArray();
		}
		return output;
	}

	protected static <T extends RealType<T>> Img<T> reverseReorder(Img<T> image, int[] dimOrder) {
		int[] reverseDimOrder = new int[dimOrder.length];
		for (int i = 0; i < dimOrder.length; i++) {
			reverseDimOrder[dimOrder[i]] = i;
		}
		return reorder(image, reverseDimOrder);
	}

	/** Flips all dimensions {@code d0,d1,...,dn -> dn,...,d1,d0}. */
	public static <T extends RealType<T>> Img<T> reverse(Img<T> image) {
		RandomAccessibleInterval<T> reversed = reverse((RandomAccessibleInterval<T>) image);
		return ImgView.wrap(reversed, image.factory());
	}

	/** Flips all dimensions {@code d0,d1,...,dn -> dn,...,d1,d0}. */
	public static <T extends RealType<T>> RandomAccessibleInterval<T> reverse(RandomAccessibleInterval<T> image) {
		RandomAccessibleInterval<T> reversed = image;
		for (int d = 0; d < image.numDimensions() / 2; d++) {
			reversed = Views.permute(reversed, d, image.numDimensions() - d - 1);
		}
		return reversed;
	}
}
