package sc.fiji.maskrcnn;

import java.util.Arrays;
import java.util.Map;

import org.scijava.Context;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.tensorflow.Tensor;

import net.imagej.tensorflow.Tensors;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;

public class DetectionResult {

	@Parameter
	private LogService log;

	private Map<String, Tensor<Float>> detectionOutput;

	public DetectionResult(Context context) {
		context.inject(this);
	}

	public void parseOutput(Map<String, Tensor<Float>> output) {
		this.detectionOutput = output;

		Img<FloatType> detectionsImg = Tensors.imgFloat((Tensor<Float>) this.detectionOutput.get("output_detections"));
		Img<FloatType> roisImg = Tensors.imgFloat((Tensor<Float>) this.detectionOutput.get("output_rois"));
		Img<FloatType> bboxImg = Tensors.imgFloat((Tensor<Float>) this.detectionOutput.get("output_mrcnn_bbox"));
		Img<FloatType> maskImg = Tensors.imgFloat((Tensor<Float>) this.detectionOutput.get("output_mrcnn_mask"));

		log.info(this.detectionOutput.get("output_detections"));
		log.info(this.detectionOutput.get("output_rois"));
		log.info(this.detectionOutput.get("output_mrcnn_bbox"));
		log.info(this.detectionOutput.get("output_mrcnn_mask"));

		// Python code:
		// https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py#L2358

		long nImage = detectionsImg.dimension(2);
		long nResults = detectionsImg.dimension(1);

		long currentImage = 0;

		log.info("N Images : " + nImage);
		log.info("N Results : " + nResults);
		log.info(detectionsImg.dimension(0));

		float y1;
		float x1;
		float y2;
		float x2;
		float class_id;
		float score;

		// Need to de-normalized boxes coordinates.
		
		RandomAccess<FloatType> ra = detectionsImg.randomAccess();
		for (int i = 0; i < nResults; i++) {
			ra.setPosition(new long[] { 0, i, currentImage });
			y1 = ra.get().get();
			ra.setPosition(new long[] { 1, i, currentImage });
			x1 = ra.get().get();
			ra.setPosition(new long[] { 2, i, currentImage });
			y2 = ra.get().get();
			ra.setPosition(new long[] { 3, i, currentImage });
			x2 = ra.get().get();
			ra.setPosition(new long[] { 4, i, currentImage });
			class_id = ra.get().get();
			ra.setPosition(new long[] { 5, i, currentImage });
			score = ra.get().get();

			log.info(Arrays.asList(x1, y1, x2, y2, class_id, score));
		}

	}
}
