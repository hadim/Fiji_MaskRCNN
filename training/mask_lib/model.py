from pathlib import Path
import zipfile

import numpy as np
import skimage

import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util

import tqdm

from mrcnn import model as modellib
from mrcnn.config import Config


def preprocess_image(original_image):
    if len(original_image.shape) > 2:
        original_image = [im for im in original_image]
    else:
        original_image = [original_image]

    image = original_image.copy()
    image = [skimage.exposure.rescale_intensity(im) for im in image]
    image = [skimage.util.img_as_ubyte(im) for im in image]
    image = [skimage.color.grey2rgb(im) for im in image]
    image = np.array(image)

    return image


def split_as_batches(images, config):
    image_batches = []
    config.BATCH_SIZE = config.IMAGES_PER_GPU
    n_batches = max(1, images.shape[0] // config.BATCH_SIZE + 0)

    for i in range(n_batches):
        n = i * config.BATCH_SIZE
        image_batch = images[n:n + config.BATCH_SIZE]
        image_batches.append(image_batch)
        
    return image_batches


def predict(image, model, progress=False, verbose=0):
    # Split image in multiple batches
    image_batches = split_as_batches(image, model.config)

    # Run Prediction on Batches
    results = []
    for image_batch in tqdm.tqdm(image_batches, disable=not progress):
        results.extend(model.detect(image_batch, verbose=verbose))
        
    return results


def load_model(model_dir, config, mode="inference"):
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode=mode, config=config, model_dir=str(model_dir))
    return model
    
    
def load_weights(model, init_with="last", model_name=None, coco_model_path=None):
    
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    
    elif init_with == "coco":
        model.load_weights(str(coco_model_path), by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    elif init_with == "last":   
        # Load last weights
        if not model_name:
            last_model_dir = Path(model.find_last()[0])
        else:
            last_model_dir = Path(model.model_dir) / model_name
        model_candidates = sorted(list(last_model_dir.glob(f"mask_rcnn_{model.config.NAME}_*.h5")))
        model_path = model_candidates[-1]
        model.load_weights(str(model_path), by_name=True)

        
def export_to_tensorflow(model, tf_model_path, tf_model_zip_path=None):
    # Get keras model and save
    model_keras= model.keras_model

    # All new operations will be in test mode from now on.
    K.set_learning_phase(0)

    # Create output layer with customized names
    prediction_node_names = ["detections", "mrcnn_class", "mrcnn_bbox",
                             "mrcnn_mask", "rois", "rpn_class", "rpn_bbox"]
    prediction_node_names = ["output_" + name for name in prediction_node_names]
    num_output = len(prediction_node_names)

    predidction = []
    for i in range(num_output):
        tf.identity(model_keras.outputs[i], name = prediction_node_names[i])

    sess = K.get_session()

    # Get the object detection graph
    od_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), 
                                                             prediction_node_names)

    with tf.gfile.GFile(str(tf_model_path), 'wb') as f:
        f.write(od_graph_def.SerializeToString())
        
    if tf_model_zip_path:
        z = zipfile.ZipFile(tf_model_zip_path, "w")
        z.write(tf_model_path, arcname="model.pb")
        z.close()