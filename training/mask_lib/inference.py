from pathlib import Path
import yaml
import zipfile
import importlib
import tempfile

import quilt
import tensorflow as tf
import numpy as np


def get_tensor(graph, name, suffix=":0"):
    return graph.get_tensor_by_name(f"{name}{suffix}")


def inference(image, model_location):
    """If model_location ends with ".zip", the archive will be downloaded.
    If it doesn't the quilt package will be used."""


    # Configurations of the model
    INPUT_NODE_IMAGE_NAME = "input_image"
    INPUT_NODE_IMAGE_METADATA_NAME = "input_image_meta"
    INPUT_NODE_ANCHORS_NAME = "input_anchors"
    OUTPUT_NODE_NAMES = ["detections", "mrcnn_class", "mrcnn_bbox", "mrcnn_mask", "rois"]

    if not model_location.endswith(".zip"):
        quilt_path = model_location
        quilt_user = quilt_path.split("/")[0]
        quilt_model_name = quilt_path.split("/")[1]

        quilt.install(quilt_path)

        quilt_model = importlib.import_module(f"quilt.data.{quilt_user}.{quilt_model_name}")

        tf_model_path = quilt_model.model()
        preprocessing_model_path = quilt_model.preprocessing_graph()
        postprocessing_model_path = quilt_model.postprocessing_graph()
        parameters_path = quilt_model.parameters()

    else:
        zip_tf_model_path = model_location

        temp_path = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(zip_tf_model_path, "r") as z:
            z.extractall(temp_path)

        tf_model_path = temp_path / f"model.pb"
        preprocessing_model_path = temp_path / "preprocessing_graph.pb"
        postprocessing_model_path = temp_path / "postprocessing_graph.pb"
        parameters_path = temp_path / "parameters.yml"

    # Load parameters
    parameters = yaml.load(open(parameters_path))

    BATCH_SIZE = 1

    IMAGE_MIN_DIM = parameters["image_min_dimension"]
    IMAGE_MAX_DIM = parameters["image_max_dimension"]
    MIN_SCALE = parameters["minimum_scale"]
    MEAN_PIXEL = parameters["mean_pixels"]
    CLASS_IDS = parameters["class_ids"]
    IMAGE_SHAPE = [IMAGE_MAX_DIM, IMAGE_MAX_DIM, 3]

    BACKBONE_STRIDES = parameters["backbone_strides"]
    RPN_ANCHOR_SCALES = parameters["rpn_anchor_scales"]
    RPN_ANCHOR_RATIOS = parameters["rpn_anchor_ratios"]
    RPN_ANCHOR_STRIDE = parameters["rpn_anchor_stride"]

    ## Preprocessing

    # Load preprocessing graph
    with tf.gfile.FastGFile(str(preprocessing_model_path), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.reset_default_graph()
        tf.import_graph_def(graph_def, name='')

    # Do preprocessing
    with tf.Session() as sess:

        # Declare inputs
        image_tensor = get_tensor(sess.graph, "input_image")
        h = get_tensor(sess.graph, "original_image_height")
        w = get_tensor(sess.graph, "original_image_width")
        min_dim = get_tensor(sess.graph, "image_min_dimension")
        max_dim = get_tensor(sess.graph, "image_max_dimension")
        min_scale = get_tensor(sess.graph, "minimum_scale")
        mean_pixels = get_tensor(sess.graph, "mean_pixels")
        class_ids = get_tensor(sess.graph, "class_ids")
        backbone_strides = get_tensor(sess.graph, "backbone_strides")
        rpn_anchor_scales = get_tensor(sess.graph, "rpn_anchor_scales")
        rpn_anchor_ratios = get_tensor(sess.graph, "rpn_anchor_ratios")
        rpn_anchor_stride = get_tensor(sess.graph, "rpn_anchor_stride")

        feed_dict = {image_tensor: image,
                     h: image.shape[0],
                     w: image.shape[1],
                     min_dim: IMAGE_MIN_DIM,
                     max_dim: IMAGE_MAX_DIM,
                     min_scale: MIN_SCALE,
                     mean_pixels: MEAN_PIXEL,
                     class_ids: np.zeros([len(CLASS_IDS)], dtype=np.int32),
                     backbone_strides: BACKBONE_STRIDES,
                     rpn_anchor_scales: RPN_ANCHOR_SCALES,
                     rpn_anchor_ratios: RPN_ANCHOR_RATIOS,
                     rpn_anchor_stride: RPN_ANCHOR_STRIDE}

        # Declare outputs
        fetches = {"molded_image": get_tensor(sess.graph, "molded_image"),
                   "image_metadata": get_tensor(sess.graph, "image_metadata"),
                   "window": get_tensor(sess.graph, "window"),
                   "anchors": get_tensor(sess.graph, "anchors")}

        preprocessing_results = sess.run(fetches, feed_dict=feed_dict)

    ## Prediction

    # Load Mask-RCNN model
    with tf.gfile.FastGFile(str(tf_model_path), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.reset_default_graph()
        tf.import_graph_def(graph_def, name='')

    # Prepare inputs
    molded_images = [preprocessing_results["molded_image"]]
    image_metas = [preprocessing_results["image_metadata"]]
    anchors = [preprocessing_results["anchors"]]

    with tf.Session() as sess:

        # Prepare inputs and outputs
        input_image_tensor = get_tensor(sess.graph, INPUT_NODE_IMAGE_NAME)
        input_image_metadat_tensor = get_tensor(sess.graph, INPUT_NODE_IMAGE_METADATA_NAME)
        input_anchor_tensor = get_tensor(sess.graph, INPUT_NODE_ANCHORS_NAME)

        input_data = {input_image_tensor: molded_images,
                      input_image_metadat_tensor: image_metas,
                      input_anchor_tensor: anchors}

        output_data = {node_name: get_tensor(sess.graph, f'output_{node_name}')
                       for node_name in OUTPUT_NODE_NAMES}

        # Run the prediction
        raw_results = sess.run(fetches=output_data, feed_dict=input_data)

    ## Postprocessing

    # Load preprocessing graph
    with tf.gfile.FastGFile(str(postprocessing_model_path), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.reset_default_graph()
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:

        # Declare inputs
        detections = get_tensor(sess.graph, "detections")
        mrcnn_mask = get_tensor(sess.graph, "mrcnn_mask")
        original_image_shape = get_tensor(sess.graph, "original_image_shape")
        image_shape = get_tensor(sess.graph, "image_shape")
        window = get_tensor(sess.graph, "window")

        feed_dict = {detections: raw_results["detections"],
                     mrcnn_mask: raw_results["mrcnn_mask"],
                     original_image_shape: list(image.shape[:2]) + [1],
                     image_shape: IMAGE_SHAPE,
                     window: preprocessing_results["window"]}

        # Declare outputs
        fetches = {"rois": get_tensor(sess.graph, "rois"),
                   "class_ids": get_tensor(sess.graph, "class_ids"),
                   "scores": get_tensor(sess.graph, "scores"),
                   "masks": get_tensor(sess.graph, "masks")}

        postprocessing_results = sess.run(fetches, feed_dict=feed_dict)

    return postprocessing_results
