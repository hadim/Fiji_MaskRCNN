# Source

This data package contains the model as TensorFlow graphs for the [Fiji MaskRCNN plugin](https://github.com/hadim/Fiji_MaskRCNN).

This model has been trained to recognize mirotubules from TIRF images of *in vitro* reconstitution assay. 

# Content

The package contains the following files:

- `model.pb`: The Mask RCNN graph used for object detection. 
- `preprocessing_graph.pb`: The graph performing preprocessing on the input image.
- `postprocessing_graph.pb`: The graph that rescale the outputs of the Mask RCNN prediction to the original image.
- `parameters.yml`: A YAML file that contain hyper-parameters of the model such as label names and maximum image size.
- `README.md`: This README.
- `LICENSE.txt`: The license file of this dataset.

# License

Creative Commons Attribution 4.0 International.

# Authors

- [Hadrien Mary](mailto:hadrien.mary@gmail.com).
