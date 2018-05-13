# Mask RCNN Training

The [notebooks](Notebooks/) contain everything you need to train your own Mask RCNN detector. Under the hood it uses [this Mask RCNN TensorFlow implementation](https://github.com/matterport/Mask_RCNN). The notebooks also use [this small Python library](./mask_lib) to reduce code duplication in the noteboks.

The current notebooks have been used to train the model to detect microtubule-like objects in microscopy images. Wether you have an already labeled dataset or want to artifically generate it, you can easily train a new model to detect whatever objects you want. **Please share your trained model here.**

## Usage

It is suggested to use the [Python distribution Anaconda](https://www.anaconda.com/download/#linux). Install the required dependencies using [`environment.yml`](environment.yml).

- [1_Simulate_Training_Dataset](./Notebooks/1_Simulate_Training_Dataset.ipynb): Generate artifical microtubules microscopy images with boolean masks for each artifical microtubules.

- [2_Train](./Notebooks/2_Train.ipynb): Train a Mask RCNN model with the artifical microtubules dataset.

- [3_Predict](./Notebooks/3_Predict.ipynb): Use the trained model to run prediction on custom images.

- [4_Build_Processing_Graph](./Notebooks/4_Build_Processing_Graph.ipynb): Since the pre- and post-processing steps are quite complicated, this notebook embed them in a TensorFlow graph so they can be easily reused in another TensorFlow compatible languages such as Java.

- [5_Convert_Model_To_TensorFlow](./Notebooks/5_Convert_Model_To_TensorFlow.ipynb): Convert the trained model in a TensorFlow compatible format and embed it with pre- and post-processing graphs in a ZIP file.

- [6_Predict_From_SavedModel](./Notebooks/6_Predict_From_SavedModel.ipynb): Use the converted TensorFlow model to run prediction.
