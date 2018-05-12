[![](https://travis-ci.org/hadim/Fiji_MaskRCNN.svg?branch=master)](https://travis-ci.org/hadim/Fiji_MaskRCNN)

# Fiji_MaskRCNN

A Fiji plugin for [Mask RCNN semantic segmentation](https://arxiv.org/abs/1703.06870).

The trained model is generated using [this Tensorflow implementation](https://github.com/matterport/Mask_RCNN) of Mask RCNN.

See [here](training/) about the training part.

If you build a model using this project and you think can be useful to others please contact me. I could add an URL to your model in the Fiji plugin.

## Available Models

| Objects | Version | Description | URL |
| --- | --- | --- | --- |
| Microtubules | 1.0 | Trained with an articially generated dataset. | https://github.com/hadim/Fiji_MaskRCNN/releases/download/Fiji-MaskRCNN-0.1.0/tf_model_coco_512_new.zip

## Limitations

- Only work on 2D images but not stack of 2D images.

## Authors

`Fiji_MaskRCNN` has been created by [Hadrien Mary](mailto:hadrien.mary@gmail.com).

## License

MIT. See [LICENSE.txt](LICENSE.txt)
