import json

import pandas as pd
import numpy as np
import cv2

import tqdm
import skimage
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='skimage')

from mrcnn import utils
from mrcnn import visualize


class MicrotubuleDataset(utils.Dataset):

    def load_microtubules(self, fnames, line_thickness=4, verbose=False):
        self.add_class("microtubule", 1, "microtubule")

        for i, fname in tqdm.tqdm(enumerate(fnames), total=len(fnames), disable=not verbose, leave=False):
            im = skimage.io.imread(fname)
            self.add_image("microtubule", image_id=i, path=fname,
                           width=im.shape[0], height=im.shape[1],
                           line_thickness=line_thickness)
        self.prepare()

    def load_image(self, image_id):
        info = self.image_info[image_id]
        im = skimage.io.imread(info["path"])
        # Convert to 8bit
        im = skimage.util.img_as_ubyte(im)
        im = skimage.exposure.rescale_intensity(im)
        # Convert to RGB
        im = skimage.color.grey2rgb(im)
        return im

    def load_mask(self, image_id):

        info = self.image_info[image_id]
        json_path = info["path"].parent / (info["path"].stem + ".json")
        with open(json_path) as f:
            d = json.load(f)
            data = pd.DataFrame.from_dict(d["microtubule"])

        def get_line(x):
            d = {}
            d["start_x"] = x[x.type == "seed"]["start_x"].values[0]
            d["start_y"] = x[x.type == "seed"]["start_y"].values[0]
            d["end_x"] = x[x.type == "seed"]["end_x"].values[0]
            d["end_y"] = x[x.type == "seed"]["end_y"].values[0]
            return pd.DataFrame([d])

        def draw_line(image, line, line_thickness):
            line = np.round(line).astype("int16")
            p1 = (line["start_x"], line["start_y"])
            p2 = (line["end_x"], line["end_y"])
            _, p1, p2 = cv2.clipLine((0, 0, image.shape[0], image.shape[1]), p1, p2)
            image = cv2.line(image, p1, p2, (1,), line_thickness)
            return image

        lines = data.groupby("mt_id").apply(get_line).reset_index(drop=True)
        count = lines.shape[0]

        mask = np.zeros([info['width'], info['height'], count], dtype=np.uint8)
        for i, line in lines.iterrows():
            mask[:, :, i] = draw_line(mask[:, :, i].copy(), line, info["line_thickness"])

        # Handle occlusions
        handle_occlusion = True
        if handle_occlusion:
            occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
            for i in range(count - 2, -1, -1):
                mask[:, :, i] = mask[:, :, i] * occlusion
                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # Map class names to class IDs (all "microtubule" here).
        class_ids = np.repeat(self.class_names.index("microtubule"), count)

        return mask.astype(np.bool), class_ids.astype(np.int32)

    
    def random_display(self, n=4):
        # Load and display random samples
        image_ids = np.random.choice(self.image_ids, n, replace=True)
        for image_id in image_ids:
            print(self.image_info[image_id])
            image = self.load_image(image_id)
            mask, class_ids = self.load_mask(image_id)
            visualize.display_top_masks(image, mask, class_ids, self.class_names, limit=1)