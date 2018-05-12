import colorsys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cv2
import tqdm
from skimage.measure import find_contours

from ipywidgets import widgets
from ipywidgets import interact
from ipywidgets import fixed


def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def show_images(images_list, lines=None, size=14, cmap="viridis"):

    if not isinstance(images_list, list):
        images_list = [images_list]

    axs = get_ax(rows=1, cols=len(images_list), size=size)
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    else:
        axs = axs.flat
    fig = axs[0].get_figure()
    plt.close(fig)

    def show(t):   
        for i, ax in enumerate(axs):
            ax.clear()
            ax.imshow(images_list[i][t], cmap=cmap)
            
            if isinstance(lines, pd.DataFrame):
                frame_data = lines[lines.frame == t]
                for i, row in frame_data.iterrows():
                    ax.plot(row["points"][:, 0], row["points"][:, 1], lw=2)
            
        fig = axs[0].get_figure()
        display(fig)

    interact(show, t=widgets.IntSlider(value=0, min=0, max=images_list[0].shape[0] - 1))
    
    
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    np.random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def get_masked_fixed_color(image, boxes, masks, class_ids, class_names,
                           colors = None, scores=None, title="",
                           draw_boxes=False, draw_masks=False,
                           draw_contours=False, draw_score=False):

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    if colors == None:
        classN = len(class_names)
        colors = random_colors(classN)

    masked_image = np.array(image)

    for i in range(N):
        color = colors[class_ids[i]]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if draw_boxes:
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), color, thickness=1)

        # Label
        if draw_score:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = np.random.randint(x1, (x1 + x2) // 2)
            caption = f"{score:.3f}" if score else label
            cv2.putText(masked_image, caption, (x1 + 5, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.2, color)

        # Mask
        mask = masks[:, :, i]
        if draw_masks:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if draw_contours:
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                verts = verts.reshape((-1, 1, 2)).astype(np.int32)
                # Draw an edge on object contour
                cv2.polylines(masked_image, verts, True, color)

    return masked_image


def draw_results(image, results, class_names,
                 resize_ratio=1, colors=None,
                 draw_boxes=False, draw_masks=True,
                 draw_contours=True, draw_score=True):
    
    masked_image_batch = []
    
    if not colors:
        colors = random_colors(len(class_names))
        colors = [(0, 255, 255), (255, 0, 0)]

    n_results = len(results)
    for i in tqdm.tqdm(range(n_results), total=n_results):
        r = results[i]
        im = image[i]
        masked_image = get_masked_fixed_color(im, r['rois'], r['masks'], r['class_ids'],
                                              class_names, colors, r['scores'],
                                              draw_boxes=draw_boxes, draw_masks=draw_masks,
                                              draw_contours=draw_contours, draw_score=draw_score)
        masked_image = cv2.resize(masked_image, None, interpolation=cv2.INTER_NEAREST,
                                  fx=resize_ratio, fy=resize_ratio)
        masked_image_batch.append(masked_image)

    masked_image_batch = np.array(masked_image_batch)
    return masked_image_batch
