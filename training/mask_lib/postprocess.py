import pandas as pd
import numpy as np
from scipy.spatial import distance
import cv2


def find_filament(mask):
    """Find a filament from a binary mask"""
    
    # Get the contours of all the blob objects
    # We assume all o them are one single microtubule
    mask = mask.astype("uint8")
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Merge all the contours together
    contours = np.concatenate([contour for contour in contours])

    # We select the two most distant from each other points on this contour.
    # It defines the line describing the filament.
    d = distance.squareform(distance.pdist(contours[:, 0, :]))
    p1_index, p2_index = np.unravel_index(d.argmax(), d.shape)
    p1 = contours[p1_index][0]
    p2 = contours[p2_index][0]
    p = np.array([p1, p2])
    return p


def find_filaments(results):
    data = []
    filament_id = 0
    for frame, frame_result in enumerate(results):
        for i in range(frame_result["masks"].shape[-1]):
            mask = frame_result["masks"][:, :, i]
            points = find_filament(mask)

            datum = {}
            datum["id"] = filament_id
            filament_id += 1
            datum["frame"] = frame
            datum["points"] = points
            data.append(datum)

    data = pd.DataFrame(data)
    return data


def filter_results(results, min_diameter):
    filtered_results = []
    for frame, frame_result in enumerate(results):
        index_to_keep = []
        for i in range(frame_result["masks"].shape[-1]):
            mask = frame_result["masks"][:, :, i]

            mask_diameter = np.sum(mask == True)
            if mask_diameter > min_diameter:
                index_to_keep.append(i)

        new_frame_result = {}
        for k, v in frame_result.items():
            if k == "masks":
                new_frame_result[k] = v[:, :, index_to_keep]
            else:
                new_frame_result[k] = v[index_to_keep]

        filtered_results.append(new_frame_result)
    return filtered_results
