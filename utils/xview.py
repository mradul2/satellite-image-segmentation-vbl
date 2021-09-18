import json
import os

import cv2
from cv2 import fillPoly
import numpy as np
from shapely.geometry import mapping



def generate_localization_polygon(json_path):
    with open(json_path, "r") as f:
        annotations = json.load(f)
    h = annotations["metadata"]["height"]
    w = annotations["metadata"]["width"]
    mask_img = np.zeros((h, w), np.uint8)
    for feat in annotations['features']['xy']:
        feat_shape = wkt.loads(feat['wkt'])
        coords = list(mapping(feat_shape)['coordinates'][0])
        fillPoly(mask_img, [np.array(coords, np.int32)], (1))
    return mask_img


def generate_damage_polygon(json_path):
    with open(json_path, "r") as f:
        annotations = json.load(f)

    h = annotations["metadata"]["height"]
    w = annotations["metadata"]["width"]
    mask_img = np.zeros((h, w), np.uint8)

    damage_dict = {
        "no-damage": 1,
        "minor-damage": 2,
        "major-damage": 3,
        "destroyed": 4,
        "un-classified": 5
    }
    for feat in annotations['features']['xy']:
        feat_shape = wkt.loads(feat['wkt'])
        coords = list(mapping(feat_shape)['coordinates'][0])
        fillPoly(mask_img, [np.array(coords, np.int32)], damage_dict[feat['properties']['subtype']])
    return mask_img
