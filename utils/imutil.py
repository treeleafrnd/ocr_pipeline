import os
import sys

import numpy as np
import cv2


def load_images(crop_path):
    img_paths = []
    img_names = []
    # Arranging words
    for img in os.listdir(crop_path):
        img = img.split('.')[0]
        img_names.append(img)
    img_names.sort(key=int)
    # Joining full image path
    for items in img_names:
        if crop_path[-1] != '/':
            image_path = crop_path + '/' + items + '.jpg'
        else:
            image_path = crop_path + img + '.jpg'
        img_paths.append(image_path)
    return img_paths


# Sequencing the words
def word_sequencer(bbox):
    diff = []
    ratio = 3  # Diving ratio to cover portion of the sentence in y-axis (Lower covers more)
    # Arranging lines
    for k in range(len(bbox)):
        for i in range(len(bbox) - 1):
            if bbox[i][3] > bbox[i + 1][3]:
                temp = bbox[i]
                bbox[i] = bbox[i + 1]
                bbox[i + 1] = temp
    # Gap between sentences
    for i in range(len(bbox) - 1):
        diff.append(abs(bbox[i][3] - bbox[i + 1][3]))
    # Range within which a sentence is covered
    if len(bbox) > 1:
        divider = np.max(diff) // ratio
    else:
        divider = 0
    # Arranging words
    for k in range(len(bbox)):
        for i in range(len(bbox) - 1):
            if (bbox[i][0] > bbox[i + 1][0]) and (
                    (bbox[i + 1][3] - divider) <= bbox[i][3] <= (
                    bbox[i + 1][3] + divider)):
                temp = bbox[i]
                bbox[i] = bbox[i + 1]
                bbox[i + 1] = temp
    return bbox


# Writing crops into directory
def cropper(bbox, original_image):
    # TODO checking for empty bbox
    counter = 0
    dir = 'data/crops'
    original_image = cv2.imread(original_image)
    for i in range(len(bbox)):
        counter = counter + 1
        crops = original_image[bbox[i][1]:bbox[i][3], bbox[i][0]:bbox[i][2]]
        if not cv2.imwrite(f"{dir}/{counter}.jpg", crops):
            raise Exception("Could not write image")
    if counter != len(os.listdir(dir)):
        return False
    else:
        return True


# Emptying crop folder
def empty_folder(crop_path):
    for images in os.listdir(crop_path):
        os.remove(crop_path + '/' + images)
