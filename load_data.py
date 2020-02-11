import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

FOLDER_PATH = "CADDY_gestures_complete_v2_release"
ROIS_PATH = "hands-ROIs"
ALL_TRUE_CSV = "CADDY_gestures_all_true_positives_release_v2.csv"
ALL_NEG_CSV = "CADDY_gestures_all_true_negatives_release_v2.csv"


def load_img(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    if image is None:
        raise Exception("could not load image !")
    return image


def preview_img(image):
    plt.imshow(image, 'gray')
    plt.show()
    plt.close()


def resize(image):
    return cv2.resize(image, (64, 64))


def extract_roi(img, coords):
    return img[coords[0]: coords[0] + coords[2], coords[1]: coords[1] + coords[3]]


def get_array(param):
    if "[" in param[0]:
        first = param[0].split("[")[1]
    else:
        first = param[0].split(";")[1]
    if "]" in param[3]:
        last = param[3].split("]")[0]
    else:
        last = param[3].split(";")[0]
    return [int(first), int(param[1]), int(param[2]), int(last)]


def get_array_mosaic(param):
    first = param[0].split("[")[1]
    last = param[3].split(";")[0]
    return [int(first), int(param[1]), int(param[2]), int(last)]


def get_array_mosaic_end(param):
    first = param[0].split(";")[1]
    last = param[3].split("]")[0]
    return [int(first), int(param[1]), int(param[2]), int(last)]


def load_positive_csv():
    signs = []
    labels = []
    images = []
    neg_signs = []
    with open(ALL_TRUE_CSV) as file:
        lines = file.readlines()
        for line in lines[1:]:
            cols = line.replace("\n", "").split(",")
            if "raw" in cols[2]:
                if cols[1] == "biograd-A" or cols[1] == "biograd-B":
                    try:
                        img_left = load_img(FOLDER_PATH + cols[2])
                        roi_left = load_img(ROIS_PATH + cols[2])
                        # img_right = load_img(FOLDER_PATH + cols[3])
                        roi_right = load_img(ROIS_PATH + cols[3])
                        signs.append(resize(roi_left))
                        signs.append(resize(roi_right))
                        labels.append(cols[5])
                        labels.append(cols[5])
                        images.append(img_left)
                        # images.append(img_right)

                        neg_signs.append(get_array(cols[6:10]))
                        if cols[5] == 9 or cols[4] == "mosaic":
                            neg = get_array(cols[9:13])
                        else:
                            neg = get_array(cols[10:14])
                        neg_signs.append(neg)
                    except:
                        raise Exception("Could not load images")

    signs = np.array(signs)
    images = np.array(images)
    labels = np.array(labels)
    neg_signs = np.array(neg_signs)
    print(signs.shape)
    print(images.shape)
    print(labels.shape)
    print(neg_signs.shape)

    return signs, labels, neg_signs, images


def load_negative_csv(neg_rois):
    neg_signs = []
    labels = []
    # images = []

    with open(ALL_NEG_CSV) as file:
        lines = file.readlines()
        index = 0
        for line in lines[1:]:
            cols = line.replace("\n", "").split(",")
            if "raw" in cols[2]:
                if cols[1] == "biograd-A" or cols[1] == "biograd-B":
                    try:
                        img_left = load_img(FOLDER_PATH + cols[2])
                        img_right = load_img(FOLDER_PATH + cols[3])

                        neg_signs.append(resize(extract_roi(img_left, neg_rois[index])))
                        neg_signs.append(resize(extract_roi(img_right, neg_rois[index])))
                        labels.append(cols[5])
                        labels.append(cols[5])

                        # images.append(img_left)
                        # images.append(img_right)
                        index += 1
                    except:
                        raise Exception("Could not load images")

    # images = np.array(images)
    labels = np.array(labels)
    neg_signs = np.array(neg_signs)
    # print(images.shape)
    print(labels.shape)
    print(neg_signs.shape)
    return labels, neg_signs
