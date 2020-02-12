import load_data as ld
import cv2
import numpy as np
import train_hog
from PIL import Image


font = cv2.FONT_HERSHEY_SIMPLEX


def label(str):
    operation = int(str)
    switcher = {
        0: 'start_comm',
        1: 'end_comm',
        2: 'up',
        3: 'down',
        4: 'photo',
        5: 'backwards',
        6: 'carry',
        7: 'boat',
        8: 'here',
        9: 'mosaic',
        10: 'num_delimiter',
        11: 'one',
        12: 'two',
        13: 'three',
        14: 'four',
        15: 'five'
    }
    return switcher.get(operation)


def generate_data(rectangles):
    for (x, y, w, h) in rectangles:
        img = image.copy()
        image2d = np.array(ld.extract_roi(img, [x, y, w, h]))
        if image2d.shape[0] == 0 or image2d.shape[1] == 0:
            continue
        image2d = ld.resize(image2d)
        im = Image.fromarray(image2d)
        im.save("generated_false_signs/sign" + str(index) + ".png")


def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


# load data
signs, pos_labels, neg_rois, images = ld.load_positive_csv()
neg_labels, neg_signs = ld.load_negative_csv(neg_rois)

# hog for hand or not
svm, x, y, hog = train_hog.hog_it(signs[0], signs, neg_signs)

# hog for signs
svm_signs, x_signs, y_signs = train_hog.hog_signs(signs[0], signs, pos_labels)

for index, image in enumerate(images):

    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    hog.setSVMDetector(svm.coef_)
    (rects, weights) = hog.detectMultiScale(image)
    rects = non_max_suppression_fast(rects, 0.3)

    # if index > 1000:
    #     generate_data(rects)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
        img = image.copy()
        image2d = ld.resize(np.array(ld.extract_roi(img, [x, y, w, h])))
        if image2d == []:
            continue
        data = hog.compute(image2d)
        data_tr = data.transpose()
        cv2.putText(image, label(svm_signs.predict(data_tr)), (x, y - 5), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        ld.preview_img(image)
