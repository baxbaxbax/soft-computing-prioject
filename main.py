import load_data as ld
import cv2
import numpy as np
import train_hog
from PIL import Image


def generate_data(rectangles):
    for (x, y, w, h) in rectangles:
        img = image.copy()
        image2d = np.array(ld.extract_roi(img, [x, y, w, h]))
        if image2d.shape[0] == 0 or image2d.shape[1] == 0:
            continue
        im = Image.fromarray(image2d)
        im.save("generated_false_signs/sign" + str(index) + ".png")
        # scipy.misc.imsave("generated_false_signs/sign" + str(index) + ".png", image2d)
        # png.from_array(image2d, mode='L').save("generated_false_signs/sign" + str(index) + ".png")

def crop_image(input_image, output_image, start_x, start_y, width, height):
    """Pass input name image, output name image, x coordinate to start croping, y coordinate to start croping, width to crop, height to crop """
    input_img = Image.open(input_image)
    box = (start_x, start_y, start_x + width, start_y + height)
    output_img = input_img.crop(box)
    output_img.save(output_image +".png")


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

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

    ld.preview_img(image)
