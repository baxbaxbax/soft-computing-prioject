import load_data as ld
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import serialize


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx * ny))


def hog_it(img):
    pos_features = []
    neg_features = []
    labels = []

    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    pos_features = []
    neg_features = []
    labels = []

    for img in signs:
        pos_features.append(hog.compute(img))
        labels.append(1)

    for img in neg_signs:
        neg_features.append(hog.compute(img))
        labels.append(0)

    pos_features = np.array(pos_features)
    neg_features = np.array(neg_features)
    x = np.vstack((pos_features, neg_features))
    y = np.array(labels)

    return hog, x, y


def hog_it(img):
    pos_features = []

    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    pos_features = []

    for img in signs:
        pos_features.append(hog.compute(img))

    pos_features = np.array(pos_features)

    x = np.array(pos_features)
    y = np.array(pos_labels)

    return hog, x, y


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
hog, x, y = hog_it(signs[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = reshape_data(x_train)
x_test = reshape_data(x_test)

print('Train shape: ', x_train.shape, y_train.shape)
print('Test shape: ', x_test.shape, y_test.shape)

yesno_svm = serialize.load_trained_svm("yesno")

# ako je ann=None, znaci da model nije ucitan u prethodnoj metodi i da je potrebno istrenirati novu mrezu
if yesno_svm is None:
    print("Training model")
    yesno_svm = SVC(kernel='linear', probability=True)
    yesno_svm.fit(x_train, y_train)
    # clf_svm = LinearSVC()
    # clf_svm.fit(x_train, y_train)
    serialize.serialize_svm(yesno_svm, "yesno")
else:
    print("Model loaded")

y_train_pred = yesno_svm.predict(x_train)
y_test_pred = yesno_svm.predict(x_test)
print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

# hog for signs
hog_signs, x_signs, y_signs = hog_it(signs[0])

x_train, x_test, y_train, y_test = train_test_split(x_signs, y_signs, test_size=0.2, random_state=42)
x_train = reshape_data(x_train)
x_test = reshape_data(x_test)

print('Train shape: ', x_train.shape, y_train.shape)
print('Test shape: ', x_test.shape, y_test.shape)

signs_svm = serialize.load_trained_svm("signs")

# ako je ann=None, znaci da model nije ucitan u prethodnoj metodi i da je potrebno istrenirati novu mrezu
if signs_svm is None:
    print("Training model")
    signs_svm = SVC(kernel='linear', probability=True)
    signs_svm.fit(x_train, y_train)
    # clf_svm = LinearSVC()
    # clf_svm.fit(x_train, y_train)
    serialize.serialize_svm(signs_svm, "signs")
else:
    print("Model loaded")

y_train_pred = signs_svm.predict(x_train)
y_test_pred = signs_svm.predict(x_test)
print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))


for image in images:
    detections = [[0, 0, 0, 0, "DRUGI"]]  # x_min, y_min, x_max, y_max, tip znaka

    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    hog.setSVMDetector(yesno_svm)
    (rects, weights) = hog.detectMultiScale(image)
    rects = non_max_suppression_fast(rects, 0.3)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
    ld.preview_img(image)
