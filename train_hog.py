from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import serialize
import numpy as np
import cv2


def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx * ny))


def hog_it(img, signs, neg_signs):
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

    return yesno_svm, x, y, hog


def hog_signs(img, signs, pos_labels):
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

    for sign in signs:
        pos_features.append(hog.compute(sign))

    x = np.array(pos_features)
    y = np.array(pos_labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

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

    return signs_svm, x, y