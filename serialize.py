from joblib import dump, load


def serialize_svm(clf_svm, name):
    with open("serialization_folder/" + name + ".json", "w"):
        dump(clf_svm, open("serialization_folder/" + name + ".json", 'wb'))


def load_trained_svm(name):
    try:
        clf_svm = load("serialization_folder/" + name + ".json")
        return clf_svm
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None

