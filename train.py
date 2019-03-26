import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.io, scipy.misc
from skimage import io
from sklearn import svm

def crop_images(old_dir, new_dir):
    """
    Crop .jpg images in old_dir given coordinates of
    left eye, right eye, nose and mouth in .mat files.
    Save cropped images in new_dir.
    """
    print("crop_images")
    os.mkdir(new_dir)

    for filename in os.listdir(old_dir):
        if not filename.endswith(".jpg"):
            continue

        index = filename.find(".jpg")
        name = filename[:index]

        # Approximate coordinates of face
        coords = scipy.io.loadmat(old_dir + name + ".mat")
        start_x = int(coords["x"][0][0] - 0.5*(coords["x"][1][0] - coords["x"][0][0]))
        end_x = int(coords["x"][1][0] + 0.5*(coords["x"][1][0] - coords["x"][0][0]))
        start_y = int(coords["y"][0][0] - (coords["y"][3][0] - coords["y"][0][0]))
        end_y = int(coords["y"][3][0] + (coords["y"][3][0] - coords["y"][2][0]))
        img = io.imread(old_dir + filename)
        face = img[start_y:end_y, start_x:end_x]
        # Save cropped image
        scipy.misc.imsave(new_dir + name + ".png", face)

def get_features(input_dir):
    """
    Get keypoints and descriptors of the images in input_dir with SIFT.
    """
    kps = None
    des = None
    sift = cv2.xfeatures2d.SIFT_create()
    for filename in os.listdir(input_dir):
        if not filename.endswith(".png"):
            continue
        img = io.imread(input_dir + filename)
        kp, d = sift.detectAndCompute(img, None)
        # kps.append(kp)
        # des.append(d)
        if des is None:
            kps = kp
            des = d
        else:
            kps = np.concatenate([kps, kp], axis=0)
            des = np.concatenate([des, d], axis=0)
    return kps, des

def train_model(deg):
    OLD_F_DIR = "original_data/female/"
    OLD_M_DIR = "original_data/male/"
    F_TRAIN_DIR = "data/female_train/"
    M_TRAIN_DIR = "data/male_train/"
    MODEL_NAME = "svm_model.sav"

    if not os.path.isdir("data/"):
        os.mkdir("data/")
        crop_images(OLD_F_DIR, F_TRAIN_DIR)
        crop_images(OLD_M_DIR, M_TRAIN_DIR)

    f_train_kps, f_train_des = get_features(F_TRAIN_DIR)
    m_train_kps, m_train_des = get_features(M_TRAIN_DIR)
    x_train = np.concatenate([f_train_des, m_train_des], axis=0)
    y_train = [-1] * len(f_train_des) + [1] * len(m_train_des)

    model = svm.SVC(kernel="rbf", gamma="scale", C=10.0)
    print("Start training")
    model.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(MODEL_NAME, "wb"))
    print("Model saved as " + MODEL_NAME)

def predict():
    F_TEST_DIR = "data/female_test/"
    M_TEST_DIR = "data/male_test/"
    MODEL_NAME = "svm_model.sav"

    model = pickle.load(open(MODEL_NAME, "rb"))

    f_test_kps, f_test_des = get_features(F_TEST_DIR)
    m_test_kps, m_test_des = get_features(M_TEST_DIR)
    x_test = np.concatenate([f_test_des, m_test_des], axis=0)
    y_test = [-1] * len(f_test_des) + [1] * len(m_test_des)

    result = model.score(x_test, y_test)
    print(result)

def face_detection(input_dir):
