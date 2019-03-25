import cv2
import matplotlib.pyplot as plt
import os
import scipy.io, scipy.misc
from skimage import io

def crop_images(old_dir, new_dir):
    """
    Crop .jpg images in old_dir given coordinates of
    left eye, right eye, nose and mouth in .mat files.
    Save cropped images in new_dir.
    """
    print("crop_images")
    # os.mkdir(new_dir)

    for filename in os.listdir(old_dir):
        if not filename.endswith(".jpg"):
            continue

        index = filename.find(".jpg")
        name = filename[:index]

        # Approximate coordinates of face
        coords = scipy.io.loadmat(old_dir + name + ".mat")
        # print(coords)
        # print(name, coords["x"][1][0], 0.5*(coords["x"][0][0] - coords["x"][1][0]))
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
    kps = []
    des = []
    sift = cv2.xfeatures2d.SIFT_create()
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jpg"):
            continue
        img = cv2.imread(img_name)
        kp, d = sift.detectAndCompute(img, None)
        kps.append(kp)
        des.append(d)
    return kps, des

def train_model(**kwargs):
    print("train_model")
    OLD_F_DIR = "original_data/female/"
    OLD_M_DIR = "original_data/male/"
    NEW_F_DIR = "data/female/"
    NEW_M_DIR = "data/male/"

    # if not os.path.isdir("data/"):
        # os.mkdir("data/")
    crop_images(OLD_F_DIR, NEW_F_DIR)
    crop_images(OLD_M_DIR, NEW_M_DIR)

    f_kps, f_des = get_features(NEW_F_DIR)
    m_kps, m_des = get_features(NEW_M_DIR)
