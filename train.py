import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.io, scipy.misc
from skimage import io
from skimage import transform as tf
from skimage.color import rgb2hsv
from sklearn import svm

def visualize_distributions(new_dir):
    """
    Visualize the distributions of values in HSV and BGR of pictures in new_dir.
    """
    H, S, V = None, None, None
    B, G, R = None, None, None
    for filename in os.listdir(new_dir):
            if not filename.endswith(".png"):
                continue
            img = cv2.imread(new_dir + filename)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if H is None:
                H = hsv_img[..., 0].flatten()
                S = hsv_img[..., 1].flatten()
                V = hsv_img[..., 2].flatten()
                B = img[..., 0].flatten()
                G = img[..., 1].flatten()
                R = img[..., 2].flatten()
            else:
                H = np.concatenate([H, hsv_img[..., 0].flatten()])
                S = np.concatenate([S, hsv_img[..., 1].flatten()])
                V = np.concatenate([V, hsv_img[..., 2].flatten()])
                B = np.concatenate([B, img[..., 0].flatten()])
                G = np.concatenate([G, img[..., 1].flatten()])
                R = np.concatenate([R, img[..., 2].flatten()])

    # Plot
    f = plt.figure()
    ax1 = f.add_subplot(1, 3, 1)
    ax1.hist(H, bins=180,
        range=(0.0, 180.0), histtype="stepfilled", color="b", label="Hue")
    plt.title("Hue")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    ax2 = f.add_subplot(1, 3, 2)
    ax2.hist(S, bins=256,
        range=(0.0, 255.0),histtype="stepfilled", color="g", label="Saturation")
    plt.title("Saturation")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    ax3 = f.add_subplot(1, 3, 3)
    ax3.hist(V, bins=256,
        range=(0.0, 255.0), histtype="stepfilled", color="r", label="Value")
    plt.title("Value")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    f.tight_layout()
    f.savefig(new_dir + "_hsv_distributions.png")
    plt.show()

    f = plt.figure()
    ax1 = f.add_subplot(1, 3, 1)
    ax1.hist(B, bins=256,
        range=(0.0, 255.0), histtype="stepfilled", color="b", label="Blue")
    plt.title("Blue")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    ax2 = f.add_subplot(1, 3, 2)
    ax2.hist(G, bins=256,
        range=(0.0, 255.0),histtype="stepfilled", color="g", label="Saturation")
    plt.title("Green")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    ax3 = f.add_subplot(1, 3, 3)
    ax3.hist(R, bins=256,
        range=(0.0, 255.0), histtype="stepfilled", color="r", label="Red")
    plt.title("Red")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    f.tight_layout()
    f.savefig(new_dir + "_rgb_distributions.png")
    plt.show()

def crop_images(old_dir, new_dir):
    """
    Crop .jpg images in old_dir given coordinates of
    left eye, right eye, nose and mouth in .mat files.
    Save cropped images in new_dir.
    """
    print("crop_images")
    os.mkdir(new_dir)

    H, S, V = None, None, None
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

def warp_face(input_dir, output_dir):
    """
    Warp images so faces are front facing.
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jpg"):
            continue
        print(filename)
        img = io.imread(os.path.join(input_dir, filename))
        index = filename.find(".jpg")
        name = filename[:index]
        # Feature coordiantes
        coords = scipy.io.loadmat(os.path.join(input_dir, name + ".mat"))

        x = coords["x"].reshape(-1)
        y = coords["y"].reshape(-1)
        # New feature coordinates
        x2 = [104, 153, 126, 125]
        y2 = [114, 114, 141, 162]

        # Compute homography
        src = np.vstack((x, y)).T
        dst = np.vstack((x2, y2)).T
        H = tf.estimate_transform("projective", src, dst)
        warped = tf.warp(img, inverse_map=H.inverse)

        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(img)
        f.add_subplot(1,2, 2)
        plt.imshow(warped)
        plt.plot(x2, y2, "r")
        plt.show()

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

def train_model():
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

    # model = svm.LinearSVC(C=1.0) # 0.4947
    # model = svm.LinearSVC(C=10.0) # 0.5370
    model = svm.LinearSVC(C=50.0) # 0.5200
    # model = svm.LinearSVC(C=100.0) # 0.5137
    # model = svm.SVC(kernel="rbf", C=1.0) # 0.4691
    # model = svm.SVC(kernel="rbf", gamma="scale", C=1.0) # 0.6106
    # model = svm.SVC(kernel="rbf", gamma="scale", C=10.0) # 0.6221
    # model = svm.SVC(kernel="rbf", gamma="scale", C=100.0) # 0.61195
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

def face_detection(input_dir, output_dir):
    print("face_detection")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jpg"):
            continue

        img = cv2.imread(os.path.join(input_dir, filename))

        # HSV color detection
        # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # lower1 = np.array([0, 48, 80])
        # upper1 = np.array([20, 255, 255])
        # mask1 = cv2.inRange(hsv_img, lower1, upper1) # img_hsv.shape
        # lower2 = np.array([170, 0, 0])
        # upper2 = np.array([180, 255, 255])
        # mask2 = cv2.inRange(hsv_img, lower2, upper2) # img_hsv.shape
        # mask = cv2.bitwise_or(mask1, mask2)

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        # # mask = cv2.erode(mask, kernel, iterations=2)
        # # mask = cv2.dilate(mask, kernel, iterations=2)
        # # mask = cv2.GaussianBlur(mask, (3, 3), 0)
        # new_img = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imwrite(os.path.join(output_dir, filename), new_img)
