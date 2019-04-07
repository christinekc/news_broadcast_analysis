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

def visualize_distributions(imgs_dir):
    """
    Visualize the distributions of values in HSV and BGR of pictures in imgs_dir.
    """
    H, S, V = None, None, None
    B, G, R = None, None, None
    for filename in os.listdir(imgs_dir):
            if not filename.endswith(".png"):
                continue
            img = cv2.imread(imgs_dir + filename)
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
    f.savefig(imgs_dir + "_hsv_distributions.png")
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
    f.savefig(imgs_dir + "_rgb_distributions.png")
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

def resize_images(input_dir, output_dir, size=72):
    """
    Resize images in input_dir to size x size.
    """
    print("resize_images")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".png"):
            continue
        img = cv2.imread(os.path.join(input_dir, filename))
        x, y, d = img.shape
        # Pad image to square
        if x > y:
            padded_img = np.pad(img, ((0, 0), (0, x - y), (0, 0)),
                mode="constant", constant_values=0)
        else:
            padded_img = np.pad(img, ((0, y - x), (0, 0), (0, 0)),
                mode="constant", constant_values=0)
        resized_img = cv2.resize(padded_img, (size, size))
        scipy.misc.imsave(output_dir + filename, cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

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
    kps, des = None, None
    sift = cv2.xfeatures2d.SIFT_create()

    for filename in os.listdir(input_dir):
        if not filename.endswith(".png"):
            continue
        img = io.imread(input_dir + filename)
        kp, d = sift.detectAndCompute(img, None)

        if des is None:
            kps = kp
            des = d
        else:
            kps = np.concatenate([kps, kp], axis=0)
            des = np.concatenate([des, d], axis=0)
    return kps, des

def get_data(f_dir, m_dir):
    x = []
    y = [-1] * len(os.listdir(f_dir)) + [1] * len(os.listdir(m_dir))
    imgs = os.listdir(f_dir) + os.listdir(m_dir)
    for filename in os.listdir(f_dir):
        if not filename.endswith(".png"):
            continue
        x.append(cv2.imread(os.path.join(f_dir, filename)))
    for filename in os.listdir(m_dir):
        if not filename.endswith(".png"):
            continue
        x.append(cv2.imread(os.path.join(m_dir, filename)))

    # Shuffle
    assert len(x) == len(y)
    idx = np.random.permutation(len(x))
    x, y = np.array(x)[idx], np.array(y)[idx]
    return x, y

def train_model(model_path, classification):
    OLD_F_DIR = "original_data/female/"
    OLD_M_DIR = "original_data/male/"
    F_TRAIN_DIR = "data/female_train/"
    M_TRAIN_DIR = "data/male_train/"
    F_TEST_DIR = "data/female_test/"
    M_TEST_DIR = "data/male_test/"
    F_TRAIN_CNN_DIR = "data/female_cnn_train/"
    M_TRAIN_CNN_DIR = "data/male_cnn_train/"
    F_TEST_CNN_DIR = "data/female_cnn_test/"
    M_TEST_CNN_DIR = "data/male_cnn_test/"

    if not os.path.isdir("data/"):
        os.mkdir("data/")
        crop_images(OLD_F_DIR, F_TRAIN_DIR)
        crop_images(OLD_M_DIR, M_TRAIN_DIR)

    if classification == "SVM":
        f_train_kps, f_train_des = get_features(F_TRAIN_DIR)
        m_train_kps, m_train_des = get_features(M_TRAIN_DIR)
        x_train = np.concatenate([f_train_des, m_train_des], axis=0)
        y_train = [-1] * len(f_train_des) + [1] * len(m_train_des)

        model = svm.SVC(kernel="rbf", gamma="scale", C=10.0)
        print("Start training")
        model.fit(x_train, y_train)

        # Save model
        pickle.dump(model, open(model_path, "wb"))
        print("Model saved as " + model_path)

    elif classification == "CNN":
        import keras
        from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
        from keras.models import Sequential
        from keras.optimizers import SGD

        np.random.seed(123)
        size = 72

        # resize_images(F_TRAIN_DIR, F_TRAIN_CNN_DIR, size)
        # resize_images(M_TRAIN_DIR, M_TRAIN_CNN_DIR, size)
        # resize_images(F_TEST_DIR, F_TEST_CNN_DIR, size)
        # resize_images(M_TEST_DIR, M_TEST_CNN_DIR, size)

        x_train, y_train = get_data(F_TRAIN_CNN_DIR, M_TRAIN_CNN_DIR)
        y_train = keras.utils.to_categorical(y_train, num_classes=2)
        x_test, y_test = get_data(F_TEST_CNN_DIR, M_TEST_CNN_DIR)
        y_test = keras.utils.to_categorical(y_test, num_classes=2)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(size, size, 3)))
        model.add(Conv2D(32, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation="softmax"))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

        model.fit(x_train, y_train, epochs=3)
        score = model.evaluate(x_test, y_test)
        print(score)

        model.save(model_path)

    else:
        raise ValueError("Illegal classification value")

def predict_model(model_path):
    print("predict_model")
    F_TEST_DIR = "data/female_test/"
    M_TEST_DIR = "data/male_test/"

    model = pickle.load(open(model_path, "rb"))

    f_test_kps, f_test_des = get_features(F_TEST_DIR)
    m_test_kps, m_test_des = get_features(M_TEST_DIR)
    x_test = np.concatenate([f_test_des, m_test_des], axis=0)
    y_test = [-1] * len(f_test_des) + [1] * len(m_test_des)

    result = model.score(x_test, y_test)
    print(result)

def face_detection_hsv(input_dir, output_dir):
    print("face_detection_hsv")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(input_dir, filename))

        # HSV color detection
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 48, 80])
        upper1 = np.array([20, 255, 255])
        mask1 = cv2.inRange(hsv_img, lower1, upper1) # img_hsv.shape
        lower2 = np.array([170, 0, 0])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_img, lower2, upper2) # img_hsv.shape
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        new_img = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(os.path.join(output_dir, filename), new_img)

def face_detection_cascade(input_dir, output_dir, model_path, classification):
    print("face_detection_cascade")
    if classification == "SVM":
        model = pickle.load(open(model_path, "rb"))
        sift = cv2.xfeatures2d.SIFT_create()
    elif classification == "CNN":
        from keras.models import load_model
        model = load_model(model_path)
        size = 72
    else:
        raise ValueError("Illegal classification value")

    XML_FILENAME = "haarcascade_frontalface_default.xml"
    F_TEXT = "Female"
    M_TEXT = "Male"
    O_TEXT = "Not sure"
    F_COLOR = (0, 0, 255)
    M_COLOR = (255, 0, 0)
    O_COLOR = (0, 255, 0)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(input_dir, filename))
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Cascade face detection
        face_cascade = cv2.CascadeClassifier(XML_FILENAME)
        faces = face_cascade.detectMultiScale(img_g, scaleFactor=1.3,
            minNeighbors=5, minSize=(30, 30))

        if classification == "SVM":
            model = pickle.load(open(model_path, "rb"))
            sift = cv2.xfeatures2d.SIFT_create()
            for (x, y, w, h) in faces:
                # Get SIFT descriptors
                kps, des = sift.detectAndCompute(img[y:y+h, x:x+w], None)
                # Predict female or male
                result = model.predict(des)
                f_count = np.count_nonzero(result == -1)
                m_count = np.count_nonzero(result == 1)
                # Plot color box
                if f_count > m_count:
                    text = F_TEXT + ": " + format(f_count/(f_count + m_count)*100, ".2f") + "%"
                    color = F_COLOR
                elif m_count > f_count:
                    text = M_TEXT + ": " + format(m_count/(f_count + m_count)*100, ".2f") + "%"
                    color = M_COLOR
                else:
                    text = O_TEXT
                    color = O_COLOR
                cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=2)
                cv2.putText(img, text, (x, y-10), color=color,
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1)

        elif classification == "CNN":
            for (x, y, w, h) in faces:
                assert w == h
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (size, size))
                prediction = model.predict_classes(np.array([face]), verbose=0)
                if prediction == -1:
                    text = F_TEXT + ": " + format(prediction[0]*100, ".2f") + "%"
                    color = F_COLOR
                elif prediction == 1:
                    text = M_TEXT + ": " + format(prediction[0]*100, ".2f") + "%"
                    color = M_COLOR
                else:
                    text = O_TEXT
                    color = O_COLOR
                cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=2)
                cv2.putText(img, text, (x, y-10), color=color,
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1)

        cv2.imwrite(os.path.join(output_dir, filename), img)
