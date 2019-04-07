import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def shot_detection(input_dir, type):
    scores = []
    img_names = [img for img in os.listdir(input_dir) if img.endswith(".jpg")]
    # Sort images by name in ascending order
    img_names.sort(key=lambda img: int("".join(filter(str.isdigit, img))))
    start_idx = int(img_names[0][:img_names[0].find(".jpg")])
    end_idx = int(img_names[-1][:img_names[-1].find(".jpg")])

    # Sum of absolute differences
    if type == "SAD2":
        prev_img = None
        for i in range(len(img_names)):
            # Default type is numpy.uint64
            curr_img = cv2.imread(os.path.join(input_dir, img_names[i])).astype(np.int64)
            if i == 0:
                r, c, d = curr_img.shape
                next_img = cv2.imread(os.path.join(input_dir, img_names[i + 1])).astype(np.int64)
                score = np.sum(np.abs(curr_img - next_img))
            elif i == len(img_names) - 1:
                score = np.sum(np.abs(curr_img - prev_img))
            else:
                next_img = cv2.imread(os.path.join(input_dir, img_names[i + 1])).astype(np.int64)
                score = 0.5 * np.sum(np.abs(curr_img - prev_img)) + 0.5 * np.sum(np.abs(curr_img - next_img))
            scores.append(score)
            prev_img = curr_img
        x = np.arange(start_idx, end_idx + 1)
        scores = np.array(scores) / (r * c * d)
        title = "Sum of absolute differences"
        new_filename = "output/" + input_dir.name + "_score_sad2.png"

    elif type == "SAD":
        prev_img = None
        for i in range(len(img_names)):
            # Default type is numpy.uint64
            curr_img = cv2.imread(os.path.join(input_dir, img_names[i])).astype(np.int64)
            if i == 0:
                r, c, d = curr_img.shape
                prev_img = curr_img
                continue
            score = np.sum(np.abs(curr_img - prev_img))
            scores.append(score)
            prev_img = curr_img
        x = np.arange(start_idx + 1, end_idx + 1)
        scores = np.array(scores) / (r * c * d)
        title = "Sum of absolute differences"
        new_filename = "output/" + input_dir.name + "_score_sad2.png"

    # Histogram differences
    elif type == "HD":
        prev_histogram = None
        for i in range(len(img_names)):
            # Default type is numpy.uint8
            curr_img = cv2.imread(os.path.join(input_dir, img_names[i]))#.astype(np.int64)
            # Default type is numpy.uint8
            curr_img_g = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY).astype(np.int16)
            histogram = np.histogram(np.ravel(curr_img_g), bins=np.arange(-1, 256))
            if i == 0:
                r, c = curr_img_g.shape
                prev_histogram = histogram[0]
                continue
            score = np.sum(np.abs(histogram[0] - prev_histogram))
            scores.append(score)
            prev_histogram = histogram[0]

        x = np.arange(start_idx + 1, end_idx + 1)
        scores = np.array(scores) / (r * c)
        title = "Histogram differences"
        new_filename = "output/" + input_dir.name + "_score_hd.png"

    else:
        raise ValueError("Illegal type value")

    print(len(x), len(scores))
    f = plt.figure(figsize=(10, 5))
    ax = f.gca()
    ax.set_xticks(np.arange(start_idx, end_idx, 10))
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.plot(x, scores)
    plt.grid()
    f.savefig(new_filename)
    print("Output saved to " + new_filename)