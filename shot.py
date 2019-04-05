import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def shot_detection(input_dir):
    scores = []
    prev_img = None
    img_names = [img for img in os.listdir(input_dir) if img.endswith(".jpg")]
    # Sort images by name in ascending order
    img_names.sort(key=lambda img: int("".join(filter(str.isdigit, img))))
    start_idx = int(img_names[0][:img_names[0].find(".jpg")])
    end_idx = int(img_names[-1][:img_names[-1].find(".jpg")])

    for i in range(len(img_names)):
        curr_img = cv2.imread(os.path.join(input_dir, img_names[i]))

        if i == 0:
            prev_img = curr_img
            continue

        score = np.sum(curr_img - prev_img)
        scores.append(score)
        prev_img = curr_img

    x = np.arange(start_idx+1, end_idx+1)
    print(len(x), len(scores))
    f = plt.figure()
    ax = f.gca()
    ax.set_xticks(np.arange(start_idx, end_idx, 10))
    plt.title("Sum of absolute differences")
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.plot(x, scores)
    plt.grid()
    f.savefig(input_dir.name + "_score.png")