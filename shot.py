import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def add_shot_number(input_dir, output_dir, shots):
    """
    Add shot number using the shot changes specified in variable shots.
    """
    print("add_shot_number")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    exts = [".jpg", ".png"]
    img_names = [img for img in os.listdir(input_dir) if img.endswith(tuple(exts))]
    # Sort images by name in ascending order
    img_names.sort(key=lambda img: int("".join(filter(str.isdigit, img))))

    i = 0
    shot = 0
    for filename in img_names:
        image_num = int("".join(filter(str.isdigit, filename)))
        img = cv2.imread(os.path.join(input_dir, filename))
        if i < len(shots) and image_num == shots[i]:
            shot += 1
            i += 1
        # Add shot number to frame
        text = str(shot)
        # Get width and height of the text box
        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, thickness=2)[0]
        # Set the text start position
        text_x = 10
        text_y = img.shape[0] - 10
        box_coords = ((text_x, text_y + 2), (text_x + text_width - 2, text_y - text_height - 4))
        cv2.rectangle(img, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(0, 0, 0), thickness=2)
        # Save frame
        cv2.imwrite(os.path.join(output_dir, filename), img)
        # cv2.imshow("A box!", img)
        # cv2.waitKey(0)

def shot_detection(input_dir, method, k):
    """
    Get graphs of scores of shot changes using the method specify in method.
    Method should be either SAD2 or HD.
    """
    scores = []
    exts = [".jpg", ".png"]
    img_names = [img for img in os.listdir(input_dir) if img.endswith(tuple(exts))]
    # Sort images by name in ascending order
    img_names.sort(key=lambda img: int("".join(filter(str.isdigit, img))))
    start_idx = int(img_names[0][:img_names[0].find(".jpg")])
    end_idx = int(img_names[-1][:img_names[-1].find(".jpg")])

    # Sum of absolute differences
    if method == "SAD2":
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

    # Histogram differences
    elif method == "HD":
        prev_histogram = None
        for i in range(len(img_names)):
            # Default type is numpy.uint8
            curr_img = cv2.imread(os.path.join(input_dir, img_names[i]))
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
        raise ValueError("Illegal method value")

    y_mean = [np.mean(scores)] * len(x)
    threshold = [np.mean(scores) + k * np.std(scores)] * len(x)
    f = plt.figure(figsize=(10, 5))
    ax = f.gca()
    ax.set_xticks(np.arange(start_idx, end_idx, 10))
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.plot(x, scores, label="Scores")
    # Plot the average line
    plt.plot(x, y_mean, label="Mean", linestyle="--")
    plt.plot(x, threshold, label="Threshold k = " + str(k), linestyle="--")
    plt.legend(loc="upper right")
    plt.grid()
    f.savefig(new_filename)
    print("Output saved to " + new_filename)