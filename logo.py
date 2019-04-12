import cv2
import imutils
import numpy as np
import os

# def logo_detection(input_dir, output_dir,logo_path):
#     print("detect_logo")

#     if not os.path.isdir(output_dir):
#         os.mkdir(output_dir)

#     template = cv2.imread(str(logo_path))
#     template_g = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#     w, h = template_g.shape[::-1]

#     # Compute template of different sizes
#     scales = np.linspace(0.1, 1.0, 25)[::-1]
#     templates = []
#     ratios = []
#     for scale in scales:
#         resized = imutils.resize(template_g, width=int(template_g.shape[1] * scale))
#         template_canny = cv2.Canny(resized, 50, 200)
#         templates.append(template_canny)
#         ratios.append(resized.shape[1] / float(template_g.shape[1]))

#     for img_name in os.listdir(input_dir):
#         if not img_name.endswith(".jpg"):
#             continue
#         img = cv2.imread(os.path.join(input_dir, img_name))
#         img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img_canny = cv2.Canny(img_g, 50, 200)
#         # cv2.imshow("img_canny", img_canny)
#         # cv2.waitKey(0)

#         found = None
#         # Loop through the templates from small to big
#         for i in range(len(templates) - 1, -1, -1):
#             # Stop when template is bigger than image
#             if img_g.shape[0] < templates[i].shape[0] or img_g.shape[1] < templates[i].shape[1]:
#                 break

#             result = cv2.matchTemplate(img_canny, templates[i], cv2.TM_CCORR_NORMED) # img_g.shape - template_g.shape + 1
#             _, max_val, _, max_loc = cv2.minMaxLoc(result)
#             # print(max_val)
#             # temp_img = img
#             # r = ratios[i]
#             # start_x, start_y = max_loc[0], max_loc[1]
#             # end_x, end_y = int((max_loc[0] + w * r)), int((max_loc[1] + h * r))
#             # cv2.rectangle(temp_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), thickness=2)
#             # cv2.imshow("temp_img", temp_img)
#             # cv2.waitKey(0)

#             if found is None or max_val > found[0]:
#                 found = (max_val, max_loc, ratios[i])

#         max_val, max_loc, r = found
#         start_x, start_y = max_loc[0], max_loc[1]
#         end_x, end_y = int((max_loc[0] + w * r)), int((max_loc[1] + h * r))

#         cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), thickness=2)
#         cv2.imwrite(os.path.join(output_dir, img_name), img)

#         cv2.imshow("img", img)
#         cv2.waitKey(0)

def get_score(img1, img2):
    """
    Get similarity score between img1 and img2 using SIFT features matching and
    Lowe's ratio testing.
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if len(kp1) < 2 or len(kp2) < 2:
        return 0
    index_params = dict(algorithm=0, trees=5)
    flann = cv2.FlannBasedMatcher(index_params, None)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good_matches.append(m)
    num_kps = 0
    if len(kp1) <= len(kp2):
        num_kps = len(kp1)
    else:
        num_kps = len(kp2)
    score = len(good_matches) / num_kps * 100
    return score

def logo_detection(input_dir, output_dir, logo_path, min_threshold):
    print("detect_logo")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    template = cv2.imread(str(logo_path))
    template_g = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = template_g.shape[::-1]

    # Compute template of different sizes
    if template.shape[0] < 50:
        scales = np.linspace(0.8, 1.0, 6)[::-1]
    else:
        # scales = np.linspace(0.1, 1.0, 25)[::-1]
        scales = np.linspace(0.5, 1.0, 10)[::-1]
    templates = []
    ratios = []
    for scale in scales:
        resized = imutils.resize(template_g, width=int(template_g.shape[1] * scale))
        templates.append(resized)
        ratios.append(resized.shape[1] / float(template_g.shape[1]))

    for img_name in os.listdir(input_dir):
        if not img_name.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(input_dir, img_name)) 
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_canny = cv2.Canny(img_g, 100, 200)

        p = -1
        q = -1
        matches = []
        # Loop through the templates from small to big
        for i in range(len(templates) - 1, -1, -1):
            # Stop when template is bigger than image
            if img_g.shape[0] < templates[i].shape[0] or img_g.shape[1] < templates[i].shape[1]:
                break

            # First pass - normalized cross correlation
            match = cv2.matchTemplate(img_g, templates[i], cv2.TM_CCORR_NORMED) # img_g.shape - template_g.shape + 1
            if p == -1 and q == -1:
                p, q = match.shape
            m, n = match.shape
            matches.append(np.pad(match, ((0, p - m), (0, q - n)), mode="constant", constant_values=0))

        boxes = []

        matches = np.array(matches)
        r, max_y, max_x = np.unravel_index(np.argmax(matches), matches.shape)
        r = ratios[len(ratios) - 1 - r]
        max_val = np.max(matches)
        max_thresh = max(max_val * 0.95, min_threshold)
        # If the match with the highest score is smaller than the min threshold,
        # there is no match in this image and just saves the input image.
        if max_val < min_threshold:
            cv2.imwrite(os.path.join(output_dir, img_name), img)
            continue
        start_x, start_y = max_x, max_y
        end_x, end_y = int((max_x + w * r)), int((max_y + h * r))
        match_score = get_score(template, img[start_y:end_y, start_x:end_x])
        if match_score < 5:
            cv2.imwrite(os.path.join(output_dir, img_name), img)
            continue
        boxes.append((r, max_y, max_x, 1))

        # Matches obtained from first pass
        match_locations = np.where(matches >= max_thresh)
        for i in range(len(match_locations[0])):
            r1, y1, x1 = ratios[len(ratios) - 1 - match_locations[0][i]], match_locations[1][i], match_locations[2][i]
            found = False
            for j in range(len(boxes)):
                r2, y2, x2, count = boxes[j]
                # Check if two boxes of the same size overlap or
                # if a smaller one is contained in the bigger one
                if (r1 == r2 and np.abs(x1 - x2) < w * r1 and np.abs(y1 - y2) < h * r1) or\
                    ((r1 < r2) and (x1 <= (x2 + w * r2) <= (x1 + w * r1)) and (y1 <= (y2 + h * r2) <= (y1 + h * r1))) or\
                    (np.abs(x1 - x2) < 0.5 * w * r1 and np.abs((x1 + w * r1) - (x2 + w * r2)) < 0.5 * w * r1 and\
                    np.abs(y1 - y2) < 0.5 * h * r1 and np.abs((y1 + h * r1) - (y2 + h * r2)) < 0.5 * h * r1):
                    boxes[j] = (r2, y2, x2, count + 1)
                    found = True
                    break
            if not found:
                start_x, start_y = x1, y1
                end_x, end_y = int((x1 + w * r1)), int((y1 + h * r1))
                # Second pass - SIFT features matching
                match_score = get_score(template, img[start_y:end_y, start_x:end_x])
                if match_score > 5:
                    boxes.append((r1, y1, x1, 1))

        for r, y, x, count in boxes:
            start_x, start_y = x, y
            end_x, end_y = int((x + w * r)), int((y + h * r))
            # print(start_x, start_y, end_x, end_y)
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), thickness=2)

        cv2.imwrite(os.path.join(output_dir, img_name), img)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
