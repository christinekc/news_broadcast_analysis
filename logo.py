import cv2
import imutils
import numpy as np
import os

import copy


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







def logo_detection(input_dir, output_dir,logo_path):
    print("detect_logo")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    template = cv2.imread(str(logo_path))
    template_g = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = template_g.shape[::-1]

    # Compute template of different sizes
    scales = np.linspace(0.1, 1.0, 25)[::-1]
    templates = []
    ratios = []
    for scale in scales:
        resized = imutils.resize(template_g, width=int(template_g.shape[1] * scale))
        template_canny = cv2.Canny(resized, 50, 200)
        templates.append(template_canny)
        ratios.append(resized.shape[1] / float(template_g.shape[1]))

    for img_name in os.listdir(input_dir):
        if not img_name.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(input_dir, img_name)) 
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_canny = cv2.Canny(img_g, 50, 200)

        p = -1
        q = -1
        matches = []
        # Loop through the templates from small to big
        for i in range(len(templates) - 1, -1, -1):
            # Stop when template is bigger than image
            if img_g.shape[0] < templates[i].shape[0] or img_g.shape[1] < templates[i].shape[1]:
                break

            match = cv2.matchTemplate(img_canny, templates[i], cv2.TM_CCORR_NORMED) # img_g.shape - template_g.shape + 1
            if p == -1 and q == -1:
                p, q = match.shape
            m, n = match.shape
            matches.append(np.pad(match, ((0, p - m), (0, q - n)), mode="constant", constant_values=0))

        matches = np.array(matches)
        r, max_y, max_x = np.unravel_index(np.argmax(matches), matches.shape)
        r = ratios[len(ratios) - 1 - r]
        max_val = np.max(matches)

        max_thresh = max_val * 0.95
        match_locations = np.where(matches >= max_thresh)

        start_x, start_y = max_x, max_y
        end_x, end_y = int((max_x + w * r)), int((max_y + h * r))

        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), thickness=2)
        cv2.imwrite(os.path.join(output_dir, img_name), img)

        cv2.imshow("img", img)
        cv2.waitKey(0)
