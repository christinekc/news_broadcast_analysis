import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def find_affine(kp1, kp2, matches):
    # p*a = p_prime
    p = []
    p_prime = []
    for i in range(len(matches)):
        x, y = kp1[matches[i].queryIdx].pt
        x_prime, y_prime = kp2[matches[i].trainIdx].pt
        p.append([x, y, 0, 0, 1, 0])
        p.append([0, 0, x, y, 0, 1])
        p_prime.append([x_prime])
        p_prime.append([y_prime])
    p = np.array(p)
    p_prime = np.array(p_prime)
    
    p_T = np.transpose(p)
    a = np.matmul(np.matmul(np.linalg.inv(np.matmul(p_T, p)), p_T), p_prime)
    return a

# def logo_detection(input_dir, logo_path):
#     print("detect_logo")
#     sift = cv2.xfeatures2d.SIFT_create()
#     img1 = cv2.imread(str(logo_path))

#     kp1, des1 = sift.detectAndCompute(img1, None)
#     h1, w1, _ = img1.shape
#     coords1 = np.array([[1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 0, 1],
#                         [1, h1, 0, 0, 1, 0], [0, 0, 1, h1, 0, 1],
#                         [w1, h1, 0, 0, 1, 0], [0, 0, w1, h1, 0, 1],
#                         [w1, 1, 0, 0, 1, 0], [0, 0, w1, 1, 0, 1]])

#     print(str(input_dir))
#     for img_name in os.listdir(input_dir):
#         if not img_name.endswith(".jpg"):
#             continue
#         img2 = cv2.imread(os.path.join(input_dir, img_name))
#         kp2, des2 = sift.detectAndCompute(img2, None)

#         # FLANN parameters
#         FLANN_INDEX_KDTREE = 0
#         index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#         search_params = dict(checks=10)
#         flann = cv2.FlannBasedMatcher(index_params, search_params)
#         matches = flann.knnMatch(des1, des2, k=2)
#         # bf = cv2.BFMatcher()
#         # matches = bf.knnMatch(des1,des2, k=2)

#         if len(matches) < 4:
#             # No logo in current img      
#             continue

#         good = []
#         pts1 = []
#         pts2 = []
#         # Lowe's ratio test
#         for i, (m, n) in enumerate(matches):
#             if m.distance < 0.8*n.distance:
#                 good.append([m])
#                 pts1.append(kp1[m.queryIdx].pt)
#                 pts2.append(kp2[m.trainIdx].pt)

#         if len(good) <= 4:
#             continue

#         result1 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good,
#             matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), outImg=img2)            
#         cv2.imshow("ImageWindow", result1)
#         cv2.waitKey()


#         # H = find_affine(kp1, kp2, good)
#         # new_coords = np.matmul(coords1, H)

#         # f = plt.figure()
#         # f.set_size_inches(20, 20)
#         # plt.axis('off')
#         # plt.imshow(img2)
#         # plt.plot(
#         #     [new_coords[0], new_coords[2], new_coords[4], new_coords[6], new_coords[0]],
#         #     [new_coords[1], new_coords[3], new_coords[5], new_coords[7], new_coords[1]],
#         #     "r")
#         # plt.show()

def logo_detection(input_dir, logo_path):
    print("detect_logo")
    template = cv2.imread(str(logo_path))
    template_g = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_canny = cv2.Canny(template_g, 50, 200)

    w, h = template_g.shape[::-1]

    for img_name in os.listdir(input_dir):
        if not img_name.endswith(".jpg"):
            continue
        img = cv2.imread(os.path.join(input_dir, img_name)) 
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_canny = cv2.Canny(img_g, 50, 200)
        # cv2.imshow("Match", img_canny)
        # cv2.waitKey(0)
        
        # result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED) # img_g.shape - template_g.shape + 1
        # result = cv2.matchTemplate(img_g, template_g, cv2.TM_SQDIFF_NORMED) # img_g.shape - template_g.shape + 1
        result = cv2.matchTemplate(img_canny, template_canny, cv2.TM_CCORR_NORMED) # img_g.shape - template_g.shape + 1

        # min_x, max_y, minloc, maxloc = cv2.minMaxLoc(result)
        # print("max_y:", max_y)
        # x, y = maxloc

        # result = (result - np.min(result)) * 255.0 / (np.max(result) - np.min(result)) # shift range

        result2 = np.reshape(result, result.shape[0]*result.shape[1])
        sort = np.argsort(result2)
        y1, x1 = np.unravel_index(sort[-1], result.shape) # best match
        y2, x2 = np.unravel_index(sort[-2], result.shape) # second best match

        print(result[y1, x1], result[y2, x2])
        # cv2.imshow("result", result)
        # cv2.waitKey(0)

        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), thickness=2)
        cv2.rectangle(img, (x2, y2), (x2+w, y2+h), (0, 0, 255), thickness=2)
        cv2.imshow("Match", img)
        cv2.waitKey(0)
