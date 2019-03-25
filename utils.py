import cv2
import os

def make_video(imgs_dir, vid_name, fps):
    """
    Make vid_name.mp4 using images in imgs_dir.
    """
    OUTPUT_DIR = "output/"
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    imgs = [img for img in os.listdir(imgs_dir) if img.endswith(".jpg")]
    # Sort images by name in ascending order
    imgs.sort(key=lambda img: int("".join(filter(str.isdigit, img))))
    frame = cv2.imread(os.path.join(imgs_dir, imgs[0]))
    h, w, _ = frame.shape

    # Make sure vid_name does not include a file extension
    index = vid_name.find(".")
    if index != -1:
        vid_name = vid_name[:index]
    file_path = OUTPUT_DIR + vid_name + ".mp4"
    vid = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (w, h))
    for img in imgs:
        vid.write(cv2.imread(os.path.join(imgs_dir, img)))
    vid.release()
    print("Video is now in ", file_path)