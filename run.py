import click
from pathlib import Path

@click.group()
def main():
    pass

@main.command()
# @click.option("--model_path", "-m", type=Path, default="models/cnn_model.h5")
@click.option("--model_path", "-m", type=Path, default="models/svm_model.sav")
@click.option("--classification", "-c", default="SVM", help="SVM or NN_SIFT or CNN")
def train(**kwargs):
    from face import train_model
    train_model(**kwargs)

@main.command()
# @click.option("--model_path", "-m", type=Path, default="models/nn_model.h5")
@click.option("--model_path", "-m", type=Path, default="models/svm_model.sav")
def predict(**kwargs):
    from face import predict_model
    predict_model(**kwargs)

@main.command()
@click.option("--model_path", "-m", type=Path, default="models/svm_model.sav")
# @click.option("--model_path", "-m", type=Path, default="models/nn_model.h5")
@click.option("--classification", "-c", default="SVM", help="SVM or NN_SIFT or CNN")
# Clip 1
@click.option("--input_dir", "-i", type=Path, default="original_data/clip_1/")
@click.option("--output_dir", "-o", type=Path, default="output/clip_1_face/")
@click.option("--min_size", "-s", default=70, type=int, help="Min size for face detection")
# Clip 2
# @click.option("--input_dir", "-i", type=Path, default="original_data/clip_2/")
# @click.option("--output_dir", "-o", type=Path, default="output/clip_2/")
# @click.option("--min_size", "-s", default=70, type=int, help="Min size for face detection")
def face_detection(**kwargs):
    from face import face_detection_hsv, face_detection_cascade
    # face_detection_hsv(**kwargs)
    face_detection_cascade(**kwargs)

@main.command()
# Clip 1
# @click.option("--input_dir", "-i", type=Path, default="original_data/clip_1/")
# @click.option("--output_dir", "-o", type=Path, default="output/clip_1_logo/")
# @click.option("--logo_path", "-d", type=Path, default="data/clip_1_logo2.png")
# @click.option("--min_threshold", "-t", type=float, default=0.87)
# Clip 2
# @click.option("--input_dir", "-i", type=Path, default="original_data/clip_2/")
# @click.option("--output_dir", "-o", type=Path, default="output/clip_2/")
# @click.option("--logo_path", "-d", type=Path, default="data/clip_2_logo.png")
# @click.option("--min_threshold", "-t", type=float, default=0.82)
# Clip 3
@click.option("--input_dir", "-i", type=Path, default="original_data/clip_3/")
@click.option("--output_dir", "-o", type=Path, default="output/clip_3/")
@click.option("--logo_path", "-d", type=Path, default="data/clip_3_logo.png")
@click.option("--min_threshold", "-t", type=float, default=0.82)
def logo_detection(**kwargs):
    from logo import logo_detection
    logo_detection(**kwargs)

@main.command()
@click.option("--input_dir", "-i", type=Path, default="original_data/clip_3/")
@click.option("--method", "-m", default="SAD2", help="SAD2 or HD")
@click.option("--k", "-k", default=1, help="k used in threshold")
def shot_detection(**kwargs):
    from shot import shot_detection
    shot_detection(**kwargs)

@main.command()
@click.option("--imgs_dir", "-i", default="original_data/clip_1/", type=Path)
@click.option("--vid_name", "-v", default="clip_1")
@click.option("--fps", "-fps", default=6, type=int, help="Frame per second")
def make_video(**kwargs):
    from utils import make_video
    make_video(**kwargs)

@main.command()
# Clip 1
@click.option("--input_dir", "-i", type=Path, default="original_data/clip_1/")
@click.option("--output_dir", "-o", type=Path, default="output/clip_1_shots/")
@click.option("--shots", "-s", type=list, default=[22, 156])
# Clip 2
# @click.option("--input_dir", "-i", type=Path, default="original_data/clip_2/")
# @click.option("--output_dir", "-o", type=Path, default="output/clip_2_shots/")
# @click.option("--shots", "-s", type=list, default=[65, 120, 138, 144, 152, 164, 177, 188])
# Clip 3
# @click.option("--input_dir", "-i", type=Path, default="original_data/clip_3/")
# @click.option("--output_dir", "-o", type=Path, default="output/clip_3_shots/")
# @click.option("--shots", "-s", type=list, default=[16, 51, 57, 59, 61, 63, 65, 67, 69, 71, 72, 74, 78, 80, 84, 91, 95, 102, 165, 187, 260, 267])
def add_shot_number(**kwargs):
    from shot import add_shot_number
    add_shot_number(**kwargs)

@main.command()
# Clip 1
# @click.option("--input_dir", "-i", type=Path, default="output/clip_1_shots/")
# @click.option("--output_dir", "-o", type=Path, default="output/clip_1/")
# @click.option("--logo_path", "-d", type=Path, default="data/clip_1_logo.png")
# @click.option("--min_threshold", "-t", type=float, default=0.87)
# @click.option("--model_path", "-m", type=Path, default="models/svm_model.sav")
# @click.option("--classification", "-c", default="SVM", help="SVM or NN_SIFT or CNN")
# @click.option("--vid_name", "-v", default="clip_1")
# @click.option("--fps", "-fps", default=6, type=int, help="Frame per second")
# @click.option("--min_size", "-s", default=70, type=int, help="Min size for face detection")
# Clip 2
@click.option("--input_dir", "-i", type=Path, default="output/clip_2_shots/")
@click.option("--output_dir", "-o", type=Path, default="output/clip_2/")
@click.option("--logo_path", "-d", type=Path, default="data/clip_2_logo.png")
@click.option("--min_threshold", "-t", type=float, default=0.82)
@click.option("--model_path", "-m", type=Path, default="models/svm_model.sav")
@click.option("--classification", "-c", default="SVM", help="SVM or NN_SIFT or CNN")
@click.option("--vid_name", "-v", default="clip_2")
@click.option("--fps", "-fps", default=6, type=int, help="Frame per second")
@click.option("--min_size", "-s", default=70, type=int, help="Min size for face detection")
# Clip 3
# @click.option("--input_dir", "-i", type=Path, default="output/clip_3_shots/")
# @click.option("--output_dir", "-o", type=Path, default="output/clip_3/")
# @click.option("--logo_path", "-d", type=Path, default="data/clip_3_logo.png")
# @click.option("--min_threshold", "-t", type=float, default=0.82)
# @click.option("--model_path", "-m", type=Path, default="models/svm_model.sav")
# @click.option("--classification", "-c", default="SVM", help="SVM or NN_SIFT or CNN")
# @click.option("--vid_name", "-v", default="clip_3")
# @click.option("--fps", "-fps", default=8, type=int, help="Frame per second")
# @click.option("--min_size", "-s", default=40, type=int, help="Min size for face detection")
def run_all(input_dir, output_dir, logo_path, min_threshold, model_path, \
    classification, vid_name, fps, min_size):
    from face import face_detection_cascade
    from logo import logo_detection
    from utils import make_video
    logo_detection(input_dir=input_dir, output_dir=output_dir,\
        logo_path=logo_path, min_threshold=min_threshold)
    face_detection_cascade(input_dir=output_dir, output_dir=output_dir,\
    # face_detection_cascade(input_dir=input_dir, output_dir=output_dir,\
        model_path=model_path, classification=classification, min_size=min_size)
    make_video(imgs_dir=output_dir, vid_name=vid_name, fps=fps)

if __name__ == "__main__":
    main()
