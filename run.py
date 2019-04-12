import click
from pathlib import Path

@click.group()
def main():
    pass

@main.command()
@click.option("--model_path", "-m", type=Path, default="cnn_model.h5")
# @click.option("--model_path", "-m", type=Path, default="svm_model.sav")
@click.option("--classification", "-c", default="CNN", help="SVM or NN_SIFT or CNN")
def train(**kwargs):
    from face import train_model
    train_model(**kwargs)

@main.command()
@click.option("--model_path", "-m", type=Path, default="nn_model.h5")
# @click.option("--model_path", "-m", type=Path, default="svm_model.sav")
def predict(**kwargs):
    from face import predict_model
    predict_model(**kwargs)

@main.command()
@click.option("--model_path", "-m", type=Path, default="svm_model.sav")
# @click.option("--model_path", "-m", type=Path, default="nn_model.h5")
@click.option("--classification", "-c", default="SVM", help="SVM or NN_SIFT or CNN")
# Clip 1
# @click.option("--input_dir", "-i", type=Path, default="original_data/clip_1/")
# @click.option("--output_dir", "-o", type=Path, default="output/clip_1_face/")
# Clip 2
@click.option("--input_dir", "-i", type=Path, default="original_data/clip_1/")
@click.option("--output_dir", "-o", type=Path, default="output/clip_1/")
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
@click.option("--input_dir", "-i", type=Path, default="original_data/clip_1/")
@click.option("--type", "-t", default="HD", help="SAD2 or HD")
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
# @click.option("--input_dir", "-i", type=Path, default="original_data/clip_1/")
# @click.option("--output_dir", "-o", type=Path, default="output/clip_1/")
# @click.option("--logo_path", "-d", type=Path, default="data/clip_1_logo.png")
# @click.option("--min_threshold", "-t", type=float, default=0.87)
# @click.option("--model_path", "-m", type=Path, default="svm_model.sav")
# @click.option("--classification", "-c", default="SVM", help="SVM or NN_SIFT or CNN")
# @click.option("--vid_name", "-v", default="clip_1")
# @click.option("--fps", "-fps", default=6, type=int, help="Frame per second")
# Clip 2
# @click.option("--input_dir", "-i", type=Path, default="original_data/clip_2/")
# @click.option("--output_dir", "-o", type=Path, default="output/clip_2/")
# @click.option("--logo_path", "-d", type=Path, default="data/clip_2_logo.png")
# @click.option("--min_threshold", "-t", type=float, default=0.82)
# @click.option("--model_path", "-m", type=Path, default="svm_model.sav")
# @click.option("--classification", "-c", default="SVM", help="SVM or NN_SIFT or CNN")
# @click.option("--vid_name", "-v", default="clip_2")
# @click.option("--fps", "-fps", default=6, type=int, help="Frame per second")
# Clip 3
@click.option("--input_dir", "-i", type=Path, default="original_data/clip_3/")
@click.option("--output_dir", "-o", type=Path, default="output/clip_3/")
@click.option("--logo_path", "-d", type=Path, default="data/clip_3_logo.png")
@click.option("--min_threshold", "-t", type=float, default=0.82)
@click.option("--model_path", "-m", type=Path, default="svm_model.sav")
@click.option("--classification", "-c", default="SVM", help="SVM or NN_SIFT or CNN")
@click.option("--vid_name", "-v", default="clip_3")
@click.option("--fps", "-fps", default=6, type=int, help="Frame per second")
def run_all(input_dir, output_dir, logo_path, min_threshold, model_path, \
    classification, vid_name, fps):
    from face import face_detection_cascade
    from logo import logo_detection
    from utils import make_video
    logo_detection(input_dir=input_dir, output_dir=output_dir,\
        logo_path=logo_path, min_threshold=min_threshold)
    face_detection_cascade(input_dir=output_dir, output_dir=output_dir,\
        model_path=model_path, classification=classification)
    make_video(imgs_dir=output_dir, vid_name=vid_name, fps=fps)

if __name__ == "__main__":
    main()
