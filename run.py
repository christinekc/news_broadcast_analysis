import click
from pathlib import Path

@click.group()
def main():
    pass

@main.command()
@click.option("--model_path", "-m", type=Path, default="cnn_model.h5")
@click.option("--classification", "-c", default="CNN", help="SVM or CNN")
def train(**kwargs):
    from face import train_model
    train_model(**kwargs)

@main.command()
@click.option("--model_path", "-m", type=Path, default="svm_model.sav")
def predict(**kwargs):
    from face import predict_model
    predict_model(**kwargs)

@main.command()
@click.option("--input_dir", "-i", type=Path, default="original_data/clip_1/")
@click.option("--output_dir", "-o", type=Path, default="output/clip_1_face/")
# @click.option("--model_path", "-m", type=Path, default="svm_model.sav")
@click.option("--model_path", "-m", type=Path, default="cnn_model.h5")
@click.option("--classification", "-t", default="CNN", help="SVM or CNN")
def face_detection(**kwargs):
    from face import face_detection_hsv, face_detection_cascade
    # face_detection_hsv(**kwargs)
    face_detection_cascade(**kwargs)

@main.command()
@click.option("--input_dir", "-i", type=Path, default="original_data/clip_1/")
@click.option("--output_dir", "-o", type=Path, default="output/clip_1_logo/")
@click.option("--logo_path", "-d", type=Path, default="data/logo.png")
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
def pipeline(**kwargs):
    pass

@main.command()
@click.option("--imgs_dir", "-i", default="original_data/clip_1/", type=Path)
@click.option("--vid_name", "-v", default="clip_1")
@click.option("--fps", "-fps", default=6, type=int, help="Frame per second")
def make_video(**kwargs):
    from utils import make_video
    make_video(**kwargs)

if __name__ == "__main__":
    main()
