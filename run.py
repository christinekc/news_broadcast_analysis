import click
from pathlib import Path

@click.group()
def main():
    pass

@main.command()
def train(**kwargs):
    from train import train_model
    train_model(**kwargs)

@main.command()
def predict(**kwargs):
    from train import predict
    predict(**kwargs)

@main.command()
@click.option("--input_dir", "-d", type=Path, default="original_data/clip_1/")
@click.option("--output_dir", "-o", type=Path, default="output/clip_1_face/")
def face_detection(**kwargs):
    from train import face_detection_hsv, face_detection_cascade
    # face_detection_hsv(**kwargs)
    face_detection_cascade(**kwargs)

@main.command()
@click.option("--input_dir", "-d", type=Path, default="original_data/female/")
@click.option("--output_dir", "-o", type=Path, default="data/female_warped/")
def warp_face(**kwargs):
    from train import warp_face
    warp_face(**kwargs)

@main.command()
@click.option("--input_dir", "-d", type=Path, default="original_data/clip_1/")
@click.option("--logo_path", "-d", type=Path, default="data/logo1.png")
def detect_logo(**kwargs):
    from detect import detect_logo
    detect_logo(**kwargs)

@main.command()
def pipeline(**kwargs):
    pass

@main.command()
@click.option("--imgs_dir", "-d", default="original_data/clip_1/", type=Path)
@click.option("--vid_name", "-v", default="clip_1")
@click.option("--fps", "-fps", default=6, type=int, help="Frame per second")
def make_video(**kwargs):
    from utils import make_video
    make_video(**kwargs)

if __name__ == "__main__":
    main()
