import click
from pathlib import Path

@click.group()
def main():
    pass

@main.command()
@click.option("--data-dir", "-d", type=Path)
def train(**kwargs):
    from train import train_model
    train_model(**kwargs)

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
