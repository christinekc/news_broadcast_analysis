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

if __name__ == "__main__":
    main()
