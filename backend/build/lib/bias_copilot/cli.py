# bias_copilot/cli.py
import click
import tensorflow as tf
from .core import mitigate_bias
import pandas as pd
import os

@click.group()
def cli():
    pass

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--model', default='tf', help='Model type (tf)')
@click.option('--prebuilt', is_flag=True, help='Use pre-built model')
def analyze(file, model, prebuilt):
    """Analyze bias in a CSV file."""
    if prebuilt and os.path.exists('prebuilt_model.h5'):
        model = tf.keras.models.load_model('prebuilt_model.h5')
    else:
        model = None  # Will be trained in mitigate_bias
    mitigated_model, metrics = mitigate_bias(model, file)
    click.echo(f"Before Mitigation: {metrics['before']}")
    click.echo(f"Before Accuracy: {metrics['before_accuracy']:.4f}")
    click.echo(f"After Mitigation: {metrics['after']}")
    click.echo(f"After Accuracy: {metrics['after_accuracy']:.4f}")

if __name__ == "__main__":
    cli()