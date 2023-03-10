import argparse
import typing
from pathlib import Path
import click
import time
from models import train_polynomial_model, predict


@click.group()
def cli() -> None:
    '''
    interface for training, testing and making prediction for polynomial regression model
    '''


@click.command(name='train')
@click.option('--p', default='RealEstate.csv', help='dataset path')
@click.option('--d', default=2, help='polynom degree')
@click.option('--l', default='Y house price of unit area', help='column name from dataset for label')
@click.option('--dc', default='No', help='column name to drop from dataset, could be a list of columns')
@click.option('--sl', default=20, help='percentage to split data to the validation and train')
@click.option('--mn', default='house_prediction',
              help='model name to save trained model, give it with full path. Example: /your/path/<model_name>')
def train(p: str, d: int, l: str, sl: int, mn: str, dc ) -> None:
    # with click.progressbar(label='processing', length=100, show_eta=True) as progress:
    # click.pass_context(f=train_polynomial_model)
    # with click.pass_context() as progress:
    train_polynomial_model(data_path=p, split_rate=sl, label_name=l, polynom_degree=d, model_name=mn, dropped_columns=dc)


@click.command(name='test')
@click.option('--dp', default='RealEstate.csv', help='dataset path')
@click.option('--d', default=2, help='polynom degree that you used during training')
@click.option('--l', default='Y house price of unit area', help='column name from dataset for label')
@click.option('--mn', default='house_prediction',
              help='model name to load and make prediction. Give it with full path . Example: /your/path/<model_name>')
def test(p: str, l: str, mn: str, d: int) -> None:
    predict(data_path=p, label_name=l, model_name=mn, polynom_degree=d)


# @click.command(name='predict')
# @click.option('--data', type=(int, int),
#               help='list of int, float or string  values of one instance. Order of data should be same as training'
#               )
# def predict(data: typing.Tuple[int]) -> None:
#     print('prediction done', data)


if __name__ == '__main__':
    cli.add_command(train)
    cli.add_command(test)
    cli()
