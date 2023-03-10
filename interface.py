import argparse
import typing
from pathlib import Path
import click


@click.group()
def cli() -> None:
    '''
    interface for training, testing and making prediction for polynomial regression model
    '''


@click.command(name='train')
@click.option('--p', default='RealEstate.csv', help='dataset path')
@click.option('--d', default=2, help='polynom degree')
def train(p: str, d: int) -> None:
    print(p, d)
    click.echo('provide dataset path and degree of polynom')


@click.command(name='test')
@click.option('--p', default='RealEstate.csv', help='dataset path')
def test(p: str) -> None:
    print(p)
    click.echo('provide dataset path to test')


@click.command(name='predict')
@click.option('--data', type=(int, int),
              help='list of int, float or string  values of one instance. Order of data should be same as training'
                  )
def predict(data: typing.Tuple[int]) -> None:
    print('prediction done', data)


if __name__ == '__main__':
    cli.add_command(train)
    cli.add_command(predict)
    cli.add_command(test)
    cli()
