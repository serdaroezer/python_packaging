import click
from .models import train_regression_model, predict


@click.group()
def cli() -> None:
    '''
    interface for training and making prediction for polynomial regression model
    '''


@click.command(name='train')
@click.option('--p', default='RealEstate.csv', help='dataset path')
@click.option('--l', default='Y house price of unit area', help='column name from dataset for label')
@click.option('--dc', default=['No', 'X1 transaction date', 'X2 house age'],
              help='column name to drop from dataset, could be a list of columns',
              multiple=True)
@click.option('--sl', default=20, help='percentage to split data to the validation and train')
@click.option('--mn', default='house_prediction',
              help='model name to save trained model, give it with full path. Example: /your/path/<model_name>')
def train(p, l, sl, mn, dc) -> None:
    train_regression_model(data_path=p, split_rate=sl, label_name=l, model_name=mn, dropped_columns=list(dc))


@click.command(name='test')
@click.option('--p', default='RealEstate.csv', help='dataset path')
@click.option('--l', default='Y house price of unit area', help='column name from dataset for label')
@click.option('--dc', default=['No', 'X1 transaction date', 'X2 house age'],
              help='column name to drop from dataset, could be a list of columns',
              multiple=True)
@click.option('--mn', default='house_prediction',
              help='model name to load and make prediction. Give it with full path . Example: /your/path/<model_name>')
def test(p, l, dc, mn) -> None:
    predict(data_path=p, label_name=l, model_name=mn, dropped_columns=list(dc))


if __name__ == '__main__':
    cli.add_command(train)
    cli.add_command(test)
    cli()
