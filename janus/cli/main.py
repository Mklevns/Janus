import click
from janus.cli.commands.train import train
from janus.cli.commands.evaluate import evaluate
from janus.cli.commands.discover import discover
from janus.cli.commands.visualize import visualize

@click.group()
def cli():
    """Janus Command-Line Interface"""
    pass

cli.add_command(train)
cli.add_command(evaluate)
cli.add_command(discover)
cli.add_command(visualize)
# Add other commands here later

if __name__ == '__main__':
    cli()
