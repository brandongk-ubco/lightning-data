import argh
from lightningdata import commands

parser = argh.ArghParser()
parser.add_commands(commands.__all__)

if __name__ == '__main__':
    parser.dispatch()
