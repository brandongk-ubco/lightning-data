import argh
from lightningdata import commands
from inspect import getmembers, isfunction

command_list = [o[1] for o in getmembers(commands) if isfunction(o[1])]

parser = argh.ArghParser()
parser.add_commands(command_list)

if __name__ == '__main__':
    parser.dispatch()
