import argh
from inspect import getmembers, isfunction
from .vision import commands as vision_commands
from .timeseries import commands as timeseries_commands
import sys

usage = """Run some commands, pick one of {vision, timeseries}"""

if len(sys.argv) < 2:
    print(usage)
    sys.exit()

command_family = sys.argv.pop(1)

if command_family == "vision":
    commands = vision_commands
elif command_family == "timeseries":
    commands = timeseries_commands
else:
    print(usage)
    sys.exit()

command_list = [o[1] for o in getmembers(commands) if isfunction(o[1])]

parser = argh.ArghParser()
parser.add_commands(command_list)

if __name__ == '__main__':
    parser.dispatch()
