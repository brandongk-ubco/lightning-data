import argh
from lightningdata import commands
from inspect import getmembers, isfunction
import os
import json
import importlib

command_list = [o[1] for o in getmembers(commands) if isfunction(o[1])]

parser = argh.ArghParser()
parser.add_commands(command_list)

if __name__ == '__main__':
    datasets_file = os.path.join(os.getcwd(), "datasets.json")
    if os.path.exists(datasets_file):
        with open(datasets_file, "r") as jsonfile:
            datasets = json.load(jsonfile)
        for module in datasets:
            importlib.import_module(module)
    parser.dispatch()
