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
    modules_file = os.path.join(os.getcwd(), "modules.json")
    if os.path.exists(modules_file):
        with open(modules_file, "r") as jsonfile:
            modules = json.load(jsonfile)
        if "datasets" in modules:
            for module in modules["datasets"]:
                importlib.import_module(module)
    parser.dispatch()
