import os
import json
import importlib
from . import vision

modules_file = os.path.join(os.getcwd(), "modules.json")
if os.path.exists(modules_file):
    with open(modules_file, "r") as jsonfile:
        modules = json.load(jsonfile)
    if "datasets" in modules:
        for module in modules["datasets"]:
            importlib.import_module(module)

__all__ = ["vision"]
