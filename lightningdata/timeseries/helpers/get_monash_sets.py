import os
import json


def get_monash_sets():
    monash_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "monash.json")
    with open(monash_json_path, "r") as f:
        monash_json = json.load(f)
    monash_sets = dict([(h["files"][0]["key"][:-4],
                         h["files"][0]["links"]["self"])
                        for h in monash_json['hits']['hits']])
    return monash_sets
