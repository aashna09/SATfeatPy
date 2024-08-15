# cython: language_level=3
import json

def write_features_to_json(dict results_dict):
    with open("features.json", "w") as f:
        json.dump(results_dict, f)
