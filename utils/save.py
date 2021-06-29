from enum import Enum
import json
import os
from dataclasses import is_dataclass


def get_json_dict(t):
    if isinstance(t,  Enum):
        return t.value

    if isinstance(t, str):
        return t

    if isinstance(t, int) | isinstance(t, float):
        return t

    json_dict = {}

    for k, v in vars(t).items():
        if not k.endswith("__"):
            if type(v) == type or is_dataclass(v):
                json_dict[k] = get_json_dict(v)
            elif isinstance(v, Enum):
                json_dict[k] = v.value
            elif isinstance(v, list):
                json_dict[k] = [get_json_dict(v_i) for v_i in v]
            else:
                json_dict[k] = v
    return json_dict


def save_parameters_json(path: str, parameters):
    parameters_dict = get_json_dict(parameters)
    save_parameters_json_dict(path, parameters_dict)


def save_parameters_json_dict(path: str, dictionary):
    with open(path, "w") as output_file:
        json.dump(dictionary, output_file, indent="\t")


def load_parameters(folder_path: str, file_name: str):
    parameters_loading_path = os.path.join(
        folder_path, file_name
    )
    with open(parameters_loading_path, "r") as output_file:
        parameters = json.load(output_file)
    return parameters
