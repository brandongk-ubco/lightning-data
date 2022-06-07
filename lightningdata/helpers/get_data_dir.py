import os


def get_data_dir(dataset_name):
    data_dir = os.environ.get("OVERRIDE_DATA_DIR", os.environ["DATA_DIR"])
    if not data_dir:
        raise ValueError(
            "Must set data_dir, either through command line or DATA_DIR environment variable"
        )
    return os.path.join(os.path.abspath(data_dir), dataset_name)
