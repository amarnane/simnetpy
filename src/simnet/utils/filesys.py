import pathlib
import tarfile
import shutil
import yaml
import warnings
import json
import pickle


import numpy as np


from datetime import datetime

# resolve to make sure full path, parents[1] as file expected to be SRCDIR/utils/utils.py
SRCDIR = (pathlib.Path(__file__).resolve()).parents[1] 


def create_dirs_on_path(f, create_parent_if_file=True):
    """
    function to create directories on given path if don't exist. Can be file or directory. 
    If file needs create_parent_if_file flag

    Args:
        f (pathlib.Path or str): path to create dict on. can be dictionary or file
        create_parent_if_file (bool, optional): if f is a file create parent directory. Defaults to True.

    Raises:
        NotADirectoryError: raises and error if f is file that has no suffix

    Returns:
        pathlib.Path: path with all directories created
    """
    p = pathlib.Path(f)
    if p.suffix != '':
        if create_parent_if_file:
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        else:
            raise NotADirectoryError
    
    p.mkdir(parents=True, exist_ok=True)
    return p

    
def tardir(src, dst):
    if src is None:
        return

    with tarfile.open(dst, 'w:gz') as tar:
        for p in src.rglob('*.py'):
            tar.add(p,arcname=p.relative_to(src))

def copyfile(src, dst):
    shutil.copy(src, dst)

def copy_src_to_folder(exppath, src=SRCDIR):
    dst = exppath/ "code-chkpt.tar.gz"
    tardir(src, dst)


def append_timestamp2path(path, timeformat="%Y-%m-%d-%H_%M_%S"):
    path = pathlib.Path(path)
    timestamp = datetime.now().strftime(timeformat)
    path = path.parent / (path.stem + '-'+timestamp)
    return path


def create_experiment_folder(path, timestamp=True, config_path=None, copysrc=True, src=SRCDIR):
    """Create experiment folder. Append timestamp to make name unique, copy src code to folder. copy config to folder

    Args:
        path (str or pathlib.Path): name/path for experiment output
        src (_type_, optional): location of source code that generated the model. Defaults to HOME/phd/phd-year2/src/phd_year2.
        config (str or pathlib.Path): location of config file.
    """
    if timestamp:
        # add timestamp
        path = append_timestamp2path(path)

    # create folder
    path = create_dirs_on_path(path)

    if copysrc:
        # copy source code
        copy_src_to_folder(path, src=src)

    if config_path is not None:
        copyfile(config_path, path)
    return path

def create_output_path(config):
    home = config['output']['home']
    folder = config['output']['folder']
    name = config['output']['name']
    p = pathlib.Path(home) / folder / name
    return p


def create_experiment(config_path):
    # config_path = config # path is argument
    config = load_config(config_path)

    # create save location and copy config
    outputfolder = create_output_path(config)
    outpath = create_experiment_folder(outputfolder, config_path)

    return outpath, config


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            # if obj.flags['C_CONTIGUOUS']:
            #     obj_data = obj.data
            # else:
            #     cont_obj = np.ascontiguousarray(obj)
            #     assert(cont_obj.flags['C_CONTIGUOUS'])
            #     obj_data = cont_obj.data
            # # data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=obj.tolist(),
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        return json.JSONEncoder.default(self, obj)

def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        # data = base64.b64decode(dct['__ndarray__'])
        data = dct['__ndarray__']
        return np.array(data, dct['dtype']).reshape(dct['shape'])
    return dct

def json_save(data, f, outpath):
    outpath = pathlib.Path(outpath)
    p = outpath / f
    with open(p, 'w', encoding='utf-8') as ff:
        json.dump(data, ff, ensure_ascii=False, indent=2, cls=NumpyEncoder)


def json_load(f, outpath):
    outpath = pathlib.Path(outpath)
    p = outpath / f
    # with open(p, 'w', encoding='utf-8') as ff:
    with open(p, 'r') as ff:
        data = json.load(ff, object_hook=json_numpy_obj_hook)
    return data

def pickle_save(obj, f, outpath):
    outpath = pathlib.Path(outpath)
    p = outpath / f
    with open(p, 'wb') as ff:
        # json.dump(data, ff, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        pickle.dump(obj , ff)
    
def load_config(path):
#     """Method to load config dictionary from given config yml"""
    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def save_config(path, config):
    with open(path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)



