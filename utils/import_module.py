from importlib import import_module
import logging
import glob
import re
import os


def verify(module, module_path):
    chs = glob.glob(os.path.join(module_path, "*.py"))
    models = [re.search(r"\\(.+?).py", ch).group(1) for ch in chs]
    if module["name"] not in models:
        logging.error("Model not found")
    else:
        logging.info("Model charged")


def charge_dynamic_class(classname, path_to_module):
    try:
        module = import_module(path_to_module)
        _class = eval("module.{}".format(classname))
        logging.info("Class {} succesfully charged".format(path_to_module))
        return _class
    except ModuleNotFoundError:
        logging.error("File {} not found".format(path_to_module))
    except AttributeError:
        logging.error("Class {} not found".format(classname))


def charge_dynamic_class_old(classname, path_to_module, folder="models"):
    try:
        module = import_module("{}.{}".format(folder, path_to_module))
        _class = eval("module.{}".format(classname))
        logging.info("Class {} succesfully charged".format(path_to_module))
        return _class
    except ModuleNotFoundError:
        logging.error("File {} not found".format(path_to_module))
    except AttributeError:
        logging.error("Class {} not found".format(classname))


def charge_dynamic_function(function, path_to_module):
    try:
        module = import_module(path_to_module)
        _function = eval("module.{}".format(function))
        logging.info("Function {} succesfully charged".format(path_to_module))
        return _function
    except ModuleNotFoundError:
        logging.error("File {} not found".format(path_to_module))
    except AttributeError:
        logging.error("Function {} not found".format(function))
