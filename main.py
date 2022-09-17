import configparser
import logging
import glob
import os
import re

# model - train - dataset - metrics
config = configparser.ConfigParser()
config.read('config.ini')

models_path = "models"

def verify(module):
    chs = glob.glob(os.path.join(models_path, '*.py'))
    models = [re.search(r'\\(.+?).py', ch).group(1) for ch in chs]
    if config[module]["name"] not in models:
        logging.error("Model not found")
    else:
        logging.info("Model charged")

verify("model")