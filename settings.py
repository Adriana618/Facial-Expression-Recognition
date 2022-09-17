import torch
import configparser
from utils.import_module import charge_dynamic_class, charge_dynamic_function
import logging

config = configparser.ConfigParser()
config.read('config.ini')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model settings
model = charge_dynamic_class(config["model"]["name"], config["model"]["model_file"])

# dataset settings
dataset = charge_dynamic_class(config["dataset"]["name"], config["dataset"]["dataset_file"])
get_dataset = charge_dynamic_function("get_dataset", config["dataset"]["dataset_file"])

# train settings
epochs = int(config["train"]["epochs"])
checkpoints_path = config["train"]["checkpoints_path"]
train_step = charge_dynamic_function("train_step", config["train"]["train_file"])
valid_step = charge_dynamic_function("valid_step", config["train"]["train_file"])
save_step = int(config["train"]["save_step"])

# eval settings


# metrics settings
confussion_matrix = config["metrics"].getboolean("confussion_matrix", False)
graphics = config["metrics"].getboolean("graphics", False)
