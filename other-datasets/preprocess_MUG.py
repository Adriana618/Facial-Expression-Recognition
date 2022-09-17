# labels 0=anger 1=disgust, 2=fear, 3=happy, 4=sadness, 5=surprise, 6=neutral


import csv
import os
import numpy as np
import random
from config import MUG_path, training_ratio


anger_path = os.path.join(MUG_path, "anger")
disgust_path = os.path.join(MUG_path, "disgust")
fear_path = os.path.join(MUG_path, "fear")
happy_path = os.path.join(MUG_path, "happiness")
sadness_path = os.path.join(MUG_path, "sadness")
surprise_path = os.path.join(MUG_path, "surprise")
neutral_path = os.path.join(MUG_path, "neutral")

foldspath = os.path.join(MUG_path, "Folds")
if not os.path.exists(foldspath):
    print("fold creado")
    os.makedirs(foldspath)

train_data = []
test_data = []


def extract_emotion(emotion_path, id):
    global train_file, test_file, training_ratio
    files = os.listdir(emotion_path)  # Chambeando con los angry
    tam = len(files)
    train_len = int(tam * training_ratio)
    i = 0
    while i < train_len:
        train_data.append(os.path.join(emotion_path, files[i]) + " " + str(id) + "\n")
        i += 1
    while i < len(files):
        test_data.append(os.path.join(emotion_path, files[i]) + " " + str(id) + "\n")
        i += 1


def list_to_file(lista, path_file):
    file = open(path_file, "w")
    for line in lista:
        file.write(line)
    file.close()


extract_emotion(anger_path, 0)
extract_emotion(disgust_path, 1)
extract_emotion(fear_path, 2)
extract_emotion(happy_path, 3)
extract_emotion(sadness_path, 4)
extract_emotion(surprise_path, 5)
extract_emotion(neutral_path, 6)

random.shuffle(train_data)
random.shuffle(test_data)

list_to_file(train_data, os.path.join(foldspath, "train_file.txt"))
list_to_file(test_data, os.path.join(foldspath, "test_file.txt"))
