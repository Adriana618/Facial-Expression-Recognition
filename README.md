# Facial-Expression-Recognition
The following repository describes a framework for testing different types of neural network under different types of training with different databases.

## Installation

To standardize the process, we use conda environments. Run this command to install conda enviroment:

`conda create --name Faces --file requirements.txt`

## Run a model
To run a model that we have defined, we must execute the following command.

`python train.py`

If we want to evaluate a model, we must run the following command:

`python eval.py`
## Settings
In the config.ini file we can define which model will be used, as well as the database and the training to follow.

### Model
* name: The name of your model, should be the same name as your model class.
* model_file: The file where your model implementation is located. It should always be inside the models folder and should start with the prefix ***_models_***

### Train
* enable: If we want the model to be trainable.
* epochs: Number of epochs in which the model will be trained.
* checkpoints_path: Path, where the checkpoints of the model are saved.
* train_file: The file where your train implementation is located. It should always be inside the trainings folder and should start with the prefix ***_trainings_***
* save_step: Interval of epochs in which a checkpoint is generated.

### Dataset
* name: The name of your dataset.
* dataset_file: The file where your dataset implementation is located. It should always be inside the datasets folder and should start with the prefix ***_datasets_***.
Remember that in the file where your database is implemented, there must be a function called get_dataset that returns three DataLoaders: Train, Valid and Test. In addition to classes to classify. In short it should be something like this:
```python
def get_dataset(***):
    ***
    return (train_loader, valid_loader, test_loader, classes)
```
### Eval
* enable: If you want the model to be evaluated.
* model_to_eval: Name of the model to evaluate.

### Metrics
* confussion_matrix: If you want to generate the confusion matrix when evaluating the model.
* graphics: If you want to generate graphs of loss and accuracy at the end of the training.


## Useful information just for the owner of the repository.


Enlace a la carpeta de papers leídos: [https://drive.google.com/drive/folders/1A5JLgfl8_e3SZfaZDI5EaPcAu6lSb1Vm?usp=sharing]

Enlace a la carpeta con las bases de datos: [https://drive.google.com/drive/folders/1JHqI0sIV6AYSanFXWZ1oIwKIlzzRFLNp?usp=sharing]
