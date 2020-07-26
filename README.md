# Handwritten Text Recognition using Tensorflow 2.x

This project is reimplantation of the [SimpleHTR](https://github.com/githubharald/SimpleHTR)
in TensorFlow 2.x. The model is build using TensorFlow's object orient API.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Install Anaconda in your system

### Installing

From Anaconda prompt create a conda environment using the requirement.txt file or 

```
conda create --name <env_name> --file requirements.txt
```

Activate the environment

```
conda activate <env_name>
```
download the project to your local machine.

## Running the tests

from the project ```src``` directory run commands

### Train the model

Download the IAM dataset containing the training images, unzip it and palace the ```words``` directory inside the ```data``` directory.
for more information see [here](https://github.com/githubharald/SimpleHTR#train-model)

```python main.py --train```

### Validate the model

```python main.py --validate```

### Infer with  the model

```../data/test.png``` is used for the inference or change the file path in FilePaths class in ```src/model_helpler.py```

```python main.py```
