wft21_septum_landmark_detection
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like 'make environment' or 'make requirement'
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── metadata       <- Excel and csv files with additional metadata
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── predicted      <- Model predictions, will be used for the evaluations
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │   ├── Dataset        <- call the dataset helper functions, analyze the datasets
    │   ├── Evaluate       <- Evaluate the model performance, create plots
    │   ├── Predict        <- Use the models on new data
    │   ├── Train          <- Train a new model
    │   └── Test_IO        <- IO tests
    │   └── Test_Models    <- Tensorflow functional or subclassing tests
    │
    ├── exp            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── configs        <- Experiment config files as json
    │   ├── figures        <- Generated graphics and figures to be used in reporting
    │   ├── history        <- Tensorboard trainings history files
    │   ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── tensorboard_logs  <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Helper functions that will be used by the notebooks.
        ├── data           <- create, preprocess and extract the nrrd files
        ├── models         <- Modelzoo, Modelutils and Tensorflow layers
        ├── utils          <- Metrics, callbacks, io-utils, notebook imports
        └── visualization  <- Plots for the data, generator or evaluations
Paper:
--------
- link to paper if accepted

- info for cite

Setup native with OSX or Ubuntu
------------
### Preconditions: 
- Python 3.6 locally installed 
(e.g.:  <a target="_blank" href="https://www.anaconda.com/download/#macos">Anaconda</a>)
- Installed nvidia drivers, cuda and cudnn 
(e.g.:  <a target="_blank" href="https://www.tensorflow.org/install/gpu">Tensorflow</a>)

### Local setup
- Clone repository
```
git clone %reponame%
cd %reponame%
```
- Create a virtual environment either via virtualenv or conda
```
make environment
```
- Install dependencies via requirements.txt
```
make requirement
```
