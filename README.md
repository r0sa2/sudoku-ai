# Sudoku AI

This repository contains the implementation of an augmented reality (AR) Sudoku solver for images and video, and the implementation and comparison of different algorithms to solve the classic 9x9 Sudoku.

## Directory Structure
- `run.py`: Running script for the AR Sudoku solver for real-time video
- `run_image.ipynb`: Demo of using the AR Sudoku solver for images
- `evaluate_on_sudoku_dataset.ipynb`: Evaluation script for the Sudoku dataset
- `sudoku_dataset/`: Sudoku dataset
- `ar_sudoku_solver/`
    - `ar_sudoku_solver.py`: AR Sudoku solver
    - `solver.py`: Dancing links Sudoku solver
    - `digit_classifier/`
        - `train.ipynb`: Digit classification model training notebook
        - `dataset/`
            - `char74k/`: Char74k dataset. The dataset has been excluded due to size constraints.
                - `preprocessing.py`: Preprocessing script for Char74k
            - `mnist/`: MNIST dataset. The dataset has been excluded due to size constraints.
                - `preprocessing.py`: Preprocessing script for MNIST
        - `model/`: Trained models
            - `char74k.h5`: Model trained on Char74k
            - `mnist.h5`: Model trained on MNIST
    - `algorithms/`: Implementation and comparison of different Sudoku algorithms

## Usage
The code has been written to work with Python3.11.6. Assuming a virtual environment is being used, begin by installing requirements.
```
> pip install -r requirements.txt
```
For real-time video solving with webcam input (with device index `n`), run
```
python run.py -d n
```
For a demo of how to use the system for images, check `run_image.ipynb`.
