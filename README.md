# CS3612-ML-Project
Project of CS3612: Machine Learning, Spring 2023

## Setup
```sh
conda create -n ml-project python=3.11
activate ml-project
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib
conda install tqdm
```

## Run
The Python files can be directly run with default parameters.
### Task 1 Fashion-MNIST Clothing Classification
+ Run `ClothingClassification/main.py` to train the model. Results will be saved to `ClothingClassification/checkpoints` by default.
+ Run `ClothingClassification/visualize.py` to visualize the results. The directory of the results can be specified by the parameter `model_dir` (`best` by default).

For more details of the files refer to `ClothingClassification/README.md`.

### Task 2 Image Reconstruction
+ Run `ImageReconstruction/main.py` to train the model. Results will be saved to `ImageReconstruction/checkpoints` by default.
+ Run `ImageReconstruction/visualize.py` to visualize the results. The directory of the results can be specified by the parameter `model_dir` (`best` by default).

For more details of the files refer to `ImageReconstruction/README.md`.
