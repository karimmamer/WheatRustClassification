# Wheat Rust Classification from Ensemble Selection of CNNs

Fourth place solution for CGIAR Computer Vision for Crop Disease competition organized by CV4A workshop at ICLR 2020. The main objective of the competition is to classify a given image of wheat whether it is healthy, has a stem rust or has a leaf rust.

## Summary of Approach

Create an ensemble from a library of diverse models with different architectures and augmentations. All models are initially pre-trained on imagenet and fine-tuned on the dataset. The models and augmentations are chosen automatically using hyperparameter optimization.

### Model Architectures

The following architecturs are included in the library of models:

* ResNet [1]
* ResNext [2]
* WideResNet [3]
* DenseNet [4]

### Data Augmentations

The following augmentations are included in the search space of hyperparameter optimization to choose from:

* Rotation
* Random cropping and resizing
* Horizontal flipping
* Vertical flipping
* Brightness augmentation
* Hue augmentation
* Contrast augmentation
* Mixup augmentation [5]

### Common Configuration

The following configurations is applied on all trails in hyperparameter optimization process:

* Stochastic Gradient Descent (SGD) optimizer
* Snapshot ensemble [6]
* 5-Fold training

## Getting Started

### Prerequisites

Firstly, you need to have 

* Ubuntu 18.04 
* Python3
* At least 11 GB GPU RAM

Secondly, you need to install the challenge data and sample submission file by the following the instructions [here](https://zindi.africa/competitions/iclr-workshop-challenge-1-cgiar-computer-vision-for-crop-disease/data).

Thirdly, you need to install the dependencies by running:

```
pip3 install -r requirements.txt
```

### Project files

* prepare_dataset.py: reads training and test data, removes duplicates from training data and saves them in numpy matrices. It has the following arguments:

    --train_data_path: path to training data folder

    --test_data_path: path to test data folder

    --save_path: save path for training and test numpy matrices of images

* generate_library_of_models.py: generates a library of models with different architectures and augmentations through hyperparameter optimization search. It has the following arguments: 

    --data_path: path to training and test numpy matrices of images

    --library_size: number of models to be trained in the library of models

    --library_path: save path for validation and test predictions of the library of models

* ensemble_selection.py: applies Ensemble Selection [7] algorithm on the generated library of models to find the best ensemble with the lowest validation error and use it to create the final submission. It has the following arguments:

    --train_data_path: path to training data folder

    --data_path: path to training and test numpy matrices of images

    --sample_sub_file_path: path to sample submission file

    --library_size: number of models to be trained in the library of models

    --library_path: save path for validation and test predictions of the library of models

    --final_sub_file_save_path: save path for final submission file

* dataset.py: has the dataset class for training and test data.

* utils.py: utility functions for training, testing and reading dataset images. 

## Running

### 1- Prepare dataset

```
python3 prepare_dataset.py
```

### 2- Generate the library of models

```
python3 generate_library_of_models.py
```

### 3- Create ensemble and generate submission file

```
python3 ensemble_selection.py
```

## References
[1] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[2] Xie, Saining, et al. "Aggregated residual transformations for deep neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[3] Zagoruyko, Sergey, and Nikos Komodakis. "Wide residual networks." arXiv preprint arXiv:1605.07146 (2016). 

[4] Huang, Gao, et al. "Densely connected convolutional networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[5] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).

[6] Huang, Gao, et al. "Snapshot ensembles: Train 1, get m for free." arXiv preprint arXiv:1704.00109 (2017).

[7] Caruana, Rich, et al. "Ensemble selection from libraries of models." Proceedings of the twenty-first international conference on Machine learning. 2004.

