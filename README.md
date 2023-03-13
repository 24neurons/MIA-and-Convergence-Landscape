# Membership Inference Attack

This project is ian experiment on membership attack against machine learning model based on the variation in learning rates.

## To-do lists
1. Add non-image data utilities
2. Add more target models to both type of data
3. Allow shadow model learning rate configuration
4. Add BenchMark architecture to attack models

## Requirements

This implementation is implemented based on [Pytorch 1.13.1](https://pytorch.org/). Please refer to [requirements.txt](requirements.txt) for other dependencies.

## Datasets

Including the dataset supported by the implementation. The dataset will be automatically downloaded into your "data" folder.

- Tabular data: [Purchase Dataset](https://www.kaggle.com/datasets/raosuny/e-commerce-purchase-dataset), 
- Image data : [CIFAR10, CIFAR100](https://www.kaggle.com/datasets/fedesoriano/cifar100)

## Running experiments
### API
For the moment, the only API available is for image dataset, namingly CIFAR(CIFAR10/CIFAR100)
```
cd src/images/cifar
python3 main.py \ 
--dataset "dataset name" \
--target_model "name of target model" \
--target_size "Size of training and testing set of target model"
--target_lr "learning rate of target model"
--attack_model "name of attacking model against target model"
```

- The following target_model are supported: 
  `'resnet20', 'resnet50',  'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'`
- The following attack_model are supported: `'NN'`
