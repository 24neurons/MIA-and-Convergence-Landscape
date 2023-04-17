# Membership Inference Attack

This project is an experiment on membership attack against machine learning model based on the variation in learning rates.

## To-do lists

1. Add non-image data utilities
2. Add more target models to both type of data
3. Allow shadow model learning rate configuration
4. Add BenchMark architecture to attack models

## Documentation

1.[```TargetModel```](src/utils/base_models.py): A target model wrapper class around the actual target model such as vgg19 or resnet20
  I.e TargetModel(ResNet20) with ResNet20 being the base target model
  This class functions in these sequential steps

* Inserting dataset with format (input, label), both input and label has the type of torch.Tensor
* Training target model on train_set
* Generating attack data (using both train_set and test_set) for the attack model in the form
  (pred, membership_status) with membership_status being 1 if it is training sample, 0 otherwise
* Estimating sharpness of the target model:
  This function randomly choose 5-20 samples from the training set.
  On each random samples, train the model further, getting the weights and loss values of the new model.
  Calculate the sharpness of the model on current weight with the formula :
  ```math
  sharpness = \frac{\text{increases_in_loss}}{\text{increases_in_weight}}
  ```
2. [```ShadowModel```](src/utils/base_models.py): A shadow model wrapper for a list of shadow models (nModels), with the same architecture with base target model
  This class also functions in similar manner with the TargetModel, only that this class conteains many different models, not just
  only one like TargetModel. It operates in these sequential steps

* Inserting dataset with the same format as TargetModel
* Dividing the inserted dataset into n_models distinct set with equal size ```train_size```

3. [```AttackModel```](src/utils/base_models.py): An attacker model wrapper around the attack models (nModels), with each model is used to attack on different labels
  I.e If the original dataset contains 10 labels, then AttackModel, then AttackModel contains 10 different sub attack model.
  The class allows:

* Inserting dataset as a dict with the format ```(key, [model_pred, membership_status])```
* Train the ith attack model on the ith dataset of the provided dataset
* Generating the membership status when being provided true label and model prediction

4. [```FullAttacker```](src/images/cifar/main.py): This class intergrates all the above class into one and performs attacking action

## Requirements

This implementation is implemented based on [Pytorch 1.13.1](https://pytorch.org/). Please refer to [requirements.txt](requirements.txt) for other dependencies.

## Datasets

Including the dataset supported by the implementation. The dataset will be automatically downloaded into your "data" folder.

* Tabular data: [Purchase Dataset](https://www.kaggle.com/datasets/raosuny/e-commerce-purchase-dataset),
* Image data : [CIFAR10, CIFAR100](https://www.kaggle.com/datasets/fedesoriano/cifar100)

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
```

* The following target_model are supported:
  `'resnet20', 'resnet50',  'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'`
* The following attack_model are supported: `'NN'`

### Example command:

```
cd src/images/cifar
python3 main.py \
--dataset "CIFAR10" \
--target_model "resnet20" \
--target_size 10000 \
--target_lr 0.1 \
--target_epoch 100
```

With the above command, the program will run the experiment on CIFAR10 dataset with ResNet20 as the target model,
the executing time running on Kaggle GPU T4x2 is around 1 hour.
