import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .utils import CustomDataset, trainModel, gatherInference


class ShadowModel:
    """Shadow models architecture implementation based on
       the paper https://arxiv.org/abs/1610.05820
    """
    def __init__(self, listOfModels, trainDataSize, randomSeed):
        """
          Constructing a shadow model architecture
            Args:
                listOfModels (list): list of different shadow models
                trainDataSize (list): number of examples per training set for
                                      each shadow model
                randomSeed (int): random seed for every training sample split
            Attributes:
                listOfModels (list): list of different shadow models
                numOfModels (int): the number of shadow models
                trainDataSize (int): number of examples per training set for
                                     each shadow model
                trainLoader (torch.DataLoader): data loader for each model
                seed (int): random seed for training sample split
                rdst (np.random.RandomState): random state constructed by seed
        """
        self.listOfModels = listOfModels
        self.numOfModels = len(listOfModels)

        self.trainDataSize = trainDataSize
        self.trainLoaders = []
        self.testLoaders = []

        self.seed = randomSeed
        self.rdSt = np.random.RandomState(self.seed)

    def fit(self, X, y, numIter):
        """Train all member model of a shadow model on training data

          Args:
              X (torch.Tensor): Training input for shadow models
              y (torch.Tensor): Training true label for shadow models
              numIter (int): number of epoch for every shadow models
          Returns:
              None
        """
        dataLength = len(X)
        for model_idx in range(self.numOfModels):

            indices = self.rdSt.choice(a=dataLength,
                                       size=2 * self.trainDataSize,
                                       replace=False)
            trainIndices = indices[:self.trainDataSize]
            testIndices = indices[self.trainDataSize:]

            X_train, y_train = X[trainIndices], y[trainIndices]
            X_test, y_test = X[testIndices], y[testIndices]

            train_datasets = CustomDataset(X_train, y_train)
            train_dataloader = DataLoader(dataset=train_datasets,
                                          batch_size=5,
                                          shuffle=True)

            test_datasets = CustomDataset(X_test, y_test)
            test_dataloader = DataLoader(test_datasets, batch_size=5,
                                         shuffle=True)

            self.trainLoaders.append(train_dataloader)
            self.testLoaders.append(test_dataloader)

            model = self.listOfModels[model_idx]
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

            model = trainModel(model, train_dataloader, numIter, loss_fn, optimizer)
            print(f"Finished training model {model_idx}")

            self.listOfModels[model_idx] = model

    def transform(self):
        """Get the label data ("in", "out" ) for the attacker

            Args:

            Returns:
                shadow_in_true_label (torch.Tensor): True labels of all the training data
                shadow_in_prediction (torch.Tensor): Prediction made by shadow models on
                                                     training data
                shadow_out_true_label: True labels of all the non-training data
                shadow_out_predicition: Prediction made by shadow models on non-training data
        """

        shadow_in_true_label = []
        shadow_in_prediction = []
        shadow_out_true_label = []
        shadow_out_prediction = []

        # Get the labeled data ("in" , "out") for the attacker
        for model_idx in range(self.numOfModels):
            model = self.listOfModels[model_idx]

            trainLoader = self.trainLoaders[model_idx]
            testLoader = self.testLoaders[model_idx]

            inTrueLabel, inPredLabel = gatherInference(model, trainLoader)
            outTrueLabel, outPredLabel = gatherInference(model, testLoader)

            shadow_in_true_label.append(inTrueLabel)
            shadow_in_prediction.append(inPredLabel)
            shadow_out_true_label.append(outTrueLabel)
            shadow_out_prediction.append(outPredLabel)

        shadow_in_true_label = torch.cat(shadow_in_true_label)
        shadow_in_prediction = torch.cat(shadow_in_prediction)
        shadow_out_true_label = torch.cat(shadow_out_true_label)
        shadow_out_prediction = torch.cat(shadow_out_prediction)

        return (shadow_in_true_label, shadow_in_prediction, shadow_out_true_label, shadow_out_prediction)

    def transformClass(self, shadow_in_true_label, shadow_in_pred,
                       shadow_out_true_label, shadow_out_pred):
        """Group the labeled data ("in", "out") into different true classes

            Args:
                shadow_in_true_label(torch.Tensor): true lable of input dataset
                    shape = [N x K]
                    N is number of training samples,
                    K is number of classes
                shadow_in_pred(torch.Tensor): prediction of input dataset
                    shape = [N x K]
                shadow_out_true_label(torch.Tensor): true label of non-input dataset
                    shape = [N x K]
                shadow_out_pred(torch.Tensor): prediction of non-input dataset
            Returns:
                result(dict) [keys : values]

                keys(int): the true classes' values (from 1 -> K)

                values(tuple): (in, out)
                    in(torch.Tensor) : prediction vector of input dataset
                        shape = [N x K]
                    out(torch.Tensor) : prediction vector of non-input dataset
                        shape = [N x K]
        """

        # Get the unique labels in training/non-training dataset

        shadow_in_true_label = torch.nonzero(shadow_in_true_label)[:, 1]
        shadow_out_true_label = torch.nonzero(shadow_out_true_label)[:, 1]
        unique_labels = torch.unique(shadow_in_true_label)
        result = {}
        for current_label in unique_labels:
            current_label_in_indices = (shadow_in_true_label == current_label)
            current_label_in_pred = shadow_in_pred[current_label_in_indices]
            current_label_out_indices = (shadow_out_true_label == current_label)
            current_label_out_pred = shadow_out_pred[current_label_out_indices]

            # if(current_label_in_prediction.shape != torch.Size([0]) or
            #    current_label_out_prediction.shape != torch.Size([0])):

            result[current_label.item()] = (current_label_in_pred,
                                            current_label_out_pred)

        return result

    def fitTransform(self, X, y, numIter):
        """Train shadow model on a custom dataset, then generate training dataset
          into different classes for attacker model

            Args:
                X (torch.Tensor): Shadow model training input
                y (torch.Tensor): Shadow model training true label
                numIter : number of Epochs for every shadow models
            Returns:
                result(dict) : [keys : values]

                keys(int) : the true classes' values (from 1 -> K)

                values(tuple) : (in, out)
                    in(torch.Tensor) : prediction vector of input dataset
                        shape = [N x K]
                    out(torch.Tensor) : prediction vector of non-input dataset
                        shape = [N x K]

        """

        self.fit(X, y, numIter)

        (shadow_in_true_label,
         shadow_in_prediction,
         shadow_out_true_label,
         shadow_out_prediction) = self.transform()

        result = self.transformClass(shadow_in_true_label, shadow_in_prediction,
                                     shadow_out_true_label, shadow_out_prediction)
        for key, (In, Out) in result.items():
            attack_train_data = torch.cat((In, Out))
            attack_train_label = torch.cat(
                (torch.ones(In.shape[0]),
                 torch.zeros(Out.shape[0]))
            )
            result[key] = (attack_train_data, attack_train_label)

        return result


if __name__ == '__main__':
    print("Hello from ShadowModel.py")
