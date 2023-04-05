"""
This module contains all the base class wrapper for three types of models:
target model, shadow model, attack model
"""
from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy

from tqdm.auto import tqdm
import numpy as np

from .helper import gather_inference, CustomDataset, train_step, test_step


class TargetModel:
    def __init__(self, base_target):
        """
        The target model wrapper for base target model
        Attributes: 
            tg (nn.Module): The target model to be attacked
            trainloader (DataLoader): The dataloader for training dataset
            testloader (TestLoader):The testdataloader for test dataset
        """
        self.tg = base_target
        self.trainloader = None
        self.testloader = None
    def set_dataloader(self, X, y, partition = 0.5):
        """
        Setting dataloader for target model 
        Args: 
            X: The input data for model as a float32 tensor of shape [n_samples, (input shapes)]
            y: The label data for model as a float32 tensor of shape [n_samples, (label shapes)]
        """
        train_size = len(X) * partition

        X_train, y_train = X[0:train_size] , y[0:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        trainset = CustomDataset(X_train, y_train)
        testset = CustomDataset(X_test, y_test)

        self.trainloader = DataLoader(trainset, batch_size = 4)
        self.testloader = DataLoader(testset, batch_size = 4)
    def _fit(self, X, y, epochs=20, lr=0.02, partition=0.5):
        """
        Fitting the model on specific dataset with specified epochs and learning rate
        Args:
            X (torch.Tensor): The input data as a tensor of shape [n_samples , (input_shape)]
            y (torch.Tensor):The label data as a tensor of shape [n_samples , (label_shape)]
            epochs (int): The number of epochs that this model should be trained for, default is 20
            lr (torch.float32): The constant learning rate for the model

        Returns: None. Only log the loss and accuracy to the console. 
        """
        # First, setting the dataloader for training and testing data
        self.set_dataloader(X, y, partition)
        
        loss_fn = nn.CrossEntropyLoss()
        metric_score_fn = BinaryAccuracy()
        optimizer = torch.optim.SGD(self.tg.parameters(), lr=lr)

        #training data in epochs
        for _ in tqdm(range(epochs)):
            self.tg , train_loss, train_acc = train_step(self.tg, self.trainloader, loss_fn,
                                                        metric_score_fn, optimizer)
            test_loss, test_acc = test_step(self.tg, self.testloader, loss_fn, metric_score_fn)

            print(f"Training on Epochs {_} | Train loss: {train_loss} | Test loss {test_loss}")
            print(f"Train Accuracy: {train_acc} | Test accuracy {test_acc}")
    def _transform(self):
        """
        Gathering model inference of the provided data along with the true label of provided data.
        All the true label Tensor are in one-hot encoding format.
        Returns:
            in_true_label (torch.Tensor): True label of member data, shape [n_train_samples, n_classes]
            in_pred_label (torch.Tensor): Pred label of member data, shape [n_train_samples, n_classes]
            out_true_label (torch.Tensor): True label of non-member data, shape [n_test_samples, n_classes]
            out_true_label (torch.Tensor): Pred label of non-member data, shape [n_test_samples, n_classes] 
        """

        in_true_label, in_pred_label = gather_inference(self.tg, self.trainloader)
        out_true_label, out_pred_label = gather_inference(self.tg, self.testloader)
        return (in_true_label, in_pred_label, out_true_label, out_pred_label)
    def fit_transform(self, X, y, epochs, lr, partition=0.5):
        """
        Train model on dataset and generate attack datata for attack model
        Args:
            X (torch.Tensor): Input data with shape [n_samples, (input_shape)]
            y (torch.Tesnor): Label data with shape [n_samples, n_classes]
            epochs (int): number of epochs the target model should be trained for
            lr (torch.float32): learning rate of the model 
            partition (torch.float32): The ratio of training samples versus whole datset
        Returns:
            attack (dict): a dict of key : (pred, membership_status) with
                        -key: True label 
                        -pred: model prediction of that instance
                        -membership_status: 1 if that instance was used to be trained, 0 otherwise 
        """
        self._fit(X, y, epochs, lr,partition)

        (in_true_label, in_pred_label, 
         out_true_label, out_pred_label) = self._transform()

        # Get all the unique labels in int form
        unique_labels = torch.arange(len(X))

        prediction = {}
        attack_data = {}

        for c_label in unique_labels:
            in_indices = (in_true_label == c_label)
            in_pred = in_pred_label[in_indices]
            out_indices = (out_true_label == c_label)
            out_pred = out_pred_label[out_indices]

            prediction[c_label] = (in_pred, out_pred) 
        
        for key, (In, Out) in prediction.items():
            attack_test_X = torch.cat((In, Out))
            attack_test_y = torch.cat(
                (torch.ones(In.shape[0]),
                torch.zeros(Out.shape[0]))
            )
            attack_data[key] = (attack_test_X, attack_test_y)
        
        return attack_data 
 

class ShadowModel:
    """Shadow models architecture implementation based on 
       the paper https://arxiv.org/abs/1610.05820
    """
    def __init__(self, target_model, nModels, randomSeed = 54):
        """
          Constructing a shadow model architecture
            Args: 
                target_model (nn.Module): The target model original architecture
                nModels (int): The number of shadow models inside 
            Attributes:
                list_of_models (list): list of different shadow models
                numOfModels (int) : the number of shadow models

                trainloaders (list): list of train dataloader for each model
                testloaders (list): list of test dataloader for each model 
                seed (int): random seed for training sample split
                rdst (np.random.RandomState): random state constructed by seed
        """
        self.list_of_models = []
        for i in range(nModels):
            c_model = deepcopy(target_model)
            self.list_of_models.append(c_model)
        self.numOfModels   = nModels

        self.trainloaders  = []
        self.testloaders = []
        self.train_losses = None
        self.test_losses = None
        
        self.seed = randomSeed
        self.rdSt  = np.random.RandomState(self.seed)
        self._init_weights()

    def _init_weights(self):
        """Initializing weights for each shadow models
           Every shadow models have the same kind of architecture
           but different weights
        """
        for model in self.list_of_models:
            model._init_weights()
    def set_dataloader(self, X , y, train_size):
        total_length = len(X)

        for model_idx in range(self.numOfModels):
            
            indices = self.rdSt.choice(a=total_length,
                                       size= 2 * train_size,
                                       replace=False)
            train_indices = indices[0:train_size]
            test_indices = indices[train_size:]

            X_train_idx , y_train_idx = X[train_indices], y[train_indices]
            X_test_idx, y_test_idx = X[test_indices], y[test_indices]

            trainset = CustomDataset(X_train_idx, y_train_idx)
            testset = CustomDataset(X_test_idx, y_test_idx)

            trainloader_idx = DataLoader(trainset, batch_size=4,
                                         shuffle=True)
            testloader_idx = DataLoader(testset, batch_size=4,
                                        shuffle=True)
            self.trainloaders.append(trainloader_idx)
            self.testloaders.append(testloader_idx)
        
    def _fit(self, X, y, train_size, epochs):
        """Train all member model of a shadow model on training data

          Args:
              X (torch.Tensor): Training input for shadow models
              y (torch.Tensor): Training true label for shadow models
              epochs (int): number of epoch for every shadow models
          Returns: 
              None
        """
        
        self.train_losses = np.zeros(epochs)
        self.test_losses = np.zeros(epochs)

        self.set_dataloader(X, y, train_size)

        for model_idx in range(self.numOfModels):

            model = self.list_of_models[model_idx]
            loss_fn = torch.nn.CrossEntropyLoss()
            metric_score_fn = BinaryAccuracy()
            optimizer = torch.optim.SGD(model.parameters() , lr = 0.001)
            trainloader_idx = self.trainloaders[model_idx]
            testloader_idx = self.testloaders[model_idx]
            
            train_losses = []
            test_losses = []
            for _ in tqdm(range(epochs)):
                model, train_loss, train_acc = train_step(model, trainloader_idx,loss_fn,
                                               metric_score_fn, optimizer)
                test_loss, test_acc = test_step(model, testloader_idx, metric_score_fn, loss_fn)

                train_losses.append(train_loss)
                test_losses.append(test_loss)
            
            print(f"Finished training model {model_idx}")
            self.list_of_models[model_idx] = model
            self.train_losses += np.array(train_losses)
            self.test_losses  += np.array(test_losses)

        self.train_losses /= self.numOfModels
        self.test_losses /= self.numOfModels
    def _transform(self):
        """Get the label data ("in", "out" ) for the attacker

            Args: 

            Returns : 
                -shadow_in_true_label(torch.Tensor): true lable of input dataset
                in one-hot encoding format
                    shape = [N x K] 
                    N is number of training samples, 
                    K is number of classes
                -shadow_in_pred(torch.Tensor): prediction vector of input dataset
                    shape = [N x K]
                -shadow_out_true_label(torch.Tensor): true label of non-input dataset
                in one-hot encoding format
                    shape = [M x K]
                    M is number of test samples
                    K is number of classes
                -shadow_out_pred(torch.Tensor): prediction of non-input dataset
                    shape = [M x K]

        """

        shadow_in_true_label = []
        shadow_in_prediction = []
        shadow_out_true_label = []
        shadow_out_prediction = []

        # Get the labeled data ("in" , "out") for the attacker
        for model_idx in range(self.numOfModels):
            model = self.list_of_models[model_idx]

            trainloader = self.trainloaders[model_idx]
            testloader = self.testloaders[model_idx]

            in_true_label , in_pred_label = gather_inference(model, trainloader)
            out_true_label , out_pred_label = gather_inference(model, testloader)

            shadow_in_true_label.append(in_true_label)
            shadow_in_prediction.append(in_pred_label)
            shadow_out_true_label.append(out_true_label)
            shadow_out_prediction.append(out_pred_label)
        
        shadow_in_true_label = torch.cat(shadow_in_true_label)
        shadow_in_prediction = torch.cat(shadow_in_prediction)
        shadow_out_true_label = torch.cat(shadow_out_true_label)
        shadow_out_prediction = torch.cat(shadow_out_prediction)

        return (shadow_in_true_label , shadow_in_prediction, shadow_out_true_label , shadow_out_prediction)

    
    def fitTransform(self, X , y, train_size, numIter):
        """Train shadow model on a custom dataset, then generate training dataset
          into different classes for attacker model

            Args:
                X (torch.Tensor): Shadow model training input
                y (NxK torch.Tensor): Shadow model training true label
                numIter : number of Epochs for every shadow models
            Returns:
                result(dict) : [keys : values]

                keys(int) : the true classes' values (from 1 -> K)

                values(tuple) : (pred, membership_status)
                    pred(torch.Tensor) : prediction vector of input/non-input dataset
                        shape = [N x K]
                    membership_status(torch.Tensor) : membership of the input to shadow model
                        shape = [N x K] with value 0/1

        """

        self._fit(X , y, train_size, numIter)

        (shadow_in_true_label,
         shadow_in_prediction,
         shadow_out_true_label,
         shadow_out_prediction) = self._transform()

        prediction = {}
        attack_data = {}
        unique_labels = torch.unique(shadow_in_true_label)
        for c_label in unique_labels:
            in_indices = (shadow_in_true_label == c_label)
            in_pred = shadow_in_prediction[in_indices]
            out_indices = (shadow_out_true_label == c_label)
            out_pred = shadow_out_prediction[out_indices]

            prediction[c_label.item()] = (in_pred, out_pred) 
        
        for key, (In, Out) in prediction.items():
            attack_train_X = torch.cat((In, Out))
            attack_train_y = torch.cat(
                (torch.ones(In.shape[0]),
                torch.zeros(Out.shape[0]))
            )
            attack_data[key] = (attack_train_X, attack_train_y)
        
        return attack_data



class AttackModel:
    """
       The attacker class used to attack the target model
    """
    def __init__(self, base_attacker, num_classes):
        """Constructing attack models

            Args: 
                baseModel (torch.nn.Module) : The base model that all attack 
                                              models inherit from
            Attributes:
                baseModel (torch.nn) : The baseline architecture of an attacker
                list_of_models (list) : the list of different attacker models used on different 
                                      classes                 

        """
        self.list_of_models = []
        self.base_model = base_attacker
        self.num_classes = num_classes

        for _ in range(num_classes):
            cur_model = deepcopy(self.base_model)
            self.list_of_models.append(cur_model)

    def fit(self, attack_train_data, num_classes, epochs):
        """Training attack models on shadow model's labeled output
            Args: 
                train_data (torch.Datasets) : The labeled data (0,1) used for 
                                              training attacker models
                number_of_classes (int) : The number of attacker model used, each for
                                          each different class
                numIter (int) : Number of iteration for every model
            Returns:
                None
        """
        # Train each attack model seperately
        for cur_class in range(num_classes):

            cur_model = self.list_of_models[cur_class]

            if cur_class in attack_train_data.keys():

                # 1. Load data from training set
                train_X, train_y = attack_train_data[cur_class]
                train_datasets = CustomDataset(train_X, train_y)
                train_dataloader = DataLoader(train_datasets, 
                                               batch_size = 4,
                                               shuffle = True)
                # 2. Preparing loss function and optimizer
                loss_fn = nn.CrossEntropyLoss()
                metric_score_fn = BinaryAccuracy()
                optimizer = torch.optim.SGD(cur_model.parameters(), lr = 0.001)

                # 3. Train current model on shadow model's membership data
                for _ in range(epochs):
                    cur_model, train_loss, train_acc= train_step(cur_model, train_dataloader, loss_fn, metric_score_fn, optimizer)

            self.list_of_models.append(cur_model)
    def predict_membership_prob(self, target_pred, true_class):
        """Predicting membership probability of a single sample
            Args:
                target_pred (torch.Tensor): [K,] dimension probability vector,
                                             result by passing sample to target 
                                             model 
                true_class (float): the true output/label of the sample
            
            Returns:
                membership_prob (float): probability that this sample is a 
                                      training sample
        """
        # The logits of the attacker model prediction
        membership_logits = self.list_of_models[true_class](target_pred)
        # The probability of the attacker model prediction
        membership_prob = nn.Sigmoid()(membership_logits)
        return membership_prob
    
    def predict_membership_status(self, prediction, true_class):
        """Predicting membership status of a single sample
            Args:
                target_pred (torch.Tensor) : [K,] dimension probability vector,
                                             result by passing sample to target model
                true_class (float) : the true output/label of the sample
            
            Returns:
                0 : This sample is non-member sample
                1 : This sample is member sample
        """
        membership_prob = self.predict_membership_prob(prediction, true_class)
        return torch.round(membership_prob)
   
    def eval(self, attack_data):
        """Evaluating the attacker's accuracy and recall
            Args: 
                attack_data (dict): (keys, values)
                    -keys (int): The id of the class
                    -values (tuple): (Target/Shadow Prediction, True Membership) 
            Returns:
                acc (float): The rate of correctly classified member samples
                recall (float): The rate of 
                
        """
        membership_pred = []
        acc , recall = 0 , 0

        # 0. Loop through each class
        for cur_class, (prediction, membership_true_cur) in attack_data.items():

            # 1. Finding the membership prediction on attack_data
            membership_pred_cur = self.predict_membership_status(prediction, cur_class)

            # 2. Calculating sub accuracy and sub recall on one class
            train_pred_cur = (membership_pred_cur == 1)
            train_true_cur = (membership_true_cur == 1)

            acc_cur = sum(train_pred_cur == train_true_cur) / sum(train_pred_cur)
            recall_cur = sum(train_pred_cur == train_true_cur) / sum(train_true_cur)
            acc += acc_cur
            recall += recall_cur 
        
        acc , recall = acc / self.num_classes , recall / self.num_classes