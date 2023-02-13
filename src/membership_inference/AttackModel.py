import torch
from torch import nn
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from .utils import CustomDataset


class AttackModel:
    """
       The attacker class used to attack the target model
    """
    def __init__(self, baseModel):
        """Constructing attack models

            Args: 
                baseModel (torch.nn.Module) : The base model that all attack 
                                              models inherit from
            Attributes:
                baseModel (torch.nn) : The baseline architecture of an attacker
                listOfModels (list) : the list of different attacker models used on different 
                                      classes                 

        """
        self.listOfModels = []
        self.baseModel = baseModel
    def _fit(self, attack_train_data, number_of_classes, numIter):
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
        for cur_class in range(number_of_classes):

            cur_model = self.baseModel

            if cur_class in attack_train_data.keys():

                train_X, train_y = attack_train_data[cur_class]
                train_datasets = CustomDataset(train_X, train_y)
                train_dataloader = DataLoader(train_datasets, 
                                               batch_size = 4,
                                               shuffle = True)
                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(cur_model.parameters(), lr = 0.001)

                cur_model = trainModel(cur_model, train_dataloader, 
                                           numIter, loss_fn, optimizer)


            self.listOfModels.append(cur_model)
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
        membership_logits = self.listOfModels[true_class](target_pred)
        # The probability of the attacker model prediction
        membership_prob = nn.Sigmoid()(membership_logits)
        return membership_prob
    
    def predict_membership_status(self, target_pred, true_class):
        """Predicting membership status of a single sample
            Args:
                target_pred (torch.Tensor) : [K,] dimension probability vector,
                                             result by passing sample to target model
                true_class (float) : the true output/label of the sample
            
            Returns:
                0 : This sample is non-member sample
                1 : This sample is member sample
        """
        membership_prob = self.predict_membership_prob(target_pred, true_class)
        return torch.round(membership_prob)
        
                 