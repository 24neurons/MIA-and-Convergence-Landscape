import os 
import sys
import random

import numpy as np
import argparse
from models import * 
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(FILE_DIR , "data/")
sys.path.append(DATA_ROOT)

from ...utils import *
################################################################################
# save and read argument parsers
################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, help="Name of the dataset", default='CIFAR100',
                        choices=['CIFAR100', 'CIFAR10', 'FashionMNIST'])
    parser.add_argument('--target_model', type=str, help="Name of the target model architecture", 
                        default='resnet20', choices=['resnet20', 'resnet50', 'vgg19'])
    parser.add_argument('--target_size', type=int, help="Number of training samples for target model")
    parser.add_argument('--target_lr', type=int, help="Learning rate of target model")
    parser.add_argument('--target_epoch', type=int, help="Epochs of the target model", default=50)
    parser.add_argument('--attack_model', type=str, help="Name of the base attack model architecture",
                        default='NN', choices=['NN'])
    args = parser.parse_args()
    
    return args

################################################################################
# attacker class with target model, shadow model and a base attacker
################################################################################
class Attacker():
    def __init__(self, args):
        self.args = args 
        self.data_root = os.path.join(DATA_ROOT, self.args.dataset)
        
        if self.args.dataset == "CIFAR100":
            self.num_classes = 100
        elif self.args.dataset == "CIFAR10":
            self.num_classes = 10

        target_model = globals()[self.args.target_model](num_classes = self.num_classes)
        attack_model = globals()[self.args.attack_model](num_classes = self.num_classes)

        self.tm = TargetModel(target_model)
        self.sm = ShadowModel(target_model, num_shadow_models = 5)
        self.at = BaseAttacker(attack_model, num_classes = self.num_classes)
        self._set_data()

    def _set_data(self):
        if self.args.dataset == "CIFAR100":
            self.dataset = datasets.CIFAR100
            
        elif self.args.dataset == "CIFAR10":
            self.dataset = datasets.CIFAR10
        
        label_transform = lambda y : torch.zeros(self.num_classes, dtype = torch.float).scatter_(dim = 0, index = torch.tensor(y) , value = 1)

        trainset = self.dataset(root=self.data_root, train=True, transform=ToTensor(), download=True,
                                       target_transform=label_transform)
        testset = self.dataset(root=self.data_root, train=False, transform=ToTensor(), download=True,
                                       target_transform=label_transform)
        fullset = trainset + testset
        
        target_indicies = random.choices(np.arange(self.train_size), k= 2*self.args.target_size)

        shadow_indicies = set(range(self.dataset_size)) - set(target_indicies)
        
        fullloader = DataLoader(fullset, batch_size = 4, shuffle=True)

        for (X, y) in fullset:
            fullset_X = X
            fullset_y = y
        
        self.target_X, self.target_y = fullset_X[target_indicies], fullset_y[target_indicies]
        self.shadow_X, self.shadow_y = fullset_X[shadow_indicies], fullset_X[shadow_indicies]

    
    def run_attacks(self):

        # 1. Train shadow models and generate data for attacker
        attack_train = self.sm.fitTransform(self.shadow_X, self.shadow_y, self.args.target_epoch, self.args.target_lr)

        # 2. Fitting the attacker
        self.at.fit(attack_train, self.num_classes, 40) 

        # 3. Testing attacker accuracy on different epochs of target model 

        for _ in tqdm(range(self.args.target_epochs)):

            # 3.1 Train target model and getting attack test data 
            attack_test = self.tg.fitTransform(self.target_X, self.target_y, 1, self.args.target_lr)

            # 3.2 Predict target membership 
            target_membership = self.at.get_membership(attack_test)
            test_acc, test_recall = self.at.eval(attack_test)

            print(f"Attacking on epoch {_} of target model | Acc: {test_acc } | Recall: {test_recall}")
        
        

        
        

################################################################################
# main driver of attack
################################################################################

def main():
    args = parse_arguments()
    attacker = Attacker(args)
    attacker.run_attacks()

if __name__ == '__main__':
    main()

    