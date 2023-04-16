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
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(FILE_DIR , "data/")
UTILS_ROOT = os.path.dirname(os.path.dirname(FILE_DIR))
sys.path.append(DATA_ROOT)
sys.path.append(UTILS_ROOT)


from utils import NNTwoLayers, TargetModel, ShadowModel, AttackModel, train_step, test_step
###############################################################################
###############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, help="Name of the dataset", default='CIFAR100',
                        choices=['CIFAR100', 'CIFAR10', 'FashionMNIST'])                         
    parser.add_argument('--target_model', type=str, help="Name of the target model architecture", 
                        default='resnet20', choices=['resnet20', 'resnet50',  'vgg11', 'vgg11_bn', 'vgg13', 
                                                     'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',])
    parser.add_argument('--target_size', type=int, help="Number of training samples for target model")
    parser.add_argument('--target_lr', type=float, help="Learning rate of target model")
    parser.add_argument('--target_epoch', type=int, help="Epochs of the target model", default=100)
    args = parser.parse_args()
    
    return args

################################################################################
# attacker class with target model, shadow model and a base attacker
################################################################################
class FullAttacker():
    def __init__(self, args):
        self.args = args 
        self.data_root = os.path.join(DATA_ROOT, self.args.dataset)
        
        if self.args.dataset == "CIFAR100":
            self.num_classes = 100
        elif self.args.dataset == "CIFAR10":
            self.num_classes = 10

        target_model_archiecture = globals()[self.args.target_model]
        attack_model_architecture = globals()[self.args.attack_model]

        self.tm = TargetModel(target_model_archiecture, num_classes = self.num_classes)
        self.sm = ShadowModel(target_model_archiecture, num_classes = self.num_classes)
        self.at = AttackModel(attack_model_architecture, num_classes = self.num_classes)
        self._data_preprocess()
        self._divide_data()

    def _data_preprocess(self):
        if self.args.dataset == "CIFAR100":
            self.dataset = datasets.CIFAR100
            
        elif self.args.dataset == "CIFAR10":
            self.dataset = datasets.CIFAR10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.train_set = self.dataset(self.data_root, train=True, download=True, transform=transform_train)
        self.test_set = self.dataset(self.data_root, train=False, download=True, transform=transform_test)

        # Taking the images tensor
        X_train, X_test = torch.tensor(self.train_set.data, dtype=torch.float32).permute(0,3,1,2)/255 , torch.tensor(self.test_set.data,dtype=torch.float32).permute(0,3,1,2)/255

        # Taking the target in tensor of (N,) shape
        y_train_label, y_test_label =  torch.tensor(self.train_dataset.targets, dtype=torch.int64), torch.tensor(self.test_set.targets, dtype=torch.int64)

        # Converting label into one-hot encoding 
        y_train, y_test = torch.nn.functional.one_hot(y_train_label, 10).to(torch.float32), torch.nn.functional.one_hot(y_test_label,10).to(torch.float32)

        #Concat them into one dataset
        self.X = torch.cat((X_train , X_test))
        self.y = torch.cat((y_train, y_test))
    def _divide_data(self):

        target_size = self.args.target_size

        # Divide the data into target and shadow
        target_indices = np.random.choice(range(len(self.X)), target_size, replace=False)
        shadow_indices = np.array(list(set(range(len(self.X))) - set(target_indices)))

        self.target_X, self.target_y = self.X[target_indices], self.y[target_indices]
        self.shadow_X, self.shadow_y = self.X[shadow_indices], self.y[shadow_indices]
    
    def run_attacks(self):

        tg_train_losses, tg_test_losses = [], []
        
        # 1. Train shadow models and generate data for attacker
        shadow_attack_data = self.sm.fit_transform(self.shadow_X, self.shadow_y, self.args.target_epoch)
        # 2. Train attacker model based on shadow models labeled data, default 50 epochs
        self.at.fit(shadow_attack_data, 50)
        # 3. Evaluate the attacker model on target model per each epoch

        for _ in range(self.args.target_epoch):
            # 3.1 Train target model and generate data for attacker
            target_attack_data = self.tm.fit_transform(self.target_X, self.target_y, 1, self.args.target_lr)
            # 3.2 Evaluating target model accuracy 
            loss_fn = nn.CrossEntropyLoss()

            with torch.no_grad():
                tg_train_loss, tg_train_acc = test_step(self.tm.tg, self.tm.trainloader, loss_fn)
                tg_test_loss, tg_test_acc = test_step(self.tm.tg, self.tm.testloader, loss_fn)
                print(f"Target model train loss: {tg_train_loss:.4f} | train acc: {tg_train_acc:.4f} | test loss: {tg_test_loss:.4f} | test acc: {tg_test_acc:.4f}")
                tg_train_losses.append(tg_train_loss)
                tg_test_losses.append(tg_test_loss)

            # 3.3 Estimating model sharpness on current weights
            sharpness = self.tg.estimate_sharpness()
            print(f"Sharpness: {sharpness:.4f}")

            # 3.4 Evaluating the attacker model on target model 
            attack_acc = self.at.eval(target_attack_data)
            print(f"With learning rate {self.args.target_lr}, attacking on epoch {_}, with accuracy {attack_acc:.4f}")
            
            # 3.5 If the model is reaching plateau, then stop training
                
            if _ > 20 : # if trained for more than 20 epochs
                max_recent_loss = max(tg_train_losses[-20:])
                min_recent_loss = min(tg_test_losses[-20:])
                if(max_recent_loss - min_recent_loss < 1e-2):
                    # Stop training in this learning rate
                    print(f"Stop training target model at epoch {_}")
                    break
        
            
        

###################################################################
# main driver of attack
###################################################################

def main():
    args = parse_arguments()
    attacker = FullAttacker(args)
    attacker.run_attacks()

if __name__ == '__main__':
    main()
    