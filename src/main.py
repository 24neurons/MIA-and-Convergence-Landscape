import os 
import sys
import random


import numpy as np
import argparse
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from models import attack_models, resnet, resnet18, vgg, vgg11, MLP, NNTwoLayers
from utils import train_step, test_step, accuracy_fn


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(FILE_DIR , "data/")
UTILS_ROOT = FILE_DIR
sys.path.append(DATA_ROOT)
sys.path.append(UTILS_ROOT)


from utils import  TargetModel, ShadowModel, AttackModel
###############################################################################
###############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, help="Name of the dataset", default='CIFAR10',
                        choices=['CIFAR100', 'CIFAR10', 'FashionMNIST'])                         
    parser.add_argument('--target_model', type=str, help="Target model architecture's name", 
                        default='resnet18', choices=['resnet20', 'resnet50',  'vgg11', 'vgg11_bn', 'vgg13', 
                                                     'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',])
    parser.add_argument('--target_size', type=int, default=6000, help="Number of training samples for target model")
    parser.add_argument('--target_lr', type=float, default=0.01, help="Learning rate of target model")
    parser.add_argument('--target_epoch', type=int, help="Epochs of the target model", default=150)
    parser.add_argument('--attack_arc', type = str, help="Attacker model architecture's name",
                        default="NNTwoLayers")
    args = parser.parse_args()
    
    return args

################################################################################
# attacker class with target model, shadow model and a base attacker
################################################################################
class FullAttacker():
    def __init__(self, args):
        self.args = args 
        self.data_root = os.path.join(DATA_ROOT, self.args.dataset)
        self.in_channels = 3
        
        if self.args.dataset == "CIFAR100":
            self.num_classes = 100
        elif self.args.dataset == "CIFAR10":
            self.num_classes = 10
        elif self.args.dataset == "FashionMNIST":
            self.num_classes = 10
            self.in_channels = 1

        # Query the name of target model by the argument string
        target_model_archiecture = globals()[self.args.target_model]
        attack_model_architecture = globals()[self.args.attack_arc]

        self.tm = TargetModel(target_model_archiecture, in_channels = self.in_channels,
                              num_classes = self.num_classes)
        
        self.sm = ShadowModel(target_model_archiecture, in_channels = self.in_channels,
                              num_classes = self.num_classes)
        
        self.at = AttackModel(attack_model_architecture, num_classes = self.num_classes)
        self._data_preprocess()
        self._divide_data()

    def _data_preprocess(self):
        if self.args.dataset == "CIFAR100":
            self.dataset = datasets.CIFAR100
            
        elif self.args.dataset == "CIFAR10":
            self.dataset = datasets.CIFAR10
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32)),
        ])

        self.train_set = self.dataset(self.data_root, train=True, download=True, transform=transform_train,
                                      target_transform = lambda y : torch.zeros(self.num_classes, dtype = torch.float).scatter_(dim = 0, index = torch.tensor(y) , value = 1))
        self.test_set = self.dataset(self.data_root, train=False, download=True, transform=transform_test,
                                      target_transform = lambda y : torch.zeros(self.num_classes, dtype = torch.float).scatter_(dim = 0, index = torch.tensor(y) , value = 1))

        # Taking the images tensor
        train_loader = DataLoader(self.train_set, batch_size = len(self.train_set))
        test_loader = DataLoader(self.test_set , batch_size = len(self.test_set))

        X_train, y_train = next(iter(train_loader))[0], next(iter(train_loader))[1]
        X_test, y_test = next(iter(test_loader))[0], next(iter(test_loader))[1]

        #Concat them into one dataset
        self.X = torch.cat((X_train , X_test))
        self.y = torch.cat((y_train, y_test))
    def _divide_data(self):

        target_size = self.args.target_size

        # Divide the data into target and shadow
        rng = np.random.RandomState(1)
        target_indices = rng.choice(np.arange(len(self.X)), size=2*target_size)
        shadow_indices = list(set(np.arange(len(self.X))) - set(target_indices))

        self.target_X, self.target_y = self.X[target_indices], self.y[target_indices]
        self.shadow_X, self.shadow_y = self.X[shadow_indices], self.y[shadow_indices]
    
    def run_attacks(self):

        tg_train_losses, tg_test_losses = [], []
        tg_train_accs, tg_test_accs = [], []
        print("Training shadow models")
        # 1. Train shadow models and generate data for attacker
        shadow_attack_data = self.sm.fit_transform(self.shadow_X, self.shadow_y, self.args.target_size,self.args.target_epoch)
        # 2. Train attacker model based on shadow models labeled data, default 50 epochs
        print("Training attack model")
        self.at.fit(shadow_attack_data, 50)
        # 3. Evaluate the attacker model on target model per each epoch
        print("Training target model")

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
                tg_train_accs.append(tg_train_acc)
                tg_test_accs.append(tg_test_acc)

            # 3.3 Estimating model sharpness on current weights
            sharpness = self.tm.sharpness
            print(f"Sharpness: {sharpness:.4f}")

            # 3.4 Evaluating the attacker model on target model 
            attack_acc = self.at.eval_attack(target_attack_data)
            print(f"With learning rate {self.args.target_lr}, attacking on epoch {_}, with accuracy {attack_acc:.4f}")
            
            # 3.5 If the model is reaching plateau, then stop training
                
            if _ > 30 : # if trained for more than 20 epochs
                train_std = np.std(tg_train_accs[-30:])
                if train_std < 0.01:
                    print("Target model is reaching plateau, stop training")
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
    