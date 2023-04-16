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



class TargetModel:
    def __init__(self, target_architecture, **kwargs):
        """
        The target model wrapper for base target model
        Args:
            base_target (nn.Module): The target model architecture to be attacked
        Attributes: 
            tg (nn.Module): The target model to be attacked
            trainloader (DataLoader): The dataloader for training dataset
            testloader (TestLoader):The testdataloader for test dataset
        """
        self.tg = target_architecture(**kwargs).to("cuda")   # initialize the target model
        self.trainloader = None
        self.testloader = None
    def set_dataloader(self, X, y, partition = 0.5):
        """
        Setting dataloader for target model 
        Args: 
            X: The input data for model as a float32 tensor of shape [n_samples, (input shapes)]
            y: The label data for model as a float32 tensor of shape [n_samples, (label shapes)]
            partition (float32, optional): The ratio of training samples versus whole dataset
        """
        train_size = int(len(X) * partition)

        X_train, y_train = X[0:train_size] , y[0:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        trainset = CustomDataset(X_train, y_train)
        testset = CustomDataset(X_test, y_test)

        self.trainloader = DataLoader(trainset, batch_size = 100)
        self.testloader = DataLoader(testset, batch_size = 100)
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
        self.set_dataloader(X, y, partition)
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.tg.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        #training data in epochs
        for _ in tqdm(range(epochs)):
            self.tg , train_loss, train_acc = train_step(self.tg, self.trainloader, loss_fn,
                                                         optimizer)
            test_loss, test_acc = test_step(self.tg, self.testloader, loss_fn)

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
            y (torch.Tensor): Label data with shape [n_samples, n_classes]
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
        unique_labels = torch.arange(y.shape[1])

        prediction = {}
        attack_data = {}

        for c_label in unique_labels:
            in_indices = (in_true_label == c_label)
            in_pred = in_pred_label[in_indices]
            out_indices = (out_true_label == c_label)
            out_pred = out_pred_label[out_indices]

            prediction[c_label.item()] = (in_pred, out_pred) 
        
        for key, (In, Out) in prediction.items():
            attack_test_X = torch.cat((In, Out))
            attack_test_y = torch.cat(
                (torch.ones(In.shape[0]),
                torch.zeros(Out.shape[0]))
            )
            attack_data[key] = (attack_test_X, attack_test_y)
        
        return attack_data 
    def estimate_sharpness(self, current_l, lr, n_points = 5):
        """Estimating the loss function sharpness at a certain point of weight space
        Args:
            current_l (torch.float32): The current loss value of the model
            lr (torch.float32): The learning rate of the model
            n_points (int): The number of points that the model should be trained further to estimate sharpness
        Returns:
            sharpness (torch.float32): The sharpness of the loss function at the current point of weight space
        """
        
        #1. Getting current weight and loss value
        current_w = self.tg.parameters()
        current_l = current_l
        loss_fn = nn.CrossEntropyLoss()
        
        #2. Getting the total length of the dataset
        n_samples = len(self.trainloader)
        sharpness = 0
        #2. Training further to get another weight and loss value
        for _ in tqdm(range(n_points)):
            
            with torch.no_grad():

                #2.1 First, copy the model to the second one
                model_clone = self.tg
                optimizer = torch.optim.SGD(model_clone.parameters(), lr=lr)

                #2.2 Picking a random index of the sample
                index = np.random.randint(low=0, high=n_samples, size=1)
            
            #2.3 Training model further on this sample
            model_clone = train_step(model_clone, self.trainloader, loss_fn,
                       optimizer, index=index)
            
            #2.4 Sharpness
            #### The forumula is sharpness = increases_in_loss / (1 + increases_in_weight) 
            with torch.no_grad():
                
                #2.4.1 Getting variables for sharpness calculation
                next_w = model_clone.parameters()
                next_l, next_acc = test_step(model_clone, self.trainloader, loss_fn)

                #2.4.2 Calculating sharpness
                dist_w , dist_l = 1 , next_l - current_l
                for (c_w, n_w) in zip(current_w, next_w):
                    dist_w += torch.linalg.norm(c_w - n_w)

                sharpness += dist_l / dist_w

        # Average sharpness
        sharpness /= n_points
        torch.cuda.empty_cache()  # Emptying the cache to prevent newly built model utilizing GPU memory
        print(f"Estimated sharpness {sharpness}")
        return sharpness
             


class ShadowModel:
    """Shadow models architecture implementation based on 
       the paper https://arxiv.org/abs/1610.05820
    """
    def __init__(self, target_architecture, n_models = 4, randomSeed = 54, **kwargs):
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
        for i in range(n_models):
            c_model = target_architecture(**kwargs).to("cuda")
            self.list_of_models.append(c_model)

        self.numOfModels   = n_models

        self.trainloaders  = []
        self.testloaders = []
        self.train_losses = None
        self.test_losses = None
        
        self.seed = randomSeed
        self.rdSt  = np.random.RandomState(self.seed)

    def set_dataloader(self, X , y, train_size):
        """Setting the dataloader for each shadow model
        Args:
            X (torch.tensor): The input data
            y (torch.tensor): The target data
            train_size (int): The size of the training set
        Returns:
            None
        """
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
              train_size (int): The sample size of each shadow model
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
            
            optimizer = torch.optim.SGD(model.parameters() , lr = 0.001)
            trainloader_idx = self.trainloaders[model_idx]
            testloader_idx = self.testloaders[model_idx]
            
            train_losses = []
            test_losses = []
            for _ in tqdm(range(epochs)):
                model, train_loss, train_acc = train_step(model, trainloader_idx,loss_fn,
                                                          optimizer)
                test_loss, test_acc = test_step(model, testloader_idx,loss_fn)

                train_losses.append(train_loss)
                test_losses.append(test_loss)
                
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

    
    def fit_transform(self, X , y, train_size, numIter):
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
    def __init__(self, attacker_architecture, num_classes=10, **kwargs):
        """Constructing attack models

        Args: 
            attacker_architecture (torch.nn.Module): The base model that all attack 
                                             models inherit from
            num_classes (int): The number of classes in the dataset
        Attributes:
            list_of_models (list) : the list of different attacker models used on different 
                                      classes
            num_classes (int) : the number of classes in the dataset
            train_dataloaders (list) : the list of dataloaders for each attacker model     

        """
        self.list_of_models = []
        self.num_classes = num_classes
        self.train_dataloaders = []

        for _ in range(self.num_classes):
            cur_model = attacker_architecture(**kwargs)
            self.list_of_models.append(cur_model)
    def _set_dataloader(self, attack_train_data):
        """
        Setting dataloader for member attack model 
        Args: 
            attack_train_data (dict): A dictionary of (true_label, (shadow_prediction, membership_status))
        """
        # Setting dataloader for each model seperately
        for cur_class in range(self.num_classes):

            #1. Load data from training set
            train_X, train_y = attack_train_data[cur_class]
            train_datasets = CustomDataset(train_X, train_y)
            train_dataloader = DataLoader(train_datasets,
                                            batch_size = 4,
                                            shuffle = True)
            self.train_dataloaders.append(train_dataloader)

    def fit(self, attack_train_data, epochs):

        """Training attack models on shadow model's labeled output
            Args: 
                attack_train_data (dict) : The labeled data (0,1) used for 
                                              training attacker models
                epochs (int): Number of epochs for each attacker model
            Returns:
                None
        """
        # Setting dataloader for each model 
        self._set_dataloader(attack_train_data)

        # Specifying global loss function
        loss_fn = nn.CrossEntropyLoss()

        # Train each attack model seperately
        for cur_class in range(self.num_classes):

            # 2. Preparing model and optimizer
            cur_model = self.list_of_models[cur_class]
            optimizer = torch.optim.SGD(cur_model.parameters(), lr = 0.001)
            train_dataloader = self.train_dataloaders[cur_class]

            # 3. Train current model on shadow model's membership data
            
            for _ in tqdm(range(epochs)):
                cur_model, train_loss, train_acc= train_step(cur_model, train_dataloader, loss_fn, optimizer)

                print(f"Training attack model {cur_class} | Train loss {train_loss} | Train Accuracy {train_acc}")

            self.list_of_models[cur_class] = cur_model

    def predict_membership_prob(self, target_pred, true_class):
        """Predicting membership probability of many samples
            Args:
                target_pred (torch.Tensor): [N, K], N K-dimensional probability vector,
                                             result by passing sample to target 
                                             model 
                true_class (float): the true output/label of the sample
            
            Returns:
                membership_prob (float): probability that this sample is a 
                                      training sample
        """
        # The attacker model membership probability prediction
        membership_prob = self.list_of_models[true_class](target_pred)
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
   
    def eval(self, attack_test_data):
        """Evaluating the attacker's accuracy and recall
            Args: 
                attack_test_data (dict): (keys, values)
                    -keys (int): The id of the class
                    -values (tuple): (Target/Shadow Prediction, True Membership) 
            Returns:
                acc (float): The percentage of correctly predicted samples
        """
        ACC = 0
        # 0. Loop through each class
        for cur_class, (prediction, membership_true) in attack_test_data.items():

            #1. Finding the membership prediction on attack_data with 0/1 value
            membership_pred = self.predict_membership_status(prediction, cur_class).cpu().squeeze(dim = 1)
            total = len(membership_true)

            #2. Calculating accuracy of the attacker
            ACC += ((membership_pred == membership_true).sum()) / total

        ACC    /= self.num_classes
        return ACC