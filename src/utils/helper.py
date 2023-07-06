"""
This module is mainly supporting tools for all the types of model,
including training/testing step, custom dataset, etc.
"""
import torch
from torch.utils.data import Dataset
from torch import nn
from copy import deepcopy
from math import sqrt 

class CustomDataset(Dataset):
    """
    A custom torch Dataset that allows iteration through two rela=ted tensors: input and label
    """
    def __init__(self, X , y):
        """
        This allows initializing a custom pair dataset (X,y) 
        Args:
            X (torch.Tensor): The first tensor to be queried, namingly "input"
            y (torch.Tensor): The second tensor to be queried, namingly "label"
        """
        self.X = X
        self.y = y
    def __len__(self):
        """
        Returing the length of input tensor
        """
        return len(self.X)
    def __getitem__(self, idx):
        """
        Returning the element in index ith of the two inputed tensor
        Args:
            idx (int): The index to be queried on both tensor
        Returns:
            X[idx], y[idx]: The idx element in the two tensor (X,y) respectively
        """
        return (self.X[idx] , self.y[idx])


def train_step(model, trainloader, loss_fn, optimizer, epsilon=0.01, sharpness=False):
    """
    This function trains a model in one epoch
    Args:
        model : The model that is going to be trained
        trainloader (DataLoader): The training set DataLoader for the target model
        loss_fn : The objective function that the model is configured to minimize
        optimizer: The gradient stepping optimizer for the model
    Returns:
        If sharpness = False:
            model (nn.Module): The resuling model after one epoch of training
            train_loss (torch.float32): The training loss of this model over the entire 
                                        training dataset on this epoch
            train_acc (torch.float32):The training score over the entire training dataset
        If sharpness = True:
            model (nn.Module): The resulting model after training on one sample
            train_loss (torch.float32): The training loss of this model over the entire
                                        training dataset on this epoch
            train_acc (torch.float32):The training score over the entire training dataset
            sharpness (torch.float32): The sharpness of the minima the model is converging to
    """
    model.train()
    train_loss, train_acc = 0,0
    global_increase = 0
    for batch_idx, (X_b, y_b) in enumerate(trainloader):
        X_b, y_b = X_b.to("cuda"), y_b.to("cuda")
        optimizer.zero_grad()
        pred = model(X_b).squeeze(dim = 1) 

#             softmax_pred = nn.Softmax(dim=1)(pred)

#             print(f"Min pred {torch.log(torch.min(softmax_pred.cpu().detach()))}")

        batch_loss = loss_fn(pred, y_b)
        train_loss += batch_loss.item()
        batch_loss.backward()

        batch_acc = accuracy_fn(pred, y_b)

        train_acc += batch_acc.item()
        if sharpness:
            with torch.no_grad():
                model_clone = deepcopy(model).to("cuda")
                grad_norm = 0

                # Calculate the gradient norm
                for param in model.parameters():
                    x = param.grad
                    grad_norm += torch.sum(x**2)
                # Moving the parameters 0.01 towards the gradient direction
                grad_norm = sqrt(grad_norm)
                for ((new, new_param), (old, old_param)) in zip(model_clone.named_parameters(), model.named_parameters()):
                        new_param.data = old_param.data + epsilon * old_param.grad / grad_norm
                
                next_pred = model_clone(X_b).squeeze(dim = 1)
                next_loss = loss_fn(next_pred, y_b)

                # Calculating the increase in loss
                local_increase = next_loss - batch_loss
                global_increase += local_increase.item()
            
        optimizer.step()
    
    # Average the quantity over the entire dataset
    global_increase /= len(trainloader)
    train_loss /= len(trainloader)
    train_acc /= len(trainloader)
    
    if sharpness:
        return model, train_loss, train_acc, global_increase
    return model, train_loss, train_acc
        
#             pbar.set_description(desc=f'Loss={batch_loss} Batch_id={batch_idx} Accuracy={100*correct/total:0.2f}')
def test_step(model, testloader, loss_fn):
    """
    This function calculate the test error of a model
    Args:
        model (nn.Module): The model that is going to be tested
        testloader (nn.Module): The test dataloader of the model
        loss_fn: The objective function that should be minimized by the model
    Returns:
        test_loss (torch.float32): The test loss of the model on the test dataset
        test_acc (torch.float32): The test accuracy of the model on the test dataset
    """
    test_loss, test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for (X,y) in testloader:
            X, y = X.to("cuda"), y.to("cuda")
            pred = model(X).squeeze(dim = 1)
            
            test_loss += loss_fn(pred, y).item()
            test_acc += accuracy_fn(pred,y).item()
            
        test_loss /= len(testloader)
        test_acc /= len(testloader)
    return test_loss, test_acc

def gather_inference(model, dataloader):
    """
    The function that gather the inference of the model on a dataset
    Args:
        model (nn.Module): The model used to predict output
        dataloader: The dataloader of the dataset, with both (input, label) dimensions
    Returns:
        pred_labels (torch.Tensor): The prediction of model on provided dataset as a 
                                    [n_samples, n_classes] float32 tensor
        input_labels (torch.Tensor): The true label of provided dataset as a 
                                     [n_samples, ] int tensor
        
    """

    # First, these two output are lists to append samples
    pred_labels = []
    input_labels = []
    
    with torch.inference_mode():
        for data in dataloader:
            input, label = data
            input, label = input.to("cuda"), label.to("cuda")
            label = torch.nonzero(label)[:, 1]

            pred  = model(input)
            pred_labels.append(pred)
            input_labels.append(label)
    
    # Casting lists of output into torch.Tensor
    pred_labels = torch.cat(pred_labels, dim = 0)
    input_labels = torch.cat(input_labels, dim = 0)

    return (input_labels, pred_labels)

def accuracy_fn(pred, y):
    # if y has more than two labels
    if len(y.shape) > 1:
        pred_label = pred.argmax(dim = 1)
        y_label = y.argmax(dim = 1)
    else:
        pred_label = torch.round(nn.Sigmoid()(pred))
        y_label = y
    return ((pred_label == y_label).sum()/len(y_label)).float()