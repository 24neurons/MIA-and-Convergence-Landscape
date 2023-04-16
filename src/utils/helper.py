"""
This module is mainly supporting tools for all the types of model,
including training/testing step, custom dataset, etc.
"""
import torch
from torch.utils.data import Dataset

"""
This module is mainly supporting tools for all the types of model,
including training/testing step, custom dataset, etc.
"""
import torch
from torch.utils.data import Dataset

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


def train_step(model, trainloader, loss_fn, optimizer, index=None):
    """
    This function trains a model in one epoch
    Args:
        model (nn.Module): The model that is going to be trained
        trainloader (DataLoader): The training set DataLoader for the target model
        loss_fn : The objective function that the model is configured to minimize
        optimnizer: The gradient stepping optimizer for the model
        metric_score_fn: The metric that is used to measure the accuracy, can be F1_score, F2...
        index: 
            None: The model shall be trained on the whole training data
            int: The model shall be trained on the index-th single batch
    Returns:
        If index is None:
            model (nn.Module): The resuling model after one epoch of training
            train_loss (torch.float32): The training loss of this model over the entire 
                                        training dataset on this epoch
            train_acc (torch.float32):The training score over the entire training dataset
        If index is int:
            model (nn.Module): The resulting model after training on one sample
    """
    model.train()
    train_loss, train_acc = 0,0
    if index is None:
        for batch_idx, (X, y) in enumerate(trainloader):
            X, y = X.to("cuda"), y.to("cuda")
            optimizer.zero_grad()
            pred = model(X).squeeze(dim = 1)

            batch_loss = loss_fn(pred, y)
            train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            
            batch_acc = accuracy_fn(pred, y)

            train_acc += batch_acc.item()
#             pbar.set_description(desc=f'Loss={batch_loss} Batch_id={batch_idx} Accuracy={100*correct/total:0.2f}')
    else:
        #1. Training on a single data
        X, y = trainloader.dataset[index]
        X, y = X.to("cuda"), y.to("cuda")
        optimizer.zero_grad()
        pred = model(X).squeeze(dim = 1)

        batch_loss = loss_fn(pred, y)
        batch_loss.backward()
        optimizer.step()
        
        return model
    train_loss /= len(trainloader)
    train_acc /= len(trainloader)
    return model, train_loss, train_acc
def test_step(model, testloader, loss_fn):
    """
    This function calculate the test error of a model
    Args:
        model (nn.Module): The model that is going to be tested
        testloader (nn.Module): The test dataloader of the model
        loss_fn: The objective function that should be minimized by the model
        metric_score_fn: The metric that is used to measure accuracy, such as F1_score, recall, etc.
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
    """
    Calculate the accuracy of the model
    Args:
        pred (torch.Tensor): The prediction of the model
        y (torch.Tensor): The true label of the dataset
    Returns:
        acc (torch.float32): The accuracy of the model
    """
    # if y has more than two labels
    if len(y.shape) > 1:
        pred_label = pred.argmax(dim = 1)
        y_label = y.argmax(dim = 1)
    # if y has only two labels
    else:
        pred_label = torch.round(pred)
        y_label = y
    acc = ((pred_label == y_label).sum()/len(y_label)).float()
    return acc