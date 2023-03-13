from torch.utils.data import Dataset
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

__all__ = ['CustomDataset', 'train_step', 'test_step', 'gatherInference']
class CustomDataset(Dataset):
    def __init__(self, X , y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx] , self.y[idx])


def train_step(model, trainloader, loss_fn, optimizer):

    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(trainloader):
        X, y = X.to("cuda"), y.to("cuda")
        optimizer.zero_grad()
        pred = model(X).squeeze(dim = 1)
        batch_loss = loss_fn(pred, y)
        train_loss += batch_loss.item()

        batch_loss.backward()
        optimizer.step()

    train_loss /= len(trainloader)

    return model, train_loss
def test_step(model, testloader, loss_fn):
    
    test_loss = 0
    model.eval()
    with torch.inference_mode():
        for (X_test, y_test) in testloader:
            test_pred = model(X_test).squeeze(dim = 1)
            test_loss += loss_fn(test_pred, y_test).item()
    
        test_loss /= len(testloader)

    return test_loss

def gatherInference(model, dataLoader):

    predLabels = []
    inputLabels = []
    
    with torch.inference_mode():
        for data in dataLoader:
            input, label = data
            input = input.to("cuda")
            label = label.to("cuda")

            pred  = model(input)
            predLabels.append(pred)
            inputLabels.append(label)
    
    predLabels = torch.cat(predLabels, dim = 0)
    inputLabels = torch.cat(inputLabels, dim = 0)

    return (inputLabels, predLabels)
    