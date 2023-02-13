from torch.utils.data import Dataset
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm



class CustomDataset(Dataset):
    def __init__(self, X , y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx] , self.y[idx])

def trainModel(model : torch.nn.Module, 
               dataLoader : torch.utils.data.DataLoader,
               numIter : int, 
               loss_fn,
               optimizer : torch.optim.Optimizer):
  
    for iter in tqdm(range(numIter)):
        for data in dataLoader:
            input , true_label = data

            try:
                input, true_label = input.to("cuda"), true_label.to("cuda")

                pred_label = model(input).squeeze(dim = 1)
                pred_label_squeezed = pred_label
                loss = loss_fn(pred_label_squeezed, true_label) 
            except IndexError:
                print(f"Input shape {input.shape}")
                print(f"Prediction logits shape {pred_label.shape}")
                print(f"Prediction logits squeezed shape {pred_label_squeezed.shape}")
                print(f"True label shape {true_label.shape}")
                raise IndexError
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
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