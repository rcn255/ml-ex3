import sys
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import struct

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)

class Cifar10CnnModelDropout(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
            nn.Dropout(0.25),  # Add dropout

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
            nn.Dropout(0.25), 

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
            nn.Dropout(0.25), 

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, 10)
        )
        
    def forward(self, xb):
        return self.network(xb)

class GreyscaleCnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input channels changed to 1 for greyscale
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 14 x 14

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 7 x 7

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 3 x 3

            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1024),  # Adjusted input size to match the output of the last conv layer
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  # Output layer for 10 classes
        )
        
    def forward(self, xb):
        return self.network(xb)

class GreyscaleCnnModelDropout(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input channels changed to 1 for greyscale
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 14 x 14
            nn.Dropout(0.25),  # Add dropout

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 7 x 7
            nn.Dropout(0.25),  # Add dropout

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 3 x 3
            nn.Dropout(0.25),  # Add dropout

            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1024),  # Adjusted input size to match the output of the last conv layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout
            nn.Linear(512, 10)  # Output layer for 10 classes
        )
        
    def forward(self, xb):
        return self.network(xb)
# Aux functions
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)  # Assuming y_data are labels and should be long integers

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]  # Change dimensions to (3, 32, 32) if necessary
        y = self.y_data[idx]
        return x, y

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def read_labels(filename):
    with open(filename, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def load_test_set(test_set):
    if test_set == 1: # CIFAR-10
        with open('../paul/data/cifar-10-batches-py/test_batch', 'rb') as file:
            test_batch = pickle.load(file, encoding='bytes')
            x_test = test_batch[b'data'].reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
            y_test = np.array(test_batch[b'labels'])

        with open('../paul/data/cifar-10-batches-py/batches.meta', 'rb') as file:
            batch = pickle.load(file, encoding='bytes')
            label_names = batch[b'label_names']

        # Convert to tensor
        x_test = torch.tensor(x_test, dtype=torch.uint8)
        x_test = x_test.permute(0, 3, 1, 2)

    else: # FashionMNIST
        base_path = '../paul/data/minst_clothing/'
        test_images_path = base_path + 't10k-images-idx3-ubyte'
        test_labels_path = base_path + 't10k-labels-idx1-ubyte'

        x_test = read_idx(test_images_path)
        y_test = read_labels(test_labels_path)

        label_names = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot"
        ]

        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        x_test = x_test.unsqueeze(1)

    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return x_test, y_test, label_names

def load_model(test_set, model_number):
    device = torch.device('cpu')
    print(f"Running on device: {device}")

    if test_set == 2:
        if model_number == 4:
            model = to_device(GreyscaleCnnModelDropout(), device)
        else:
            model = to_device(GreyscaleCnnModel(), device)
        model.load_state_dict(torch.load('models/fashion/fashion-'+str(model_number)+'.pth', map_location ='cpu'))
    else:
        if model_number == 4:
            model = to_device(Cifar10CnnModelDropout(), device)
        else:
            model = to_device(Cifar10CnnModel(), device)
        model.load_state_dict(torch.load('models/cifar10-'+str(model_number)+'.pth', map_location ='cpu'))

    return model


if __name__ == '__main__':
    try:
        test_set = int(sys.argv[1])
        model_number = int(sys.argv[2])
    except:
        print(f"Did not receive an integer for either argument.")

    x_test, y_test, label_names = load_test_set(test_set)
    model = load_model(test_set, model_number)

    test_dataset = CustomDataset(x_test, y_test)
    test_loader = DeviceDataLoader(DataLoader(test_dataset, 128*2), torch.device('cpu'))

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            preds = preds.argmax(dim=1)  # Predicted class
            all_preds.extend(preds.cpu().numpy())  # Convert to numpy array
            all_labels.extend(yb.cpu().numpy()) # Add to list

    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy:.4f}')

    # F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Weighted F1 Score: {f1:.4f}')

    # F1 score
    class_report = classification_report(all_labels, all_preds)
    print(f'Classification Report: {class_report}')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix [FashionMNIST, model 2]')
    plt.show()