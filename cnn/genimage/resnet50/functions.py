import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.io import read_image, ImageReadMode


import time

class BinaryGenImage(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.img_dir = img_dir
        self.img_labels = []
        self.img_paths = []
        for label, generated in enumerate(["nature", "ai"]):
            image_files = os.listdir(os.path.join(img_dir, generated))
            for image_name in image_files:
                self.img_labels.append(label)
                self.img_paths.append(os.path.join(generated, image_name))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        try:
            image = read_image(img_path, mode=ImageReadMode.RGB)
        except:
            self.img_paths[idx] = modify_filename(self.img_dir, self.img_paths[idx])
            img_path = os.path.join(self.img_dir, self.img_paths[idx])
            image = read_image(img_path, mode=ImageReadMode.RGB)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    


class MulticlassGenImage(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.img_dir = img_dir
        self.img_labels = []
        self.img_paths = []
        for generated in ["ai", "nature"]:
            g = 0 if generated == "nature" else 1
            image_files = os.listdir(os.path.join(img_dir, generated))
            image_files = sorted(image_files, key=lambda entry: entry.lower())
            for l, image_name in enumerate(image_files):
                if g==1:
                    mc_label = int(image_name[:3])
                else:
                    mc_label = self.img_labels[l]['multiclass']
                self.img_labels.append({'binary' : g, 'multiclass' : mc_label})
                self.img_paths.append(os.path.join(generated, image_name))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            idx_ = modify_filename(self.img_dir, self.img_paths, idx)
            self.img_paths[idx] = self.img_paths[idx_]
            img_path = os.path.join(self.img_dir, self.img_paths[idx])
            image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def modify_filename(img_dir, img_paths, idx_):
    idx = idx_
    changed = False
    if idx < len(img_paths)//2:
        while changed == False:
            idx += 1
            if idx == len(img_paths):
                idx = 0
            try:
                Image.open(os.path.join(img_dir, img_paths[idx]))
                changed = True
            except:
                changed = False
    else:
        while changed == False:
            idx -= 1
            if idx == -1:
                idx = len(img_paths)-1
            try:
                Image.open(os.path.join(img_dir, img_paths[idx]))
                changed = True
            except:
                changed = False
    return idx



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss*(1 - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0




def compute_error_multioutput(device, model, loader):
    total_correct = {
        'binary': 0,
        'multiclass': 0,
        'combined': 0
    }
    total_images = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels_b = labels['binary'].to(device)
            labels_s = labels['multiclass'].to(device)

            output_b, output_s = model(inputs)
            _, predictions_b = torch.max(output_b, 1)
            _, predictions_s = torch.max(output_s, 1)
            total_images += labels['binary'].size(0)
            total_correct['binary'] += (predictions_b == labels_b).sum().item()
            total_correct['multiclass'] += (predictions_s == labels_s).sum().item()
            total_correct['combined'] += ((predictions_b == labels_b) & (predictions_s == labels_s)).sum().item()

    errors = {
        'binary' : 1-(total_correct['binary'] / total_images),
        'multiclass' : 1-(total_correct['multiclass'] / total_images),
        'combined' : 1-(total_correct['combined'] / total_images)
    }
    return errors



def train_network_multioutput(model, device, lr, epochs, train_dl, val_dl, writer=None, scheduler_patience=None, stopping_patience=None):
    criterion1 = torch.nn.CrossEntropyLoss()  # Classification task with 2 classes
    criterion2 = torch.nn.CrossEntropyLoss()  # Classification task with 1000 classes
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001)
    if scheduler_patience:
        scheduler = ReduceLROnPlateau(opt, 'min', patience=scheduler_patience, factor=0.1, threshold=0.01, threshold_mode='rel')
    if stopping_patience:
        early_stopping = EarlyStopping(patience=stopping_patience, min_delta=0.01)

    train_errors = {
        'binary': [],
        'multiclass': [],
        'combined': []
    }
    val_errors = {
        'binary': [],
        'multiclass': [],
        'combined': []
    }
    train_loss = []

    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        current_loss = 0.0
        current_lr = opt.param_groups[0]['lr']
        for Xt, Yt in train_dl:
            Xt = Xt.to(device)
            Yt_b = Yt['binary'].to(device)
            Yt_s = Yt['multiclass'].to(device)

            # Zero the parameter gradients
            opt.zero_grad()

            # Forward pass
            pred_b, pred_s = model(Xt)
            loss_b = criterion1(pred_b, Yt_b)
            loss_s = criterion2(pred_s, Yt_s)
            loss = loss_b + loss_s
            current_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            opt.step()

        # Compute validation loss
        model.eval()  
        val_loss = 0
        with torch.no_grad():
            for Xt, Yt in val_dl:
                Xt = Xt.to(device)
                Yt_b = Yt['binary'].to(device)
                Yt_s = Yt['multiclass'].to(device)
                # Forward pass
                pred_b, pred_s = model(Xt)
                loss_b = criterion1(pred_b, Yt_b)
                loss_s = criterion2(pred_s, Yt_s)
                loss = loss_b + loss_s
                val_loss += loss.item()
        val_loss /= len(val_dl)

        current_train_errors = compute_error_multioutput(device, model, train_dl)
        current_val_errors = compute_error_multioutput(device, model, val_dl)

        if writer:
            writer.add_scalar('Loss/train', current_loss/len(train_dl), epoch)
            for label in current_train_errors:
                writer.add_scalar('Error/train/'+label, current_train_errors[label], epoch)
            for label in current_val_errors:
                writer.add_scalar('Error/validation/'+label, current_val_errors[label], epoch)

        for label in current_train_errors:
            train_errors[label].append(current_train_errors[label])

        for label in current_val_errors:
            val_errors[label].append(current_val_errors[label])

        train_loss.append(current_loss/len(train_dl))
   
        if epoch < 5 or (epoch+1)%5 == 0:
            print(f"Epoch {epoch+1}: current lr = {current_lr:.0e}\nTrain error: Combined={train_errors['combined'][epoch]*100:.2f}%; Binary={train_errors['binary'][epoch]*100:.2f}%; multiclass={train_errors['multiclass'][epoch]*100:.2f}%; \nValidation error: Combined={val_errors['combined'][epoch]*100:.2f}%; Binary={val_errors['binary'][epoch]*100:.2f}%; multiclass={val_errors['multiclass'][epoch]*100:.2f}%; \nLoss: {(current_loss/len(train_dl)):.3e}")
        
        torch.save(model.state_dict(), './weights/' + str(epoch) + '.pth')
        # Call scheduler
        if scheduler_patience:
            scheduler.step(val_loss)
        # Call early stopping
        if stopping_patience:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Stopping training...")
                break
    return train_errors, val_errors, train_loss



def plot_training_stats_multioutput(train_errors, val_errors, train_loss):
    epochs = range(1, len(train_errors['binary']) + 1)

    plt.figure(figsize=(10, 7))

    # Plot training and validation binary error
    plt.subplot(2, 2, 1)
    plt.plot(epochs, np.array(train_errors['binary'])*100, label='Training Error')
    plt.plot(epochs, np.array(val_errors['binary'])*100, label='Validation Error')
    plt.title('Binary')
    plt.xlabel('Epochs')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0, top=max([err for c in list(train_errors.values()) + list(val_errors.values()) for err in c])*100)
    plt.xticks(np.arange(min(epochs), max(epochs)+1, max(1, len(epochs)//10)))

    # Plot training and validation multiclass error
    plt.subplot(2, 2, 2)
    plt.plot(epochs, np.array(train_errors['multiclass'])*100, label='Training Error')
    plt.plot(epochs, np.array(val_errors['multiclass'])*100, label='Validation Error')
    plt.title('Multiclass')
    plt.xlabel('Epochs')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0, top=max([err for c in list(train_errors.values()) + list(val_errors.values()) for err in c])*100)
    plt.xticks(np.arange(min(epochs), max(epochs)+1, max(1, len(epochs)//10)))

    # Plot training and validation combined error
    plt.subplot(2, 2, 3)
    plt.plot(epochs, np.array(train_errors['combined'])*100, label='Training Error')
    plt.plot(epochs, np.array(val_errors['combined'])*100, label='Validation Error')
    plt.title('Combined')
    plt.xlabel('Epochs')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0, top=max([err for c in list(train_errors.values()) + list(val_errors.values()) for err in c])*100)
    plt.xticks(np.arange(min(epochs), max(epochs)+1, max(1, len(epochs)//10)))

    # Plot training loss
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_loss, label='Training Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0)
    plt.xticks(np.arange(min(epochs), max(epochs)+1, max(1, len(epochs)//10)))

    plt.tight_layout()
    plt.show()