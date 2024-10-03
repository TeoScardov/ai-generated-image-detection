import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MulticlassGenImage(Dataset):
    def __init__(self, img_dir, val_gt=None, transform=None):
        self.transform = transform
        self.img_dir = img_dir
        self.img_labels = []
        self.img_paths = []
        with os.scandir(img_dir) as entries:
            directories = [entry.name for entry in entries if entry.is_dir()]
            for g, generator in enumerate(sorted(directories, key=lambda entry: entry.lower())):
                split = 'train'
                if val_gt:
                    split = 'val'
                for b, true_or_fake in enumerate(['nature', 'ai']):
                    gen_label = g + 1
                    if b == 0:
                        gen_label = 0
                    image_files = os.listdir(os.path.join(img_dir, generator, split, true_or_fake))
                    image_files = sorted(image_files, key=lambda entry: entry.lower())

                    mc_label = 0
                    previous_class = None
                    for image_name in image_files:
                        name_parts = image_name.split('_')
                        if b == 1:
                            mc_label = int(name_parts[0])
                        else:
                            if name_parts[0] != 'ILSVRC2012':
                                if previous_class:
                                    current_class = name_parts[0]
                                    if previous_class != current_class:
                                        previous_class = current_class
                                        mc_label += 1
                                else:
                                    previous_class = name_parts[0]
                            else: 
                                if val_gt:
                                    mc_label = val_gt[int(name_parts[-1].split('.')[0])]
                        self.img_labels.append({'multiclass' : mc_label, 'generator': gen_label})
                        self.img_paths.append(os.path.join(generator, split, true_or_fake, image_name))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def val_mapping(path):
    with open(os.path.join(path, 'imagenet_class_index.json'), 'r') as file:
        data = json.load(file)
        json_class_index = {value[0]:key for key, value in data.items()}

    mapping = {}
    with open(os.path.join(path, 'ILSVRC2012_mapping.txt'), 'r') as file:
        for line in file.readlines():
            try:
                mapping[line.strip().split(' ')[0]] = json_class_index[line.strip().split(' ')[-1]]
            except:
                break

    with open(os.path.join(path, 'ILSVRC2012_validation_ground_truth.txt'), 'r') as file:
        val_ground_truth = [None] + [int(mapping[line.strip()]) for line in file.readlines()]

    return val_ground_truth



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_stat = None
        self.early_stop = False

    def __call__(self, stat):
        if self.best_stat is None:
            self.best_stat = stat
        elif stat > self.best_stat*(1 - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_stat = stat
            self.counter = 0



def compute_error_multioutput(device, model, loader):
    total_correct = {
        'multiclass': 0,
        'generator': 0,
        'combined': 0
    }
    total_images = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            for k in labels:
                labels[k] = labels[k].to(device)

            outputs = model(inputs)
            predictions = {}
            for k in outputs:
                _, predictions[k] = torch.max(outputs[k], 1)
            total_images += labels[list(labels.keys())[0]].size(0)
            combined = (predictions[list(labels.keys())[0]] == labels[list(labels.keys())[0]])
            for k in predictions:
                c_ = (predictions[k] == labels[k])
                combined = combined & c_
                total_correct[k] += (predictions[k] == labels[k]).sum().item()
            total_correct['combined'] += combined.sum().item()

    errors = {}
    for k in total_correct:
        errors[k] = 1-(total_correct[k] / total_images)
    return errors



def train_network_multioutput(model, device, lr, epochs, train_dl, val_dl, writer=None, scheduler_patience=None, min_epochs=None, stopping_patience=None):
    criterions = {
        'multiclass': torch.nn.CrossEntropyLoss(),
        'generator': torch.nn.CrossEntropyLoss()
    }
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001)
    if scheduler_patience:
        scheduler = ReduceLROnPlateau(opt, 'min', patience=scheduler_patience, factor=0.1, threshold=0.01, threshold_mode='rel')
    if stopping_patience:
        early_stopping = EarlyStopping(patience=stopping_patience, min_delta=0.01)

    train_errors = {
        'multiclass': [],
        'generator': [],
        'combined': []
    }
    val_errors = {
        'multiclass': [],
        'generator': [],
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
            for k in Yt:
                Yt[k] = Yt[k].to(device)

            # Zero the parameter gradients
            opt.zero_grad()

            # Forward pass
            predictions = model(Xt)
            losses = {}
            for k in predictions:
                losses[k] = criterions[k](predictions[k], Yt[k])
            loss = 0
            for value in losses.values():
                loss += value
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
                for k in Yt:
                    Yt[k] = Yt[k].to(device)
                # Forward pass
                predictions = model(Xt)
                losses = {}
                for k in predictions:
                    losses[k] = criterions[k](predictions[k], Yt[k])
                loss = 0
                for value in losses.values():
                    loss += value
                val_loss += loss.item()
        val_loss /= len(val_dl)

        current_train_errors = compute_error_multioutput(device, model, train_dl)
        current_val_errors = compute_error_multioutput(device, model, val_dl)

        if writer:
            writer.add_scalar('Loss/train', current_loss/len(train_dl), epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Param/LR', current_lr, epoch)
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
            print(f"- Epoch {epoch+1}: current lr = {current_lr:.0e}\nTrain error: Combined={train_errors['combined'][epoch]*100:.2f}%; Multiclass={train_errors['multiclass'][epoch]*100:.2f}%; Generator={train_errors['generator'][epoch]*100:.2f}%; \nValidation error: Combined={val_errors['combined'][epoch]*100:.2f}%;  Multiclass={val_errors['multiclass'][epoch]*100:.2f}%; Generator={val_errors['generator'][epoch]*100:.2f}%; \nTrain loss: {(current_loss/len(train_dl)):.3e}; Val loss: {(val_loss):.3e}")
        
        if (epoch+1)%20 == 0:
            torch.save(model.state_dict(), './weights/resnet50_v3_e' + str(epoch+1) + '.pth')
        # Call scheduler
        if scheduler_patience:
            scheduler.step(val_errors['combined'][-1])
        # Call early stopping
        if min_epochs:
            if epoch > min_epochs:
                early_stopping(val_errors['combined'][-1])
                if early_stopping.early_stop:
                    print("Stopping training...")
                    break
    return train_errors, val_errors, train_loss



def plot_training_stats_multioutput(train_errors, val_errors, train_loss):
    epochs = range(1, len(train_errors['combined']) + 1)

    plt.figure(figsize=(12, 9))

    # Plot training and validation error
    for n, k in enumerate(train_errors):
        plt.subplot(2, 2, n+1)
        plt.plot(epochs, np.array(train_errors[k])*100, label='Training Error')
        plt.plot(epochs, np.array(val_errors[k])*100, label='Validation Error')
        plt.title(k)
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


def multioutput_accuracy_map(device, model, loader, size):
    from sklearn.metrics import average_precision_score
    total_correct = {
        'multiclass': 0,
        'generator': 0,
        'combined': 0
    }
    total_images = 0
    
    # Variables to store true labels and predicted scores for mAP calculation
    all_labels = {
        'multiclass': [],
        'generator': []
    }
    all_outputs = {
        'multiclass': [],
        'generator': []
    }
    
    # Set the model into evaluation mode
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            for k in labels:
                labels[k] = labels[k].to(device)
                all_labels[k].append(labels[k].cpu())
            
            predictions = {}
            outputs = model(inputs)
            for k in outputs:
                _, predictions[k] = torch.max(outputs[k], 1)
                all_outputs[k].append(outputs[k].cpu())
            total_images += labels[list(labels.keys())[0]].size(0)
            combined = (predictions[list(labels.keys())[0]] == labels[list(labels.keys())[0]])
            for k in predictions:
                c_ = (predictions[k] == labels[k])
                combined = combined & c_
                total_correct[k] += (predictions[k] == labels[k]).sum().item()
            total_correct['combined'] += combined.sum().item()
    
    errors = {}
    for k in total_correct:
        errors[k] = 1-(total_correct[k] / total_images)
    
    for k in all_labels:
        all_labels[k] = torch.cat(all_labels[k])
        all_outputs[k] = torch.cat(all_outputs[k])
    
    # Calculate mAP
    mAPs = {
        'multiclass': 0.0,
        'generator': 0.0
    }
    for k in mAPs:
        for i in range(size[k]):
            binary_labels = (all_labels[k] == i).int()
            binary_outputs = all_outputs[k][:, i]
            ap = average_precision_score(binary_labels, binary_outputs)
            mAPs[k] += ap
            
        mAPs[k] = mAPs[k]/size[k]  # Average over all classes
    
    return errors, mAPs


def multioutput_accuracy_map_modified(device, model, loader, size):
    from sklearn.metrics import average_precision_score
    correct = 0
    total_images = 0
    
    # Variables to store true labels and predicted scores for mAP calculation
    all_labels = {
        'multiclass': [],
        'generator': []
    }
    all_outputs = {
        'multiclass': [],
        'generator': []
    }
    
    # Set the model into evaluation mode
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            for k in labels:
                labels[k] = labels[k].to(device)
                all_labels[k].append(labels[k].cpu())
            
            predictions = {}
            outputs = model(inputs)
            for k in outputs:
                _, predictions[k] = torch.max(outputs[k], 1)
                all_outputs[k].append(outputs[k].cpu())
            total_images += labels[list(labels.keys())[0]].size(0)
            combined = (predictions[list(labels.keys())[0]] == labels[list(labels.keys())[0]])

            p1 = (predictions['generator'] == 0) & (labels['generator'] == 0)
            p2 = (predictions['generator'] > 0) & (labels['generator'] > 0)
            correct += (p1 | p2).sum().item()
    

    error = 1-(correct / total_images)
    
    for k in all_labels:
        all_labels[k] = torch.cat(all_labels[k])
        all_outputs[k] = torch.cat(all_outputs[k])
    
    # Calculate mAP
    mAP = 0.0
    all_labels['generator'][all_labels['generator'] > 0] = 1
    for i in range(2):
        binary_labels = (all_labels['generator'] == i).int()
        if i == 0:
            binary_outputs = all_outputs['generator'][:, 0]
        if i == 1:
            binary_outputs = torch.sum(all_outputs['generator'][:, 1:], dim=1)
        ap = average_precision_score(binary_labels, binary_outputs)
        mAP += ap
        
    mAP = mAP/2  # Average over all classes
    
    return error, mAP


def natural_accuracy(device, model, loader, size):
    total_correct = 0
    total_images = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            for k in labels:
                labels[k] = labels[k].to(device)

            outputs = model(inputs)
            predictions = {}
            for k in outputs:
                _, predictions[k] = torch.max(outputs[k], 1)
            
            p1 = (labels['generator'] == 0)
            p2 = (predictions['multiclass'] == labels['multiclass'])
            total_images += p1.sum().item()
            total_correct += (p1 & p2).sum().item()

    error = 1-(total_correct / total_images)
    return error

