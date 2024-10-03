
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BinaryCIFAKE(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.img_dir = img_dir
        self.img_labels = []
        self.img_paths = []
        for label, generated in enumerate(["REAL", "FAKE"]):
            image_files = os.listdir(os.path.join(img_dir, generated))
            for image_name in image_files:
                self.img_labels.append(label)
                self.img_paths.append(os.path.join(generated, image_name))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



class MulticlassCIFAKE(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.img_dir = img_dir
        self.img_labels = []
        self.img_paths = []
        for g, generated in enumerate(["REAL", "FAKE"]):
            image_files = os.listdir(os.path.join(img_dir, generated))
            image_files = sorted(image_files, key=lambda entry: entry.lower())
            for label, image_name in enumerate(image_files):
                self.img_labels.append(label%10 + g*10)
                self.img_paths.append(os.path.join(generated, image_name))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class MultioutputCIFAKE(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.img_dir = img_dir
        self.img_labels = []
        self.img_paths = []
        for g, generated in enumerate(["REAL", "FAKE"]):
            image_files = os.listdir(os.path.join(img_dir, generated))
            image_files = sorted(image_files, key=lambda entry: entry.lower())
            for label, image_name in enumerate(image_files):
                self.img_labels.append({'binary' : g, 'multiclass' : label%10})
                self.img_paths.append(os.path.join(generated, image_name))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    


def compute_error(device, model, loader):
    total_correct = 0
    total_samples = 0
    with torch.no_grad():  # No need to track gradients for evaluation
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            _, predictions = torch.max(model(inputs), 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    return 1-accuracy



def train_network(model, device, lr, epochs, train_dl, val_dl, writer=None, scheduler_patience=None, min_epochs=None, stopping_patience=None):
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001)
    if scheduler_patience:
        scheduler = ReduceLROnPlateau(opt, 'min', patience=scheduler_patience, factor=0.1, threshold=0.01, threshold_mode='rel')
    if stopping_patience:
        early_stopping = EarlyStopping(patience=stopping_patience, min_delta=0.01)
    train_err = []
    val_err = []
    train_loss = []

    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        current_loss = 0.0
        current_lr = opt.param_groups[0]['lr']
        for Xt, Yt in train_dl:
            Xt = Xt.to(device)
            Yt = Yt.to(device)

            # Zero the parameter gradients
            opt.zero_grad()

            # Forward pass
            loss = criterion(model(Xt), Yt)
            current_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            opt.step()

            

        # Set the model to evaluation mode
        model.eval()  
        val_loss = 0
        with torch.no_grad():
            for Xt, Yt in val_dl:
                Xt = Xt.to(device)
                Yt = Yt.to(device)
            # Forward pass
            loss = criterion(model(Xt), Yt)
            val_loss += loss.item()
        val_loss /= len(val_dl)

        current_train_err = compute_error(device, model, train_dl)
        current_val_err = compute_error(device, model, val_dl)

        if writer:
            writer.add_scalar('Loss/train', current_loss/len(train_dl), epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Param/LR', current_lr, epoch)
            writer.add_scalar('Error/train', current_train_err, epoch)
            writer.add_scalar('Error/validation', current_val_err, epoch)

        train_err.append(current_train_err)
        val_err.append(current_val_err)
        train_loss.append(current_loss/len(train_dl))

        if epoch < 5 or (epoch+1)%5 == 0:
            print(f"Epoch {epoch+1}; Train err = {train_err[epoch]*100:.2f}%; Val err = {val_err[epoch]*100:.2f}%; Loss: {(current_loss/len(train_dl)):.4f}")

        # Call scheduler
        if scheduler_patience:
            scheduler.step(val_err[-1])
        # Call early stopping
        if min_epochs:
            if epoch > min_epochs:
                early_stopping(val_err[-1])
                if early_stopping.early_stop:
                    print(f"Epoch {epoch+1}; Train err = {train_err[epoch]*100:.2f}%; Val err = {val_err[epoch]*100:.2f}%; Loss: {(current_loss/len(train_dl)):.4f}")
                    print("Stopping training...")
                    break
    return train_err, val_err, train_loss



def plot_training_stats(train_err, val_err, train_loss):
    epochs = range(1, len(train_err) + 1)
    plt.figure(figsize=(10, 5))

    # Plot training and validation error
    plt.subplot(1, 2, 1)
    plt.plot(epochs, np.array(train_err)*100, label='Training Error')
    plt.plot(epochs, np.array(val_err)*100, label='Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0)  # Set the lower y-axis limit to zero
    plt.xticks(np.arange(min(epochs), max(epochs)+1, max(1, len(epochs)//10)))

    # Plot training loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Training Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0)
    plt.xticks(np.arange(min(epochs), max(epochs)+1, max(1, len(epochs)//10)))
    plt.tight_layout()
    plt.show()



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
        'binary': 0,
        'multiclass': 0,
        'combined': 0
    }
    total_images = 0
    with torch.no_grad():  # No need to track gradients for evaluation
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



def train_network_multioutput(model, device, lr, epochs, train_dl, val_dl, writer=None, scheduler_patience=None, min_epochs=None, stopping_patience=None):
    criterion1 = torch.nn.CrossEntropyLoss()  # Classification task with 2 classes
    criterion2 = torch.nn.CrossEntropyLoss()  # Classification task with 10 classes
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
            print(f"- Epoch {epoch+1}: current lr = {current_lr:.0e}\nTrain error: Combined={train_errors['combined'][epoch]*100:.2f}%; Binary={train_errors['binary'][epoch]*100:.2f}%; multiclass={train_errors['multiclass'][epoch]*100:.2f}%; \nValidation error: Combined={val_errors['combined'][epoch]*100:.2f}%; Binary={val_errors['binary'][epoch]*100:.2f}%; multiclass={val_errors['multiclass'][epoch]*100:.2f}%; \nLoss: {(current_loss/len(train_dl)):.3e}")
        
        #torch.save(model.state_dict(), './weights/' + str(epoch) + '.pth')
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



def make_confusion_matrix(device, model, loader, size):
    total_correct = 0
    total_samples = 0
    confusion_matrix = torch.zeros(size, size)
    
    # Variables to store true labels and predicted scores for mAP calculation
    all_labels = []
    all_outputs = []
    
    # Set the model into evaluation mode
    model.eval() 
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            # Collecting data for mAP calculation
            all_labels.append(labels.cpu())
            all_outputs.append(outputs.cpu())
            
            # Perform the predictions
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update the confusion matrix
            for label, prediction in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[label.long(), prediction.long()] += 1
    
    test_accuracy = total_correct / total_samples
    
    # Convert list of tensors to single tensors
    all_labels = torch.cat(all_labels)
    all_outputs = torch.cat(all_outputs)
    
    # Calculate mAP
    mAP = 0.0
    for i in range(size):
        binary_labels = (all_labels == i).int()
        binary_outputs = all_outputs[:, i]
        
        precision, recall, _ = precision_recall_curve(binary_labels, binary_outputs)
        ap = average_precision_score(binary_labels, binary_outputs)
        mAP += ap
        
    mAP /= size  # Average over all classes
    
    return confusion_matrix, test_accuracy, mAP



def plot_confusion_matrix(cm, class_names, figsize):
    # Normalize
    cm = cm.numpy()
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Plot the CM
    plt.figure(figsize=figsize)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.gca().xaxis.set_ticks_position('top') 
    plt.gca().xaxis.set_label_position('top')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation='vertical')
    plt.show()



def visualise_samples(model, test_ds, labels_map, img_size=(224, 224)):
    figure = plt.figure(figsize=(11, 11))
    cols, rows = 3, 3
    model.eval()  
    for i in range(1, cols * rows + 1):
        # Get a random image
        sample_idx = torch.randint(len(test_ds), size=(1,)).item()
        img, true_label = test_ds[sample_idx]
        img_tensor = img.unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(next(model.parameters()).device)
        # Perform the prediction
        with torch.no_grad():  
            logits = model(img_tensor)
            predicted_label = torch.argmax(logits, dim=1).item()
        # Add the image to the plot
        figure.add_subplot(rows, cols, i)
        plt.title(f'Predicted: {labels_map[predicted_label]}\nTrue: {labels_map[true_label]}')
        plt.axis("off")
        img = img.cpu().numpy().transpose((1, 2, 0))
        img = Image.fromarray((img * 255).astype('uint8'))
        plt.imshow(img.resize(img_size, resample=Image.LANCZOS)) # The image is upsampled for better visualization
    plt.show()


