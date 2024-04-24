
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


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



def train_network(model, device, lr, epochs, train_dl, val_dl):
    criterion = torch.nn.CrossEntropyLoss()
    #opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    current_epoch = 0
    train_err = []
    val_err = []
    train_loss = []

    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        current_loss = 0.0
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
        current_train_err = compute_error(device, model, train_dl)
        current_val_err = compute_error(device, model, val_dl)

        train_err.append(current_train_err)
        val_err.append(current_val_err)
        train_loss.append(current_loss/len(train_dl))

        current_epoch += 1    
        if current_epoch < 6 or current_epoch%5 == 0:
            print(f"Epoch {epoch+1}; Train err = {train_err[epoch]*100:.2f}%; Val err = {val_err[epoch]*100:.2f}%; Loss: {(current_loss/len(train_dl)):.4f}")
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



def train_network_multioutput(opt, model, device, epochs, train_dl, val_dl):
    criterion1 = torch.nn.CrossEntropyLoss()  # Classification task with 2 classes
    criterion2 = torch.nn.CrossEntropyLoss()  # Classification task with 10 classes

    current_epoch = 0
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

        # Set the model to evaluation mode
        model.eval()  
        current_train_errors = compute_error_multioutput(device, model, train_dl)
        current_val_errors = compute_error_multioutput(device, model, val_dl)

        train_errors['binary'].append(current_train_errors['binary'])
        train_errors['multiclass'].append(current_train_errors['multiclass'])
        train_errors['combined'].append(current_train_errors['combined'])

        val_errors['binary'].append(current_val_errors['binary'])
        val_errors['multiclass'].append(current_val_errors['multiclass'])
        val_errors['combined'].append(current_val_errors['combined'])

        train_loss.append(current_loss/len(train_dl))

        current_epoch += 1    
        if current_epoch < 6 or current_epoch%5 == 0:
            print(f'''  Epoch {epoch+1}:  
        Train error: Combined={train_errors['combined'][epoch]*100:.2f}%; Binary={train_errors['binary'][epoch]*100:.2f}%; multiclass={train_errors['multiclass'][epoch]*100:.2f}%; 
        Validation error: Combined={val_errors['combined'][epoch]*100:.2f}%; Binary={val_errors['binary'][epoch]*100:.2f}%; multiclass={val_errors['multiclass'][epoch]*100:.2f}%; 
        Loss: {(current_loss/len(train_dl)):.4f}''')
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
    # Set the model into evaluation mode
    model.eval() 
    device = next(model.parameters()).device
    with torch.no_grad():  
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # Perform the predictions
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            # Update the confusion matrix
            for label, prediction in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[label.long(), prediction.long()] += 1
    test_accuracy = total_correct / total_samples
    return confusion_matrix, test_accuracy



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
        plt.imshow(img.resize(img_size, resample=Image.Resampling.LANCZOS)) # The image is upsampled for better visualization
    plt.show()