
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001)

    scheduler = ReduceLROnPlateau(opt, 'min', patience=2, factor=0.1, threshold=0.05, threshold_mode='rel', min_lr=0.000001)
    early_stopping = EarlyStopping(patience=3, min_delta=0.05)

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

        train_err.append(current_train_err)
        val_err.append(current_val_err)
        train_loss.append(current_loss/len(train_dl))

        current_epoch += 1    
        if current_epoch == 1:
            print(f"Epoch {epoch+1}; Train err = {train_err[epoch]*100:.2f}%; Val err = {val_err[epoch]*100:.2f}%; Val loss: {(val_loss/len(train_dl)):.4f}; current lr = {lr:.0e}")
        else:
            print(f"Epoch {epoch+1}; Train err = {train_err[epoch]*100:.2f}%; Val err = {val_err[epoch]*100:.2f}%; Val loss: {(val_loss/len(train_dl)):.4f}; current lr = {scheduler.get_last_lr()[0]:.0e}")

        scheduler.step(val_loss)
        # Call early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Stopping training...")
            break
    return train_err, val_err, train_loss







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