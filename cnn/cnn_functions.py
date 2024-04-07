
import torch
import matplotlib.pyplot as plt
import numpy as np

def compute_error(device, model, loader):
    total_correct = 0
    total_images = 0
    with torch.no_grad():  # No need to track gradients for evaluation
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            _, predictions = torch.max(model(inputs), 1)
            total_images += labels.size(0)
            total_correct += (predictions == labels).sum().item()
    accuracy = total_correct / total_images
    return 1-accuracy

def train_network(opt, model, device, epochs, train_dl, val_dl):
    criterion = torch.nn.CrossEntropyLoss()

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
            print(f"Epoch {epoch+1}; Train err = {train_err[epoch]*100:.2f}; Val err = {val_err[epoch]*100:.2f}; Loss: {(current_loss/len(train_dl)):.4f}")
    return train_err, val_err, train_loss


def plot_accuracies(train_err, val_err, train_loss):
    epochs = range(1, len(train_err) + 1)

    plt.figure(figsize=(10, 5))

    # Plot training and validation error
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_err, label='Training Error')
    plt.plot(epochs, val_err, label='Validation Error')
    plt.title('Training and Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0)  # Set the lower y-axis limit to zero
    plt.xticks(np.arange(min(epochs), max(epochs)+1, 1.0))  # Ensure x-axis ticks are only integers

    # Plot training loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Training Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0)  # Set the lower y-axis limit to zero
    plt.xticks(np.arange(min(epochs), max(epochs)+1, 1.0))  # Ensure x-axis ticks are only integers

    plt.tight_layout()
    plt.show()