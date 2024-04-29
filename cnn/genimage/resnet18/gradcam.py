import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import numpy as np
import os
import PIL

def backward_hook(module, grad_input, grad_output):
  global gradients
  print('Backward hook running...')
  gradients = grad_output
  print(f'Gradients size: {gradients[0].size()}') 



def forward_hook(module, args, output):
  global activations
  print('Forward hook running...')
  activations = output
  print(f'Activations size: {activations.size()}')



def plot_gradcam_multioutput(model, target_layer, dataset, labels_map, output_type, sample_idx):
    global gradients
    global activations
    gradients = None
    activations = None

    # Register hooks
    backward_hook_handle = target_layer.register_full_backward_hook(backward_hook, prepend=False)
    forward_hook_handle = target_layer.register_forward_hook(forward_hook, prepend=False)

    # Get the image tensor and its label
    img_path = os.path.join(dataset.img_dir, dataset.img_paths[sample_idx])
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    img_tensor = transform(img)
    label = torch.tensor([dataset.img_labels[sample_idx][output_type]], dtype=torch.long, device='cpu') 

    model.eval()
    # Forward pass
    output = {}
    output['binary'], output['multiclass'] = model(img_tensor.unsqueeze(0))
    loss_function = torch.nn.CrossEntropyLoss()
    loss = loss_function(output[output_type], label)
    # Backward pass
    loss.backward()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    # weight the channels by corresponding gradients
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    heatmap = F.relu(heatmap)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    # Remove the hooks
    backward_hook_handle.remove()
    forward_hook_handle.remove()

    # Get the prediction
    with torch.no_grad():  
        logits = {}
        logits['binary'], logits['multiclass'] = model(img_tensor.unsqueeze(0))
        predicted_label = torch.argmax(logits[output_type], dim=1).item()

    # Plot the results
    plt.figure(figsize=(10, 5))
    # Original image
    plt.subplot(1, 3, 1)
    plt.axis('off')
    img, true_label = dataset[sample_idx]
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min())
    plt.title(f'Predicted: {labels_map[predicted_label]};   True: {labels_map[true_label[output_type]]}')
    plt.imshow(img)
    # Overlayed image
    plt.subplot(1, 3, 2)
    plt.axis('off') 
    plt.imshow(img)
    overlay = to_pil_image(heatmap.detach(), mode='F').resize((224,224))
    cmap = colormaps['jet']
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    plt.title('Overlayed image')
    plt.imshow(overlay, alpha=0.6, interpolation='nearest')
    # Heatmap
    plt.subplot(1, 3, 3)
    plt.imshow(heatmap.detach())  # Pass the figure number to matshow
    plt.title('Heatmap')  # Titles with matshow need to be handled differently as shown below
    plt.axis('off')  # Turn off the axis
    plt.tight_layout()
    plt.show()