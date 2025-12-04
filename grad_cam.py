import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt

#Example inputs for functions
#Load and preprocess image
#  img_path = "/content/extracted_files/EuroSAT_RGB/Pasture/Pasture_1.jpg"
#  img = Image.open(img_path).convert("RGB")
#  input_tensor = transform(img).unsqueeze(0).to(device)

# Directory for saving GIFs
#  os.makedirs("gradcam_imgs", exist_ok=True)
#  os.makedirs("gradcam_gifs", exist_ok=True)

# Layers to inspect. These are given from the model
#  layer_names = ["conv1", "conv2", "conv3", "conv4"]

def gradcam(model, image_tensor, target_layer):
    model.eval()

    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # forward pass
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    # zero gradients, backprop from predicted class
    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()

    # compute Grad-CAM
    grads = gradients[0]           # shape: [batch, channels, H, W]
    acts = activations[0]          # shape: same

    weights = grads.mean(dim=[2, 3], keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    # normalize and resize
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[3]))

    # remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return cam, pred_class




def create_cam_grad_gif(model, image_path, layer_names):

    image_path = Path(image_path)
    gradcam_img_dir = Path("gradcam_imgs")
    gradcam_gif_dir = Path("gradcam_gifs")

    #create directories if they don't exist
    gradcam_img_dir.mkdir(parents=True, exist_ok=True)
    gradcam_gif_dir.mkdir(parents=True, exist_ok=True)

    images = []

    #load and show original image
    img = Image.open(image_path).convert("RGB")
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.title(f"Real Class {image_path.stem.split('_', 1)[0]}")

    raw_path = gradcam_img_dir / "raw.png"
    plt.savefig(raw_path)
    images.append(Image.open(raw_path))

    # get predicted class
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    outputs = model(input_tensor)
    _, pred_class = torch.max(outputs, 1)

    # generate grad cam overlays
    for lname in layer_names:
        target_layer = getattr(model, lname)
        cam, _ = gradcam(model, input_tensor, target_layer)

        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_cv, 1, heatmap, 0.3, 0)

        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"Predicted: {categories[pred_class]}, Real: {image_path.stem.split('_', 1)[0]}")
        plt.xlabel(f'{lname}')

        out_path = gradcam_img_dir / f"{lname}.png"
        plt.savefig(out_path)
        images.append(Image.open(out_path))

    # save gif
    gif_path = gradcam_gif_dir / f"{image_path.stem}.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=2000, loop=0)

    print(f"gradcam gif saved to: {gif_path.resolve()}")

#Implementation
#create_cam_grad_gif(model, img_path, layer_names)
