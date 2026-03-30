import torch
import torch.nn.functional as F
import cv2
import numpy as np


def generate_gradcam(model, input_tensor, target_layer, target_class=None):
    """
    model         : trained PyTorch model
    input_tensor  : input image tensor (1, C, H, W)
    target_layer  : convolution layer (e.g., model.layer4[-1])
    target_class  : class index (e.g., 1 for Disease). If None, uses predicted class.
    """

    model.eval()
    gradients = []
    activations = []

    # Forward hook
    def forward_hook(module, input, output):
        activations.append(output)

    # Backward hook (correct version)
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    with torch.enable_grad():
        output = model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[:, target_class]

        model.zero_grad()
        score.backward()

    # Get gradients and activations
    grads = gradients[0].detach().cpu().numpy()[0]      # (C, H, W)
    acts = activations[0].detach().cpu().numpy()[0]     # (C, H, W)

    # Compute weights
    weights = np.mean(grads, axis=(1, 2))  # Global Average Pooling

    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)

    # Resize correctly
    _, _, H, W = input_tensor.shape
    cam = cv2.resize(cam, (W, H))

    # Normalize safely
    cam = cam - np.min(cam)
    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    return cam


def overlay_heatmap(cam, original_img):
    """
    cam           : output from generate_gradcam (H, W)
    original_img  : original image in numpy format (H, W, 3)
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    if original_img.max() > 1:
        original_img = original_img / 255.0

    overlay = heatmap + original_img
    overlay = overlay / np.max(overlay)

    return np.uint8(255 * overlay)