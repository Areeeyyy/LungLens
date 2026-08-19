"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
=====================================================

Kompatibel dengan Keras 2 dan Keras 3 untuk Sequential Model.
"""

import numpy as np
import tensorflow as tf
from PIL import Image


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str | None = None,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for Sequential models.

    Parameters
    ----------
    model : tf.keras.Model
        The trained Sequential CNN model.
    img_array : np.ndarray
        Preprocessed image array with shape (1, H, W, 3), values in [0, 1].
    last_conv_layer_name : str or None
        Name of the last convolutional layer. If None, automatically
        detects the last Conv2D layer in the model.

    Returns
    -------
    np.ndarray
        Heatmap array with shape (H, W), values normalized to [0, 1].
    """
    # Find index of the target conv layer
    target_layer_idx = None
    if last_conv_layer_name is not None:
        for idx, layer in enumerate(model.layers):
            if layer.name == last_conv_layer_name:
                target_layer_idx = idx
                break
    else:
        for idx in range(len(model.layers) - 1, -1, -1):
            if isinstance(model.layers[idx], tf.keras.layers.Conv2D):
                target_layer_idx = idx
                break

    if target_layer_idx is None:
        raise ValueError("No Conv2D layer found in the model.")

    x = tf.cast(img_array, tf.float32)

    # Layer-by-layer forward pass inside GradientTape
    with tf.GradientTape() as tape:
        # Pass input through layers up to target Conv2D layer
        curr = x
        for i in range(target_layer_idx + 1):
            curr = model.layers[i](curr)

        conv_output = curr
        tape.watch(conv_output)

        # Pass remaining layers to get output prediction
        preds = conv_output
        for i in range(target_layer_idx + 1, len(model.layers)):
            preds = model.layers[i](preds)

        loss = preds[:, 0]

    # Compute gradient of loss w.r.t. conv_output
    grads = tape.gradient(loss, conv_output)

    # Global Average Pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight activation maps
    conv_outputs = conv_output[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU & normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """
    Overlay the Grad-CAM heatmap onto the original image.
    """
    img = original_image.convert("RGB")
    img_width, img_height = img.size

    # Resize heatmap to match original image dimensions
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_pil = Image.fromarray(heatmap_uint8).resize(
        (img_width, img_height), Image.BILINEAR
    )
    heatmap_resized = np.array(heatmap_pil, dtype=np.float32) / 255.0

    # Apply Jet colormap
    heatmap_colored = _apply_jet_colormap(heatmap_resized)

    # Blend heatmap with original image
    img_array = np.array(img, dtype=np.float32) / 255.0
    superimposed = img_array * (1 - alpha) + heatmap_colored * alpha
    superimposed = np.clip(superimposed * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed)


def _apply_jet_colormap(gray: np.ndarray) -> np.ndarray:
    """Apply Jet-like colormap (blue -> cyan -> green -> yellow -> red)."""
    r = np.clip(1.5 - np.abs(4.0 * gray - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * gray - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * gray - 1.0), 0, 1)

    return np.stack([r, g, b], axis=-1)
