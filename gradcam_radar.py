import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Computes a Grad-CAM activation heatmap for a given input image and CNN model.

    Args:
        img_array: Preprocessed input image tensor of shape (1, height, width, channels)
        model: Trained Keras/TensorFlow model
        last_conv_layer_name: String name of the final convolutional layer in the CNN
        pred_index: Index of the target class to explain (defaults to top prediction)

    Returns:
        A 2D numpy array representing the normalized heatmap (values between 0 and 1).
    """
    # 1. Create a model mapping inputs to last conv layer activations and predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Track gradients using GradientTape
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. Compute gradients of target class w.r.t the last conv layer outputs
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Global Average Pooling of gradients to compute importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply the conv activations by the channel weights and sum across channels
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU (keep only features contributing positively) and normalize
    heatmap = tf.maximum(heatmap, 0)
    if tf.reduce_max(heatmap) > 0:
        heatmap = heatmap / tf.reduce_max(heatmap)
        
    return heatmap.numpy()

def overlay_gradcam_on_radar(img_path, heatmap, output_path="radar_heatmap.png", alpha=0.4):
    """
    Overlays the computed Grad-CAM heatmap onto the original radar image.

    Args:
        img_path: Path to the original radar image file
        heatmap: Normalized Grad-CAM heatmap 2D array (values 0-1)
        output_path: Path where the superimposed output image will be saved
        alpha: Blending weight for the overlay (0.0 = only original, 1.0 = only heatmap)
    """
    # Load original image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image at path: {img_path}")

    # Scale heatmap to [0, 255] and resize to original image shape
    heatmap = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Apply Jet colormap (standard for meteorological/heat map themes)
    color_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Superimpose/blend the images
    superimposed_img = color_heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Save to disk
    cv2.imwrite(output_path, superimposed_img)

    # Display plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Radar Image")
    axes[0].axis("off")
    
    # Heatmap Blend
    axes[1].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Grad-CAM Heatmap Overlay")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()
    print(f"Heatmap successfully overlaid and saved to {output_path}")

if __name__ == "__main__":
    print("Grad-CAM TensorFlow utility module loaded.")
