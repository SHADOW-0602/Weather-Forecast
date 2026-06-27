"""
Produces Gradient-weighted Class Activation Maps (Grad-CAM)
for the CNN component of the AeroClim multimodal pipeline.

References:
  Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization. ICCV. https://arxiv.org/abs/1610.02391
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings/infos
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


#  CORE GRAD-CAM FUNCTION

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    pred_index: int | None = None,
) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array])
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, pred_index]

    # Gradients of class score w.r.t. last conv layer
    grads = tape.gradient(class_score, conv_outputs)

    # Global-average-pool the gradients -> per-channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalise to [0, 1]
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


#  RADAR OVERLAY HELPER

def overlay_gradcam_on_radar(
    img_path: str,
    heatmap: np.ndarray,
    output_path: str = "radar_heatmap.png",
    alpha: float = 0.40,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load radar image: {img_path}")

    H, W = img.shape[:2]

    # Scale & resize heatmap
    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_u8, (W, H))

    # Apply meteorological colormap
    color_heatmap = cv2.applyColorMap(heatmap_resized, colormap)

    # Alpha blend
    blended = np.clip(
        color_heatmap * alpha + img * (1.0 - alpha), 0, 255
    ).astype(np.uint8)

    cv2.imwrite(output_path, blended)

    # Side-by-side matplotlib display
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0f0c1b")

    titles = ["Original Radar", "Grad-CAM Heatmap", "Blended Overlay"]
    images_rgb = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(blended, cv2.COLOR_BGR2RGB),
    ]

    for ax, title, im in zip(axes, titles, images_rgb):
        ax.imshow(im)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.axis("off")

    # Colorbar legend
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("CNN Activation Intensity", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    plt.savefig(
        output_path.replace(".png", "_figure.png"),
        dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.show()
    print(f"[GradCAM] Saved heatmap overlay -> {output_path}")
    return blended


#  SYNTHETIC HEATMAP GENERATOR (used when no real CNN is available)

def generate_synthetic_heatmap(
    height: int = 256,
    width: int = 256,
    n_storm_cells: int = 2,
    seed: int = 42,
) -> np.ndarray:
    np.random.seed(seed)
    heatmap = np.zeros((height, width), dtype=np.float32)

    for _ in range(n_storm_cells):
        # Random storm core centre
        cy = np.random.randint(height // 4, 3 * height // 4)
        cx = np.random.randint(width // 4,  3 * width // 4)
        radius = np.random.randint(height // 8, height // 4)
        intensity = np.random.uniform(0.6, 1.0)

        # Gaussian blob
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        blob = intensity * np.exp(-(dist ** 2) / (2 * (radius / 2) ** 2))
        heatmap = np.maximum(heatmap, blob)

    # Light background noise
    heatmap += np.random.uniform(0, 0.08, (height, width))
    heatmap = np.clip(heatmap, 0.0, 1.0)
    return heatmap


if __name__ == "__main__":
    print("[GradCAM] Module loaded - AeroClim Grad-CAM utility.")
    print("  Synthetic heatmap shape:", generate_synthetic_heatmap().shape)
