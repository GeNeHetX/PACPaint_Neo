import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import os

# Demerdez-vous pour installer openslide sur votre machine
OPENSLIDE_PATH = r"D:\DataManage\openslide-win64-20231011\bin"
if hasattr(os, "add_dll_directory"):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def convert_coord(img, slide, df_slide):
    h, w = img.shape[:2]
    w_slide, h_slide = slide.dimensions
    h_factor, w_factor = h / h_slide, w / w_slide
    df_slide["x_img"] = df_slide.x * w_factor * 2 * 224
    df_slide["y_img"] = df_slide.y * h_factor * 2 * 224
    return df_slide


def display_wsi_results(
    path_wsi: Path,
    pred_neo: pd.DataFrame,
    pred_comp: pd.DataFrame = None,
    thresh: float = 0.5,
) -> None:
    slide = openslide.OpenSlide(str(path_wsi))
    img_pil = slide.get_thumbnail((1000, 1000))
    img = np.array(img_pil)

    pred_neo.rename(columns={"pred": "tumour_pred"}, inplace=True)

    if pred_comp is not None:
        pred_df = pd.merge(pred_neo, pred_comp, on=["z", "x", "y"])
    else:
        pred_df = pred_neo.copy()

    pred_df = convert_coord(img, slide, pred_df)

    if pred_comp is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.flatten()
        ax1, ax2, ax3 = axes
        ax1.imshow(img)
        ax1.set_title("Original image")
        ax2.scatter(pred_df.x_img, pred_df.y_img, s=5, c=pred_df.tumour_pred, cmap="coolwarm")
        ax2.invert_yaxis()
        ax2.set_title("Tumour prediction")
        ax3.scatter(pred_df.x_img, pred_df.y_img, s=5, c=pred_df.tumour_pred > thresh, cmap="coolwarm")
        ax3.invert_yaxis()
        ax3.set_title("Tumour prediction with threshold")

    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        axes[0].imshow(img)
        axes[0].set_title("Original image")
        axes[1].scatter(pred_df.x_img, pred_df.y_img, s=5, c=pred_df.tumour_pred, cmap="coolwarm")
        axes[1].invert_yaxis()
        axes[1].set_title("Tumour prediction")
        axes[2].scatter(pred_df.x_img, pred_df.y_img, s=5, c=pred_df.Classic, cmap="coolwarm")
        axes[2].invert_yaxis()
        axes[2].set_title("Classic prediction")
        axes[3].scatter(pred_df.x_img, pred_df.y_img, s=5, c=pred_df.Basal, cmap="coolwarm")
        axes[3].invert_yaxis()
        axes[3].set_title("Basal prediction")

    # Both subplot same size
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()
