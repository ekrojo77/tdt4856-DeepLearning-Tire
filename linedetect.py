import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from skimage import feature
from PIL import Image
import cv2


def linedetect(from_folder, to_folder, sigma=1):
    for filename in tqdm(os.listdir(from_folder)):
        if filename.endswith(".jpg"):
            I_rgb = cv2.imread(from_folder + filename)
            I_gray = I_rgb[:, :, 0]  # Tar ut r fra rgb
            I_blur = cv2.GaussianBlur(
                I_gray, (5, 5), cv2.BORDER_DEFAULT
            )  # Støyfjerning

            h, w = I_blur.shape
            # Juster linjedeteksjkons-koeffisient basert på bildeoppløsning
            if h < 200 or w < 200:
                sigma = 0.50
            elif h > 800 or w > 800:
                sigma = 1.25

            I_lines = feature.canny(I_blur, sigma=sigma)
            # r to rgb format
            I_rgb_lines = np.zeros((h, w, 3))
            I_rgb_lines[:, :, 0] = I_lines
            I_rgb_lines[:, :, 1] = I_lines
            I_rgb_lines[:, :, 2] = I_lines

            plt.imshow(I_rgb_lines, cmap="gray")
            plt.savefig(f"{to_folder}Out{filename}.png")

            I_png = Image.open(f"{to_folder}Out{filename}.png")
            I_jpg = I_png.convert("RGB")
            I_jpg.save(f"{to_folder}Out{filename}", "JPEG")

        else:
            continue
