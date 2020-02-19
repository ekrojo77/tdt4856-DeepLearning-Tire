import urllib.request
import os
import zipfile

import random

import glob
import os
import shutil


from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps

from tqdm import tqdm

IMAGE_FILE = "http://folk.ntnu.no/odinu/images-latest.zip"
DOWNLOAD_PATH = "./images-tmp/"
DOWNLOAD_LOCATION_ZIP = "./images-tmp/images.zip"

"""
This script will download the latest set of images into the
"images-tmp/" directory. This folder will be deleted before the
script runs.

Images will be downloaded into the folders "images-tmp/Med" and
"images-tmp/Uten".

usage: python image_download.py

"""


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


if __name__ == "__main__":
    shutil.rmtree(DOWNLOAD_PATH)
    os.mkdir(DOWNLOAD_PATH)

    # Download zip with images
    download_url(IMAGE_FILE, DOWNLOAD_LOCATION_ZIP)

    # Extract zip
    with zipfile.ZipFile(DOWNLOAD_LOCATION_ZIP, "r") as zip_ref:
        zip_ref.extractall(DOWNLOAD_PATH)

    for Dir in ["Med", "Uten"]:
        output_path = DOWNLOAD_PATH + f"{Dir}_All/"
        os.mkdir(output_path)
        counter = 0
        for filename in glob.glob(f"./images-tmp/{Dir}/*.jpg"):
            image = Image.open(filename)
            new_image = image.resize((224, 224))
            flipped_image = ImageOps.flip(new_image)
            mirrored_image = ImageOps.mirror(new_image)
            print(f"\rCreating images in {output_path}: {counter}", end="")
            counter += 1
            for i in [new_image, flipped_image, mirrored_image]:
                # Save original image
                i.save(
                    output_path + str(random.random()) + ".jpg", format="jpeg",
                )
                for y in range(3):
                    # endre dette for større random sample size
                    rotated_image = i.rotate(random.randint(1, 359), expand=True)
                    brightness_image = ImageEnhance.Brightness(rotated_image).enhance(
                        random.uniform(0.2, 0.9)
                    )
                    # endrer Brightness, tror ikke mørkere en 0.2 som startverdi er lurt
                    contrast_image = ImageEnhance.Contrast(rotated_image).enhance(
                        random.uniform(0.2, 0.9)
                    )
                    for new_img in [rotated_image, brightness_image, contrast_image]:
                        # Save the modified images
                        new_img.save(
                            output_path + str(random.random()) + ".jpg", format="jpeg",
                        )
        print()
