import urllib.request
import os
import sys
import zipfile

import random

import glob
import os
import shutil


from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps

from tqdm import tqdm

from linedetect import linedetect

IMAGE_FILE = "https://folk.ntnu.no/odinu/images-latest.zip"

"""
This script will download the latest set of images into the
"images-tmp/" directory. This folder will be deleted before the
script runs.

Images will be downloaded into the folders "images-tmp/Med" and
"images-tmp/Uten". Canny images will be added to the canny-images-tmp/
folder

usage:
    python image_download.py

To use canny edge detection:
    python image_download.py canny


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


def main():
    canny = False
    if len(sys.argv) > 1 and sys.argv[1] == "canny":
        canny = True
        DOWNLOAD_PATH = "./canny-images-tmp/"
    else:
        DOWNLOAD_PATH = "./images-tmp/"

    DOWNLOAD_LOCATION_ZIP = f"{DOWNLOAD_PATH}images.zip"
    try:
        shutil.rmtree(DOWNLOAD_PATH)
    except Exception:
        pass
    os.mkdir(DOWNLOAD_PATH)

    # Download zip with images
    download_url(IMAGE_FILE, DOWNLOAD_LOCATION_ZIP)

    # Extract zip
    with zipfile.ZipFile(DOWNLOAD_LOCATION_ZIP, "r") as zip_ref:
        zip_ref.extractall(DOWNLOAD_PATH)

    if canny:
        os.mkdir(DOWNLOAD_PATH + f"Med_Lines/")
        linedetect(DOWNLOAD_PATH + "Med/", DOWNLOAD_PATH + "Med_Lines/")
        os.mkdir(DOWNLOAD_PATH + f"Uten_Lines/")
        linedetect(DOWNLOAD_PATH + "Uten/", DOWNLOAD_PATH + "Uten_Lines/")

    os.mkdir(DOWNLOAD_PATH + f"Training_set/")
    os.mkdir(DOWNLOAD_PATH + f"Validation_set/")
    for Dir in ["Med", "Uten"]:
        if canny:
            Dir = f"{Dir}_Lines"
        output_path = DOWNLOAD_PATH + f"Training_set/{Dir}/"
        os.mkdir(output_path)
        counter = 0
        img_counter = 0
        a = 0

        # define the name of the directory to be created
        TestSetPath = DOWNLOAD_PATH + f"Validation_set/{Dir}/"
        os.mkdir(TestSetPath)

        for filename in glob.glob(f"{DOWNLOAD_PATH}{Dir}/*.jpg"):
            image = Image.open(filename)
            new_image = image.resize((224, 224))
            flipped_image = ImageOps.flip(new_image)
            mirrored_image = ImageOps.mirror(new_image)
            counter += 1
            print(f"\rCreating images in {output_path}: {counter}", end="")

            for i in [new_image, flipped_image, mirrored_image]:
                # Save original image
                i.save(
                    output_path + str(random.random()) + ".jpg", format="jpeg",
                )
                for y in range(1):
                    # endre dette for større/mindre random sample size
                    rotated_image = i.rotate(random.randint(1, 359), expand=True)
                    # endrer Brightness, tror ikke mørkere en 0.2 som startverdi er lurt
                    brightness_image = ImageEnhance.Brightness(rotated_image).enhance(
                        random.uniform(0.2, 0.9)
                    )
                    contrast_image = ImageEnhance.Contrast(rotated_image).enhance(
                        random.uniform(0.2, 0.9)
                    )
                    for new_img in [rotated_image, brightness_image, contrast_image]:
                        # Path of directory of images.
                        img_counter += 1
                        if img_counter % 10 == 0:
                            new_img.save(
                                TestSetPath + str(random.random()) + ".jpg",
                                format="jpeg",
                            )
                        else:
                            # Save the modified images
                            new_img.save(
                                output_path + str(random.random()) + ".jpg",
                                format="jpeg",
                            )
        print()


if __name__ == "__main__":
    main()
