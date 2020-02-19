import urllib.request
import os
import zipfile

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
    try:
        # Delete old files
        os.rmdir(DOWNLOAD_PATH)
    except Exception:
        pass
    try:
        # Create directory
        os.mkdir(DOWNLOAD_PATH)
    except Exception:
        pass

    # Download zip with images
    download_url(IMAGE_FILE, DOWNLOAD_LOCATION_ZIP)

    # Extract zip
    with zipfile.ZipFile(DOWNLOAD_LOCATION_ZIP, "r") as zip_ref:
        zip_ref.extractall(DOWNLOAD_PATH)
