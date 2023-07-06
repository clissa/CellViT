"""
This script prepares Fluorescent Neuronal Cells v2 datasets.

Author: Luca Clissa <clissa@bo.infn.it>
Created: 2023-07-06
License: Apache License 2.0
"""


from PIL import Image
import xml.etree.ElementTree as ET
from skimage import draw
import numpy as np
from pathlib import Path
from typing import Union
import argparse
from tqdm.auto import tqdm


RESIZE_SHAPE = 1024
def convert_monuseg(
    input_path: Union[Path, str], output_path: Union[Path, str]
) -> None:
    """Convert the MoNuSeg dataset to a new format (1000 -> 1024, tiff to png and xml to npy)

    Args:
        input_path (Union[Path, str]): Input dataset
        output_path (Union[Path, str]): Output path
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # testing and training
    parts = ["test", "trainval"]
    # parts = ["testing"]
    for part in parts:
        print(f"Prepare: {part}")
        input_path_part = input_path / part
        output_path_part = output_path / part
        output_path_part.mkdir(exist_ok=True, parents=True)
        (output_path_part / "images").mkdir(exist_ok=True, parents=True)
        (output_path_part / "labels").mkdir(exist_ok=True, parents=True)

        # images
        images = [f for f in sorted((input_path_part / "images").glob("*.png"))]
        for img_path in tqdm(images):
            loaded_image = Image.open(img_path)
            resized = loaded_image.resize(
                (RESIZE_SHAPE, RESIZE_SHAPE), resample=Image.Resampling.LANCZOS
            )
            new_img_path = output_path_part / "images" / f"{img_path.stem}.png"
            resized.save(new_img_path)
        # masks
        annotations = [f for f in sorted((input_path_part / "ground_truths/masks").glob("*.png"))]
        for annot_path in tqdm(annotations):
            # binary_mask = np.transpose(np.zeros((1000, 1000)))
            binary_mask = Image.open(annot_path).convert("L")
            # inst_image = Image.fromarray(binary_mask)
            resized_mask = np.array(
                binary_mask.resize((RESIZE_SHAPE, RESIZE_SHAPE), resample=Image.Resampling.NEAREST)
            )
            new_mask_path = output_path_part / "labels" / f"{annot_path.stem}.npy"
            np.save(new_mask_path, resized_mask)
    print("Finished")


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Convert the MoNuSeg dataset",
)
parser.add_argument(
    "--input_path",
    type=str,
    help="Input path of the original MoNuSeg dataset",
    required=True,
)
parser.add_argument(
    "--output_path",
    type=str,
    help="Output path to store the processed MoNuSeg dataset",
    required=True,
)

if __name__ == "__main__":
    opt = parser.parse_args()
    configuration = vars(opt)

    input_path = Path(configuration["input_path"])
    output_path = Path(configuration["output_path"])

    convert_monuseg(input_path=input_path, output_path=output_path)
