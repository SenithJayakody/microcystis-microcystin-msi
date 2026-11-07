import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# 200x200 ROI [y1, y2, x1, x2], centered at (570, 570).
transmittance_region = [470, 670, 470, 670]
block_size = 20

dayToCount = {
    1: 0,
    2: 26.5,
    3: 45.5,
    4: 56.7,
    5: 83.9,
    6: 97.3,
    7: 144.2,
    8: 147.1,
    9: 163.5,
    10: 170.4,
    11: 190.4,
    12: 19.4,
    13: 44.4,
    15: 68.4,
    17: 131.1,
    19: 158.1,
    21: 181.7,
}

count_to_tox = {
    0.0: 0.0,
    19.4: 0.62,
    26.5: 1.26,
    44.4: 1.90,
    45.5: 2.19,
    56.7: 2.82,
    68.4: 3.11,
    83.9: 3.54,
    97.3: 4.21,
    131.1: 4.94,
    144.2: 5.46,
    147.1: 5.84,
    158.1: 6.29,
    163.5: 6.53,
    170.4: 6.76,
    181.7: 6.98,
    190.4: 7.56,
}


def toxicity_from_count(count_val: float) -> float:
    """
    Returns toxicity using the existing measured mapping.
    Uses rounding to avoid floating-point mismatches.
    """
    key = round(float(count_val), 1)

    if key not in count_to_tox:
        raise KeyError(
            f"No toxicity value for cell count={key}. "
            f"Available counts: {sorted(count_to_tox.keys())}"
        )

    return float(count_to_tox[key])


wavelengths = [
    "365",
    "405",
    "473",
    "530",
    "621",
    "660",
    "735",
    "770",
    "830",
    "850",
    "890",
    "940",
]

def create_csv(MainPath, output_path, count_column):
    """Run the CSV generation for one split."""

    subdirList = [
        d for d in os.listdir(MainPath) if os.path.isdir(os.path.join(MainPath, d))
    ]

    trans_list = [d for d in subdirList if d.startswith("T")]

    if not trans_list:
        raise ValueError(f"No transmittance (T*) sample folders found in {MainPath}")

    dataAll_array = np.array([])
    kk = 0

    for subdir_1 in trans_list:

        Path_1 = os.path.join(MainPath, subdir_1)
        Path_1 += "/"

        print(Path_1)

        data = np.zeros((100, len(wavelengths) + 2))

        # Extract sample information
        parts = subdir_1.split("_")

        level = parts[1]
        level = dayToCount[int(level)]

        toxicity = toxicity_from_count(level)

        subSample = parts[3]

        if subSample == "A":
            subSampleNum = 1
        elif subSample == "B":
            subSampleNum = 2
        elif subSample == "C":
            subSampleNum = 3
        else:
            subSampleNum = 4

        # Main cropping coordinates
        y1, y2, x1, x2 = transmittance_region

        # Read dark image
        img_name = os.path.join(Path_1, "000nm.png")

        base_image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        # Process each wavelength
        for i in range(len(wavelengths)):

            img_name = os.path.join(Path_1, f"{wavelengths[i]}nm.png")

            image1 = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

            # Step 1: Dark-current subtraction
            image = cv2.subtract(image1, base_image)

            # Step 2: Crop the 200x200 region
            crop_img1 = image[y1:y2, x1:x2]

            # Step 3: Bilateral filtering AFTER cropping
            corrected_img1 = cv2.bilateralFilter(
                crop_img1, d=9, sigmaColor=75, sigmaSpace=75
            )

            # Step 4: Extract 100 non-overlapping 20x20 blocks
            t = 0

            for k in range(10):
                for j in range(10):

                    a = block_size * k
                    b = block_size * j

                    avg_mat = corrected_img1[b : b + block_size, a : a + block_size]

                    avg = np.mean(avg_mat)

                    data[t, i] = avg
                    t += 1

        # Clip intensity values
        data = np.clip(data, 0, 255)

        # Add cell count and toxicity labels
        for t in range(data.shape[0]):
            data[t, len(wavelengths)] = level
            data[t, len(wavelengths) + 1] = toxicity

        # Combine sample data
        if kk == 0:
            dataAll_array = data
            kk = 1
        else:
            dataAll_array = np.vstack((dataAll_array, data))

    # Create DataFrame and save CSV
    column_names = [f"{wl}_T" for wl in wavelengths] + [count_column, "Toxicity(ug/L)"]

    df = pd.DataFrame(dataAll_array, columns=column_names)

    varName = output_path

    df.to_csv(varName, index=False, float_format="%.2f")

    return df


def main():

    root = Path(__file__).resolve().parent
    data_dir = root / "Data"
    output_dir = root / "Datafiles"

    for split in ("Train", "Validation"):
        if not (data_dir / split).is_dir():
            raise FileNotFoundError(f"Expected image folder: {data_dir / split}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split, count_column in (
        ("Train", "Count(x 10^5)"),
        ("Validation", "Count(x 10^5)"),
    ):

        output_path = output_dir / f"dataAll_Water_{split}.csv"

        df = create_csv(data_dir / split, output_path, count_column)

        print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
