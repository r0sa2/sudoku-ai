import os
from pathlib import Path

import cv2
import pandas as pd


if __name__ == "__main__":
    dir_path = Path(__file__).parent.absolute()
    image_shape = (28, 28)
    data = []

    for d in list(map(str, range(10))):
        for filename in os.listdir(os.path.join(dir_path, d)):
            image = cv2.imread(
                os.path.join(dir_path, d, filename), cv2.IMREAD_GRAYSCALE
            )
            image = cv2.resize(image, dsize=image_shape)
            data.append([d] + image.reshape(-1).tolist())

    pd.DataFrame(
        data,
        columns=(
            ["label"] +
            [f"pixel{i}" for i in range(image_shape[0] * image_shape[1])]
        )
    ).to_csv(os.path.join(dir_path, "char74k.csv"), index=False)
