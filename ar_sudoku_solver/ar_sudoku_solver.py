from itertools import product
import os
from pathlib import Path

import cv2
import numpy as np
import scipy
import tensorflow as tf

from .solver import Solver


class ARSudokuSolver:
    """
    Implementation of an augmented-reality Sudoku solver for images and video.
    """

    def __init__(self):
        self.digit_classifier = tf.keras.models.load_model(os.path.join(
            Path(__file__).parent, "digit_classifier", "model", "char74k.h5"
        ))

        # Parameters for basic object tracking support for smoother video
        # performance. The algorithm used here is adapted from
        # https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
        self.cached_grids = {}  # Tracked grids
        self.next_grid_id = 0
        # Maximum no. of consecutive frames a grid is allowed to have
        # "disappeared" before being removed from the cache
        self.max_disappeared = 50
        # Maximum Euclidean distance between the corners of two grids for them
        # to be matched
        self.max_distance = 100

    def preprocess_image(self, image):
        """
        Preprocesses `image` prior to extracting grids or digits, returning the
        preprocessed image and a gray scale version of the image.
        """
        image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
        image_gray = cv2.normalize(
            image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        image = cv2.GaussianBlur(image_gray, ksize=(5, 5), sigmaX=0, sigmaY=0)
        image = cv2.adaptiveThreshold(
            image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=45,
            C=15
        )

        return image, image_gray

    def extract_grids(self, image):
        """
        Extracts grids from `image`, returning the corners of the extracted
        grids, transformed images (after performing a perspective transform),
        and inverse transformation matrices.
        """
        original_image = image

        # Preprocess image
        image, _ = self.preprocess_image(image)
        image = cv2.bitwise_not(image)
        image = cv2.morphologyEx(
            image, op=cv2.MORPH_CLOSE, kernel=np.ones((5, 5), dtype=np.uint8)
        )
        image = cv2.morphologyEx(
            image, op=cv2.MORPH_DILATE, kernel=np.ones((5, 5), dtype=np.uint8)
        )

        # Extract contours and sort in descending order of area
        contours, _ = cv2.findContours(
            image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if len(contours) == 0:
            return np.array([]), [], []

        # Process and filter out contours with area less than a threshold
        extracted_contours = []
        max_contour_area = cv2.contourArea(contours[0])
        for contour in contours:
            if cv2.contourArea(contour) < max_contour_area / 8:
                break
            extracted_contour = cv2.approxPolyDP(
                contour,
                epsilon=0.1 * cv2.arcLength(contour, closed=True),
                closed=True
            )
            if len(extracted_contour) == 4:
                extracted_contours.append(extracted_contour)

        if len(extracted_contours) == 0:
            return np.array([]), [], []

        # Extract corners from contours
        extracted_corners = np.zeros(
            (len(extracted_contours), 4, 2), dtype=np.float32
        )
        for i, contour in enumerate(extracted_contours):
            mean_x = np.mean(contour[:, :, 0])
            mean_y = np.mean(contour[:, :, 1])
            for point in contour:
                x, y = point[0]
                if mean_x < x:
                    if mean_y < y:
                        extracted_corners[i][2] = [x, y]
                    else:
                        extracted_corners[i][1] = [x, y]
                else:
                    if mean_y < y:
                        extracted_corners[i][3] = [x, y]
                    else:
                        extracted_corners[i][0] = [x, y]

        # Transform images based on corners and get inverse transform matrices
        extracted_corners_ = []
        transformed_images = []
        inverse_transform_matrices = []
        for i, src_corners in enumerate(extracted_corners):
            transform_matrix = cv2.getPerspectiveTransform(
                src_corners,
                np.array([
                    [0, 0], [449, 0], [449, 449], [0, 449]
                ], dtype=np.float32)
            )
            try:
                inverse_transform_matrix = np.linalg.inv(transform_matrix)
                extracted_corners_.append(src_corners)
                transformed_images.append(cv2.warpPerspective(
                    original_image, M=transform_matrix, dsize=(450, 450)
                ))
                inverse_transform_matrices.append(inverse_transform_matrix)
            except Exception:
                pass

        extracted_corners_ = np.array(extracted_corners_)
        if 0 < len(extracted_corners_):
            extracted_corners_ = extracted_corners_.reshape(
                extracted_corners_.shape[0], -1
            )

        return (
            extracted_corners_, transformed_images, inverse_transform_matrices
        )

    def extract_digits(self, image):
        """
        Extracts grid digits from `image`.
        """
        grid = None
        image_height, image_width = image.shape[:2]

        # Preprocess image
        image, image_gray = self.preprocess_image(image)

        # Extract contours
        contours, _ = cv2.findContours(
            image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )
        digit_images = []
        digit_locations = []

        for contour in contours:
            # Get the bounding rectangle for each contour
            x, y, w, h = cv2.boundingRect(contour)

            if 15 < h < 50 and 5 < w < 30 and 0.3 < (w / h) < 1.0:
                # Digit Isolation: Expand the region slightly to ensure the
                # entire digit is captured
                y1 = max(y - 2, 0)
                y2 = min(y + h + 2, image_height)
                x1 = max(x - 2, 0)
                x2 = min(x + w + 2, image_width)

                # Extract the digit region from the grayscale image
                digit_image = image_gray[y1:y2, x1:x2]

                # Threshold and resize the digit image
                _, digit_image = cv2.threshold(
                    digit_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                digit_image_resized = cv2.resize(
                    digit_image, (28, 28), interpolation=cv2.INTER_NEAREST
                ).reshape(28, 28, 1)

                digit_locations.append([y + h / 2, x + w / 2])
                digit_images.append(digit_image_resized)

        if 0 < len(digit_images):
            # Predict digits
            pred_probas = self.digit_classifier.predict(
                np.array(digit_images) / 255.0, verbose=0
            )
            preds = []
            num_digits_extracted = 0

            for pred_proba in pred_probas:
                idx = np.argmax(pred_proba)

                if 0.98 < pred_proba[idx] and idx <= 9:
                    preds.append(idx)
                    num_digits_extracted += 1
                else:
                    preds.append(0)

            # Check that there are at least 17 extracted digits for the Sudoku
            # to be uniquely solvable. Also check that the extracted digits
            # don't violate any of the Sudoku constraints
            if 17 <= num_digits_extracted:
                grid = [[0 for _ in range(9)] for _ in range(9)]
                encodings = set()

                for pred, location in zip(preds, digit_locations):
                    if 0 < pred:
                        y = int(9 * location[0] // image_height)
                        x = int(9 * location[1] // image_width)
                        grid[y][x] = pred
                        row_encoding = f"{y}({grid[y][x]})"
                        col_encoding = f"({grid[y][x]}){x}"
                        box_encoding = f"{y // 3}({grid[y][x]}){x // 3}"

                        if any([
                            row_encoding in encodings,
                            col_encoding in encodings,
                            box_encoding in encodings
                        ]):
                            return None

                        encodings.update([
                            row_encoding, col_encoding, box_encoding
                        ])

        return grid

    def generate(self, image, generation_args):
        """
        Generates an image with the solved digits added.
        """
        generated_image = np.zeros((*image.shape[:2], 3), dtype=np.uint8)

        # Add each solved grid to `generated_image`
        for (
            transformed_image, inverse_transform_matrix, grid, solved_grid
        ) in generation_args:
            text_mask = np.zeros_like(transformed_image)

            for y, x in product(range(9), range(9)):
                if grid[y][x] != 0:
                    continue

                text = str(solved_grid[y][x])
                text_width, text_height = cv2.getTextSize(
                    text,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1.3,
                    thickness=2
                )[0]
                org = (
                    int((x + 0.5) * text_mask.shape[1] / 9 - text_width / 2),
                    int((y + 0.5) * text_mask.shape[0] / 9 + text_height / 2)
                )
                cv2.putText(
                    text_mask,
                    text=text,
                    org=org,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1.3,
                    color=(0, 0, 255),
                    thickness=2
                )

            generated_image = cv2.add(
                generated_image,
                cv2.warpPerspective(
                    text_mask,
                    M=inverse_transform_matrix,
                    dsize=(image.shape[1], image.shape[0])
                )
            )

        # Integrate the generated image with `image`
        _, mask = cv2.threshold(
            cv2.cvtColor(generated_image, code=cv2.COLOR_BGR2GRAY),
            thresh=1,
            maxval=255,
            type=cv2.THRESH_BINARY
        )

        return cv2.add(
            cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask)),
            generated_image
        )

    def solve_image(self, image):
        """
        Solver for a single image.
        """
        # Extract grids
        _, transformed_images, inverse_transform_matrices =\
            self.extract_grids(image)

        # Extract digits
        grids = [
            self.extract_digits(transformed_image)
            for transformed_image in transformed_images
        ]

        # Solve grids
        solved_grids = [
            Solver(grid).solve() if grid is not None else None
            for grid in grids
        ]

        # Generate image with the solved digits added
        return self.generate(
            image,
            list(filter(lambda x: x[3] is not None, zip(
                transformed_images,
                inverse_transform_matrices,
                grids,
                solved_grids
            )))
        )

    def register(
        self, extracted_corners, transformed_image, inverse_transform_matrix
    ):
        """
        Add grid to the cache for tracking and return the grid ID.
        """
        # Extract digits
        grid = self.extract_digits(transformed_image)

        if grid is None:
            return None

        # Solve the grid
        solved_grid = Solver(grid).solve()

        if solved_grid is None:
            return None

        # Add the grid to the cache
        self.cached_grids[self.next_grid_id] = {
            "extracted_corners": extracted_corners,
            "transformed_image": transformed_image,
            "inverse_transform_matrix": inverse_transform_matrix,
            "grid": grid,
            "solved_grid": solved_grid,
            "disappeared": 0,
        }

        self.next_grid_id += 1

        return self.next_grid_id - 1

    def solve_video(self, image):
        """
        Solver for a video frame.
        """
        # Extract grids
        extracted_corners, transformed_images, inverse_transform_matrices =\
            self.extract_grids(image)

        if len(extracted_corners) == 0:
            # If there are no extracted grids, increment the `disappeared`
            # count of all cached grids and remove a cached grid if the count
            # exceeds `max_disappeared`. Finally, return the image unchanged.
            for grid_id in list(self.cached_grids.keys()):
                self.cached_grids[grid_id]["disappeared"] += 1
                if (
                    self.max_disappeared <
                    self.cached_grids[grid_id]["disappeared"]
                ):
                    del self.cached_grids[grid_id]

            return image

        # Arguments for the image generator
        generation_args = []

        if len(self.cached_grids) == 0:
            # If there are no cached grids, attempt to add all extracted grids
            # to the cache
            for i in range(len(extracted_corners)):
                grid_id = self.register(
                    extracted_corners[i],
                    transformed_images[i],
                    inverse_transform_matrices[i]
                )

                if grid_id is not None:
                    generation_args.append((
                        self.cached_grids[grid_id]["transformed_image"],
                        self.cached_grids[grid_id]["inverse_transform_matrix"],
                        self.cached_grids[grid_id]["grid"],
                        self.cached_grids[grid_id]["solved_grid"]
                    ))
        else:
            # Compute pair-wise distances between extracted and cached grids
            cached_grids_ids = list(self.cached_grids.keys())
            cached_extracted_corners = [
                self.cached_grids[grid_id]["extracted_corners"]
                for grid_id in cached_grids_ids
            ]
            D = scipy.spatial.distance.cdist(
                cached_extracted_corners, extracted_corners
            )

            # Reorder indices based on nearest distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                # Ignore grids that have already been matched
                if row in used_rows or col in used_cols:
                    continue

                # Ignore matches with distance greater than `max_distance`
                if self.max_distance < D[row][col]:
                    break

                # Update the data for a cached grid with that of the matched
                # extracted grid
                grid_id = cached_grids_ids[row]
                self.cached_grids[grid_id]["extracted_corners"] = (
                    extracted_corners[col]
                )
                self.cached_grids[grid_id]["transformed_image"] = (
                    transformed_images[col]
                )
                self.cached_grids[grid_id]["inverse_transform_matrix"] = (
                    inverse_transform_matrices[col]
                )
                self.cached_grids[grid_id]["disappeared"] = 0
                generation_args.append((
                    self.cached_grids[grid_id]["transformed_image"],
                    self.cached_grids[grid_id]["inverse_transform_matrix"],
                    self.cached_grids[grid_id]["grid"],
                    self.cached_grids[grid_id]["solved_grid"]
                ))
                used_rows.add(row)
                used_cols.add(col)

            # Get unmatched cached and extracted grids
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Increment the `disappeared` count for unmatched cached grids
            # and remove then from the cache if the count exceeds
            # `max_disappeared`
            for row in unused_rows:
                grid_id = cached_grids_ids[row]
                self.cached_grids[grid_id]["disappeared"] += 1
                if (
                    self.max_disappeared <
                    self.cached_grids[grid_id]["disappeared"]
                ):
                    del self.cached_grids[grid_id]

            # Attempt to add unmatched extracted grids to the cache
            for col in unused_cols:
                grid_id = self.register(
                    extracted_corners[col],
                    transformed_images[col],
                    inverse_transform_matrices[col]
                )

                if grid_id is not None:
                    generation_args.append((
                        self.cached_grids[grid_id]["transformed_image"],
                        self.cached_grids[grid_id]["inverse_transform_matrix"],
                        self.cached_grids[grid_id]["grid"],
                        self.cached_grids[grid_id]["solved_grid"]
                    ))

        # Generate and return image for display
        return (
            self.generate(image, generation_args)
            if 0 < len(generation_args)
            else image
        )
