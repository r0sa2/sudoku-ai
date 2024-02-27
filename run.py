import argparse

import cv2

from ar_sudoku_solver.ar_sudoku_solver import ARSudokuSolver


if __name__ == "__main__":
    # Configure video capture device
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--device", type=str, help="Video capture device index"
    )
    args = parser.parse_args()
    capture = cv2.VideoCapture(int(args.device) if args.device else 0)

    # Instantiate solver
    solver = ARSudokuSolver()

    # Main loop. For each video frame, solve the frame and display the results
    while True:
        _, image = capture.read()

        cv2.imshow("AR Sudoku Solver", solver.solve_video(image))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
