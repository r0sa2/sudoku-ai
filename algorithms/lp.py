from pulp import LpProblem, LpVariable, lpSum, PULP_CBC_CMD, value


class SudokuLP:
    """
    Implementation of a solver for a 9x9 Sudoku when modelled as a linear
    program. It is assumed that a solution exists and is unique. See for
    reference https://coin-or.github.io/pulp/CaseStudies/a_sudoku_problem.html.
    """

    def __init__(self, grid: list[list[int]]) -> None:
        """
        Args:
            `grid` (list[list[int]]): Partially filled 9x9 sudoku grid; 0s
            indicate unfilled cells
        """
        # List of 3x3 boxes, with each sublist containing the (row, col)
        # indices of each cell in the respective box
        self.boxes: list[list[tuple[int, int]]] = [
            [(3 * rb + r, 3 * rc + c) for r in range(3) for c in range(3)]
            for rb in range(3)
            for rc in range(3)
        ]

        # Create LP problem
        self.prob: LpProblem = LpProblem("Sudoku")

        # Create decision variables
        self.dvs: dict = LpVariable.dicts(
            "DV", (range(1, 10), range(9), range(9)), cat="Binary"
        )

        # Create single value per cell constraints
        for r in range(9):
            for c in range(9):
                self.prob += lpSum([self.dvs[v][r][c] for v in range(1, 10)]) == 1
        # Create row, column, and box constraints
        for v in range(1, 10):
            for r in range(9):
                self.prob += lpSum([self.dvs[v][r][c] for c in range(9)]) == 1
            for c in range(9):
                self.prob += lpSum([self.dvs[v][r][c] for r in range(9)]) == 1
            for b in self.boxes:
                self.prob += lpSum([self.dvs[v][r][c] for (r, c) in b]) == 1
        # Create filled cell constraints
        for r in range(9):
            for c in range(9):
                if grid[r][c] != 0:
                    self.prob += self.dvs[grid[r][c]][r][c] == 1

    def solve(self) -> list[list[int]]:
        """
        Solves the sudoku.

        Returns:
            list[list[int]]: Solved grid
        """
        self.prob.solve(PULP_CBC_CMD(msg=False))

        solved_grid: list[list[int]] = [[0 for c in range(9)] for r in range(9)]
        for r in range(9):
            for c in range(9):
                for v in range(1, 10):
                    if value(self.dvs[v][r][c]) == 1:
                        solved_grid[r][c] = v
                        break

        return solved_grid
