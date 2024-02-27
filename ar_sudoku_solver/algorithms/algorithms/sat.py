from itertools import product

import pycosat


class SudokuSAT:
    """
    Implementation of a solver for a 9x9 Sudoku when modelled as a boolean
    satisfiability problem. It is assumed that a solution exists and is unique.
    See for reference https://www.lri.fr/~conchon/mpri/weber.pdf and
    https://github.com/conda/pycosat/blob/main/examples/sudoku.py.
    """

    def __init__(self, grid: list[list[int]]) -> None:
        """
        Args:
            `grid` (list[list[int]]): Partially filled 9x9 sudoku grid; 0s
            indicate unfilled cells
        """
        self.clauses: list[list[int]] = []
        for r, c in product(range(1, 10), range(1, 10)):
            # Create clauses to ensure that each cell contains at least one
            # value
            self.clauses.append(
                [self._get_var_num(r, c, v) for v in range(1, 10)]
            )
            # Create clauses to ensure that each cell does not contain two
            # distinct values
            for v in range(1, 10):
                for vp in range(v + 1, 10):
                    self.clauses.append(
                        [
                            -self._get_var_num(r, c, v),
                            -self._get_var_num(r, c, vp),
                        ]
                    )
        # Create clauses to ensure that each row (column) contains distinct
        # values
        for i in range(1, 10):
            self.clauses.extend(
                self._get_distinct_value_clauses(
                    [(i, j) for j in range(1, 10)]
                )
            )
            self.clauses.extend(
                self._get_distinct_value_clauses(
                    [(j, i) for j in range(1, 10)]
                )
            )
        # Create clauses to ensure that each 3x3 box contains distinct values
        for r, c in product([1, 4, 7], [1, 4, 7]):
            self.clauses.extend(
                self._get_distinct_value_clauses(
                    [(r + i % 3, c + i // 3) for i in range(9)]
                )
            )
        # Create clauses to account for given values
        for r, c in product(range(1, 10), range(1, 10)):
            if grid[r - 1][c - 1] != 0:
                self.clauses.append(
                    [self._get_var_num(r, c, grid[r - 1][c - 1])]
                )

    def _get_var_num(self, r: int, c: int, v: int) -> int:
        """
        Returns the no. of the variable corresponding to row `r`, column `c`,
        and value `v`.
        """
        return 81 * (r - 1) + 9 * (c - 1) + v

    def _get_distinct_value_clauses(
        self, cells: list[tuple[int, int]]
    ) -> list[list[int]]:
        """
        Returns clauses to ensure that the cells in `cells` contain distinct
        values.
        """
        clauses: list[list[int]] = []
        for i, xi in enumerate(cells):
            for j, xj in enumerate(cells):
                if i < j:
                    for v in range(1, 10):
                        clauses.append(
                            [
                                -self._get_var_num(xi[0], xi[1], v),
                                -self._get_var_num(xj[0], xj[1], v),
                            ]
                        )

        return clauses

    def solve(self) -> list[list[int]]:
        """
        Solves the sudoku.

        Returns:
            list[list[int]]: Solved grid
        """
        sol: list[int] = pycosat.solve(self.clauses)

        solved_grid = [[0 for c in range(9)] for r in range(9)]
        for r, c in product(range(1, 10), range(1, 10)):
            for v in range(1, 10):
                if self._get_var_num(r, c, v) in sol:
                    solved_grid[r - 1][c - 1] = v
                    break

        return solved_grid
