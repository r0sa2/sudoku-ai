from copy import deepcopy


class SudokuBT:
    """
    Implementation of a simple backtracking algorithm to solve a 9x9 sudoku.
    It is assumed that a solution exists and is unique.
    """

    def __init__(self, grid: list[list[int]]) -> None:
        """
        Args:
            `grid` (list[list[int]]): Partially filled 9x9 sudoku grid; 0s
            indicate unfilled cells
        """
        self.guesses: int = 0  # No. of guesses made by the algorithm
        self.grid = deepcopy(grid)
        # Row, column and 3x3 box encodings of filled cells
        self.encodings: set[str] = set()
        for r in range(9):
            for c in range(9):
                if self.grid[r][c] != 0:
                    self.encodings.update(self._get_encodings(r, c, self.grid[r][c]))

    def _get_encodings(self, r: int, c: int, n: int) -> set[str]:
        """
        Returns a set of row, column, and box encodings for the case wherein the
        no. `n` is added to the grid at the cell with row `r` and column `c`.
        """
        return {f"{r}({n})", f"({n}){c}", f"{r // 3}({n}){c // 3}"}

    def _backtrack(self, r: int, c: int) -> bool:
        """
        Solves the sudoku via backtracking. Cells are filled in left-to-right
        and top-to-bottom order starting at the cell with row `r` and column `c`.
        """
        if r == 9 and c == 0:  # Sudoku is solved
            return True
        elif self.grid[r][c] != 0:  # Current cell is already filled
            return self._backtrack(r, c + 1) if c < 8 else self._backtrack(r + 1, 0)
        else:
            for n in range(1, 10):
                encodings: set[str] = self._get_encodings(r, c, n)

                if not self.encodings.isdisjoint(encodings):
                    continue  # Assignment isn't valid

                self.guesses += 1
                self.grid[r][c] = n
                self.encodings.update(encodings)

                if self._backtrack(r, c + 1) if c < 8 else self._backtrack(r + 1, 0):
                    return True

                self.grid[r][c] = 0
                self.encodings.difference_update(encodings)

            return False

    def solve(self) -> tuple[int, list[list[int]]]:
        """
        Solves the sudoku.

        Returns:
            tuple[int, list[list[int]]]: A tuple with first element containing
            the no. of guesses made by the algorithm, and second element
            containing the solved grid
        """
        self._backtrack(0, 0)
        return self.guesses, deepcopy(self.grid)
