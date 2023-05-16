from __future__ import annotations
from collections import deque
from functools import cmp_to_key
from typing import Optional


class DV:
    """
    Decision variable (DV) in the CSP. Each cell in the sudoku grid has a
    unique DV associated with it.
    """

    def __init__(self, r: int, c: int, domain: set[int]) -> None:
        """
        Args:
            r (int): Row in sudoku grid
            c (int): Col in sudoku grid
            domain (set[int]): Set of feasible values
        """
        self.r = r
        self.c = c
        self.domain = domain

        # Set of DVs that share a constraint with the given DV i.e. DVs in the
        # same row, column, or 3x3 box as the given DV.
        self.neighbors: set[DV] = set()

    @staticmethod
    def cmp(dv1: DV, dv2: DV) -> int:
        """
        Compares `dv1` and `dv2` to determine which has higher priority of
        being guessed in the backtracking algorithm. The comparison is made
        using the minimum-remaining-values (MRV) and degree heuristics.

        Returns:
            int: 1 if `dv1` has higher priority, 0 if both have equal priority,
            and -1 if `dv2` has higher priority
        """
        dv1_priority: tuple[int, int] = (
            -len(dv1.domain),
            sum(1 for dv in dv1.neighbors if len(dv.domain) > 1),
        )
        dv2_priority: tuple[int, int] = (
            -len(dv2.domain),
            sum(1 for dv in dv2.neighbors if len(dv.domain) > 1),
        )

        if dv1_priority > dv2_priority:
            return 1
        elif dv1_priority == dv2_priority:
            return 0
        else:
            return -1


class SudokuCSP:
    """
    Implementation of a backtracking algorithm to solve a 9x9 sudoku when
    modelled as a constraint satisfaction problem (CSP). It is assumed that a
    solution exists and is unique. See for reference Chapter 6 of Russell,
    S. J., Norvig, P., & Davis, E. (2010). Artificial intelligence: a modern
    approach. 3rd ed. Upper Saddle River, NJ: Prentice Hall.
    """

    def __init__(self, grid: list[list[int]]) -> None:
        """
        Args:
            `grid` (list[list[int]]): Partially filled 9x9 sudoku grid; 0s
            indicate unfilled cells
        """
        self.guesses: int = 0  # No. of guesses made by the algorithm

        self.grid: list[list[DV]] = []
        self.unassigned_DVs: list[DV] = []
        for r in range(9):
            self.grid.append([])

            for c in range(9):
                if grid[r][c] != 0:  # Cell is assigned
                    self.grid[-1].append(DV(r, c, {grid[r][c]}))
                else:  # Cell is unassigned
                    self.grid[-1].append(DV(r, c, {i + 1 for i in range(9)}))
                    self.unassigned_DVs.append(self.grid[r][c])

        # Assign neighbors to the DVs
        for r in range(9):
            for c in range(9):
                for k in range(9):
                    # DVs in the same row as the given DV
                    if c != k:
                        self.grid[r][c].neighbors.add(self.grid[r][k])

                    # DVs in the same column as the given DV
                    if r != k:
                        self.grid[r][c].neighbors.add(self.grid[k][c])

                    # DVs in the same 3x3 block as the given DV
                    rb: int = r - (r % 3)
                    cb: int = c - (c % 3)
                    for r1 in range(3):
                        for c1 in range(3):
                            if rb + r1 != r or cb + c1 != c:
                                self.grid[r][c].neighbors.add(
                                    self.grid[rb + r1][cb + c1]
                                )

        # Propagate constraints using AC-3
        queue: deque[tuple[DV, DV]] = deque()
        for r in range(9):
            for c in range(9):
                for neighbor in self.grid[r][c].neighbors:
                    queue.append((self.grid[r][c], neighbor))
        self._ac3(queue)

    def _ac3(self, queue: deque[tuple[DV, DV]]) -> Optional[list[tuple[DV, int]]]:
        """
        Implementation of the arc-consistency (AC-3) algorithm.

        Args:
            queue (deque[tuple[DV, DV]]): Queue of arcs

        Returns:
            Optional[list[tuple[DV, int]]]: Returns list of inferences if no
            inconsistency is found or None otherwise. An inference is a tuple
            with first element containing a DV and second element containing the
            value removed from the DV's domain.
        """
        inferences: list[tuple[DV, int]] = []

        while 0 < len(queue):
            xi, xj = queue.popleft()
            xj_value = next(iter(xj.domain))

            # Revise xi's domain if xj is already assigned and xi's domain
            # contains xj's value
            if len(xj.domain) == 1 and xj_value in xi.domain:
                xi.domain.remove(xj_value)
                inferences.append((xi, xj_value))

                # Inconsistency if xi has an empty domain. Remove inferences
                # and return None
                if len(xi.domain) == 0:
                    for inference in inferences:
                        inference[0].domain.add(inference[1])
                    return None

                for xk in xi.neighbors:
                    if xk != xj:
                        queue.append((xk, xi))

        self.unassigned_DVs.sort(key=cmp_to_key(DV.cmp))

        return inferences

    def _mac(self, dv: DV) -> Optional[list[tuple[DV, int]]]:
        """
        Implementation of the maintaining arc consistency (MAC) algorithm.

        Args:
            dv (DV): DV whose value has been assigned

        Returns:
            Optional[list[tuple[DV, int]]]: Output of _ac3()
        """
        return self._ac3(
            deque(
                [
                    (neighbor, dv)
                    for neighbor in dv.neighbors
                    if len(neighbor.domain) > 1
                ]
            )
        )

    def _lcv(self, dv: DV) -> list[int]:
        """
        Returns the values in `dv`'s domain as a list ordered in decreasing
        order based on the least-constraining-value heuristic.
        """
        return sorted(
            dv.domain,
            key=lambda v: sum(v in neighbor.domain for neighbor in dv.neighbors),
            reverse=True,
        )

    def _backtrack(self) -> bool:
        """
        Solves the sudoku via backtracking.

        Returns:
            bool: True if the sudoku is solved and False otherwise
        """
        if len(self.unassigned_DVs) == 0:  # Sudoku is solved
            return True

        # Select DV to be assigned
        dv: DV = self.unassigned_DVs.pop()
        old_domain: set[int] = dv.domain

        for value in self._lcv(dv):
            self.guesses += 1

            # Guess DV's value
            dv.domain = set([value])

            inferences: Optional[list[tuple[DV, int]]] = self._mac(dv)
            if isinstance(inferences, list):  # No inconsistency found
                result = self._backtrack()
                if result:  # Sudoku is solved
                    return result

                # Remove inferences
                for inference in inferences:
                    inference[0].domain.add(inference[1])

        # Reset DV
        dv.domain = old_domain
        self.unassigned_DVs.append(dv)

        return False  # Sudoku cannot be solved

    def solve(self) -> tuple[int, list[list[int]]]:
        """
        Solves the sudoku.

        Returns:
            tuple[int, list[list[int]]]: A tuple with first element containing
            the no. of guesses made by the algorithm, and second element
            containing the solved grid
        """
        self._backtrack()

        grid: list[list[int]] = [[0 for c in range(9)] for r in range(9)]
        for r in range(9):
            for c in range(9):
                grid[r][c] = next(iter(self.grid[r][c].domain))

        return self.guesses, grid
