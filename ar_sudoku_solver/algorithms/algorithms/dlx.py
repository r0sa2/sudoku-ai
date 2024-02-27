from itertools import product
from typing import Optional


class Node:
    """
    Node in the DLX network. Nodes can be one of three types :-
    1. Root node - has attributes L, R
    2. Header node - has attributes L, R, U, D, S
    3. Non-header node - has attributes L, R, U, D, C, N
    """

    def __init__(self) -> None:
        self.L: Node  # Reference to left neighbor
        self.R: Node  # Reference to right neighbor
        self.U: Optional[Node]  # Reference to up neighbor
        self.D: Optional[Node]  # Reference to down neighbor
        self.C: Optional[Node]  # Reference to associated header node
        self.S = 0  # No. of non-header nodes in the column

        # Non-header nodes in the same row are associated with one possibility
        # [row, col, val] i.e. adding val to (row, col) in the sudoku grid
        self.N: Optional[tuple[int, int, int]]  # [row, col, val]


class SudokuDLX:
    """
    Implementation of Donald Knuth's Algorithm X to solve a 9x9 sudoku when
    modelled as an exact cover problem. Algorithm X is implemented using the
    dancing links technique (DLX). See for reference
    https://arxiv.org/pdf/cs/0011047.pdf.
    """

    def __init__(self, grid: list[list[int]]) -> None:
        """
        Args:
            `grid` (list[list[int]]): Partially filled 9x9 sudoku grid; 0s
            indicate unfilled cells
        """
        self.guesses = 0  # No. of guesses made by the algorithm
        self.assigned_count = 0  # No. of assigned cells
        self.O: list[Optional[Node]] = [None for i in range(81)]  # Solution
        self.root = Node()
        self.network_height = 730
        self.network_width = 324

        # Nodes in the network other than the root node. First row contains
        # the header nodes and all remaining rows contain non-header nodes
        self.network: list[list[Optional[Node]]] = [
            [None for c in range(self.network_width)]
            for r in range(self.network_height)
        ]

        # Add header nodes
        for c in range(self.network_width):
            self.network[0][c] = Node()

        # Assign L and R attributes of the root and header nodes
        self.root.L = self.network[0][self.network_width - 1]
        self.root.R = self.network[0][0]
        self.network[0][0].L = self.root
        self.network[0][0].R = self.network[0][1]
        self.network[0][self.network_width - 1].L = self.network[0][
            self.network_width - 2
        ]
        self.network[0][self.network_width - 1].R = self.root
        for c in range(1, self.network_width - 1):
            self.network[0][c].L = self.network[0][c - 1]
            self.network[0][c].R = self.network[0][c + 1]

        offset = 81
        for r, c in product(range(9), range(9)):
            # 3x3 block associated with the given (r, c)
            b = (r // 3) * 3 + (c // 3)

            for v in range(9):
                r1 = r * 81 + c * 9 + v + 1
                c0 = r * 9 + c
                c1 = offset + r * 9 + v
                c2 = offset * 2 + c * 9 + v
                c3 = offset * 3 + b * 9 + v

                # Add non-header nodes
                self.network[r1][c0] = Node()  # Row-Column constraint
                self.network[r1][c1] = Node()  # Row-Number constraint
                self.network[r1][c2] = Node()  # Column-Number constraint
                self.network[r1][c3] = Node()  # Box-Number constraint

                # Assign L, R, and C attributes of the non-header nodes
                self.network[r1][c0].L = self.network[r1][c3]
                self.network[r1][c0].R = self.network[r1][c1]
                self.network[r1][c1].L = self.network[r1][c0]
                self.network[r1][c1].R = self.network[r1][c2]
                self.network[r1][c2].L = self.network[r1][c1]
                self.network[r1][c2].R = self.network[r1][c3]
                self.network[r1][c3].L = self.network[r1][c2]
                self.network[r1][c3].R = self.network[r1][c0]
                self.network[r1][c0].C = self.network[0][c0]
                self.network[r1][c1].C = self.network[0][c1]
                self.network[r1][c2].C = self.network[0][c2]
                self.network[r1][c3].C = self.network[0][c3]

                # Increment S attribute of the associated header nodes
                self.network[r1][c0].C.S += 1
                self.network[r1][c1].C.S += 1
                self.network[r1][c2].C.S += 1
                self.network[r1][c3].C.S += 1

                self.network[r1][c0].N = [r, c, v + 1]
                self.network[r1][c1].N = [r, c, v + 1]
                self.network[r1][c2].N = [r, c, v + 1]
                self.network[r1][c3].N = [r, c, v + 1]

                if grid[r][c] == v + 1:  # Cell is assigned
                    self.O[self.assigned_count] = self.network[r1][c0]
                    self.assigned_count += 1

        # Assign U and D attributes of the header and non-header nodes
        for c in range(self.network_width):
            header: Node = self.network[0][c]
            node = header

            for r in range(1, self.network_height):
                if self.network[r][c] is not None:
                    node.D = self.network[r][c]
                    self.network[r][c].U = node
                    node = self.network[r][c]

            header.U = node
            node.D = header

        # Remove rows associated with assigned cells
        for i in range(self.assigned_count):
            self._remove_row(self.O[i])

    def _remove_row(self, node: Node) -> None:
        """
        Removes rows associated with assigned cells.

        Args:
            `node` (Node): Non-header node in the row
        """
        # Determine header nodes associated with the row
        col_heads: list[Node] = [node.C]
        row_node = node.R

        while row_node != node:
            col_heads.append(row_node.C)
            row_node = row_node.R

        # Cover the column of each header node
        for col_head in col_heads:
            self._cover_col(col_head)

    def _select_col_head(self) -> Optional[Node]:
        """
        Returns:
            Optional[Node]: Header node of the next column to be covered or
            `None` if there are no remaining header nodes
        """
        if self.root.R == self.root:  # No remaining header nodes
            return None

        # Select the header node with minimum S value
        col_head = self.root
        min_col_head = col_head
        min_rows = self.network_height

        while col_head.R != self.root:
            col_head = col_head.R

            if col_head.S < min_rows:
                min_col_head = col_head
                min_rows = col_head.S

        return min_col_head

    def _cover_col(self, col_head: Node) -> None:
        """
        Covers the column.

        Args:
            `col_head` (Node): Header node of column to be covered
        """
        col_head.R.L = col_head.L
        col_head.L.R = col_head.R

        col_node: Node = col_head.D

        while col_node != col_head:
            row_node = col_node.R

            while row_node != col_node:
                row_node.D.U = row_node.U
                row_node.U.D = row_node.D
                row_node.C.S -= 1
                row_node = row_node.R

            col_node = col_node.D

    def _uncover_col(self, col_head: Node) -> None:
        """
        Uncovers the column.

        Args:
            `col_head` (Node): Header node of column to be uncovered
        """
        col_node: Node = col_head.U

        while col_node != col_head:
            row_node = col_node.L

            while row_node != col_node:
                row_node.C.S += 1
                row_node.D.U = row_node
                row_node.U.D = row_node
                row_node = row_node.L

            col_node = col_node.U

        col_head.R.L = col_head
        col_head.L.R = col_head

    def _search(self, k: int) -> bool:
        """
        Implemention of the DLX algorithm.

        Args:
            `k` (int): Index of solution array O to be assigned next

        Returns:
            bool: `True` if the sudoku is solved and `False` otherwise
        """
        # Select the header node of the column to be covered
        col_head: Optional[Node] = self._select_col_head()

        if col_head is None:  # Sudoku is solved
            return True

        # Cover the column
        self._cover_col(col_head)

        # Select the first row in the column
        col_node: Node = col_head.D

        while col_node != col_head:
            # Add the row to the solution
            self.O[k] = col_node
            self.guesses += 1

            # Cover columns associated with the row
            row_node = col_node.R

            while row_node != col_node:
                self._cover_col(row_node.C)
                row_node = row_node.R

            if self._search(k + 1):  # Sudoku is solved
                return True

            # Uncover columns associated with the row
            row_node = col_node.L
            while row_node != col_node:
                self._uncover_col(row_node.C)
                row_node = row_node.L

            # Move to the next row
            col_node = col_node.D

        # Uncover the column
        self._uncover_col(col_head)

        return False  # Sudoku cannot be solved

    def solve(self) -> Optional[tuple[int, list[list[int]]]]:
        """
        Solves the sudoku.

        Returns:
            Optional[tuple[int, list[list[int]]]]: A tuple with first element
            containing the no. of guesses made by the algorithm, and second
            element containing the solved grid if the grid is solved and `None`
            otherwise
        """
        if not self._search(self.assigned_count):
            return

        solved_grid = [[0 for c in range(9)] for r in range(9)]
        for i in range(81):
            solved_grid[self.O[i].N[0]][self.O[i].N[1]] = self.O[i].N[2]

        return self.guesses, solved_grid
