# Sudoku-AI
This repository is a growing collection of implementations and comparisons of algorithms to solve the classic 9x9 Sudoku.

## Directory Structure
- `algorithms`
    - `bt.py`: Implementation of a simple backtracking algorithm
    - `csp.py`: Implementation of a backtracking algorithm to solve the Sudoku when modelled as a constraint satisfaction problem (CSP)
    - `dlx.py`: Implementation of Donald Knuth's Algorithm X to solve the Sudoku when modelled as an exact cover problem
- `data`
    - `scraping.gs`: Scraping code
    - `NYTimes_Sudoku_Dataset.csv`: Scraped NYTimes Sudoku dataset
- `comparison.ipynb`: Algorithm comparisons code

## Datasets
We compare algorithms in two settings.
### [NYTimes Sudoku Dataset](data/NYTimes_Sudoku_Dataset.csv)
NYTimes publishes easy, medium, and hard classic 9x9 Sudokus daily. We scrape the website to prepare a Sudoku dataset using a [Google Apps Script](data/scraping.gs). The script is setup to automatically update a Google Sheet daily.

As of May 11th, 2023, the dataset contains 711 Sudokus for each difficulty, with the following unfilled cell count distributions.
<p align="center"><img width="400" height="300" src="assets/uccd.png"></p>
We observe that:

- As expected, medium and hard Sudokus have larger unfilled cell counts compared to easy Sudokus.
- Interestingly, medium and hard Sudokus have similar unfilled cell counts. This suggests that the key factor separating medium from hard Sudokus is the arrangement of unfilled cells (as opposed to their count).

### AI Escargot (*"The Most Difficult Sudoku Puzzle"*)
In November 2006, a Finnish applied mathematician, Arto Inkala, claimed to have created the world's hardest Sudoku. In his words, *"I called the puzzle AI Escargot, because it looks like a snail. Solving it is like an intellectual culinary pleasure. AI are my initials"*, and *"Escargot demands those tackling it to consider eight casual relationships simultaneously, while the most complicated variants attempted by the public require people to think of only one or two combinations at any one time"*.
<p align="center"><img width="300" height="300" src="assets/ai_escargot.png"></p>

## Algorithms
- [**Simple Backtracking**](algorithms/bt.py): Simple backtracking is perhaps the simplest sudoku solving algorithm and serves as a baseline. It entails iterating over the sudoku grid and assigning valid values to unfilled cells (a value is considered valid if there is no other cell with the same value in the row/column/3x3 box of the given cell). In case assignments lead to an unfeasible scenario, the algorithm backtracks and attempts alternative assignments to the unfilled cells.

- [**Constraint Satisfaction Problem (CSP)**](algorithms/csp.py): Simple backtracking can be enhanced when modelling the sudoku as a [constraint satisfaction problem (CSP)](https://en.wikipedia.org/wiki/Constraint_satisfaction_problem). Of particular significance is the *Maintaining Arc Consistency (MAC) algorithm*, which trims the set of possible values for other unfilled cells whenever an unfilled cell is assigned. Additionally, the algorithm can be made more efficient using the *minimum-remaining-values (MRV) heuristic* (assign the next value to the unfilled cell with the fewest possible values), the *degree heuristic* (assign the next value to the unfilled cell that is involved in the highest no. of constraints with other unfilled cells), and the *least-constraining-value heuristic* (assign the next value that yields the highest number of consistent values of neighboring cells) (see for reference Chapter 6 of Russell, S. J., Norvig, P., & Davis, E. (2010). Artificial intelligence: a modern approach. 3rd ed. Upper Saddle River, NJ: Prentice Hall).

- [**Algorithm X (DLX)**](algorithms/dlx.py): The Sudoku can be modelled as an [exact cover problem](https://en.wikipedia.org/wiki/Exact_cover), which lends itself to solving using the [dancing links](https://en.wikipedia.org/wiki/Dancing_Links) implementation of [Donald Knuth's Algorithm X](https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X) (see for reference [here](https://arxiv.org/pdf/cs/0011047.pdf)).


## Comparisons
### NYTimes Sudoku Dataset
### AI Escargot
<div align="center">
<table>
    <tr>
        <th>Algorithm</th>
        <th>No. of guesses</th>
    </tr>
    <tr>
        <td>Simple Backtracking</td>
        <td>8969</td>
    </tr>
    <tr>
        <td>Constraint Satisfaction Problem</td>
        <td>344</td>
    </tr>
    <tr>
        <td>Algorithm X</td>
        <td>145</td>
    </tr>
</table>
</div>