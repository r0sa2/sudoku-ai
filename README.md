# Sudoku-AI
This repository is a growing collection of implementations and comparisons of 
algorithms to solve the classic 9x9 Sudoku.

## Directory Structure
- `algorithms`
    - `bt.py`: Implementation of a simple backtracking algorithm
    - `csp.py`: Implementation of a backtracking algorithm to solve the Sudoku when modelled as a constraint satisfaction problem (CSP)
    - `dlx.py`: Implementation of Donald Knuth's Algorithm X to solve the Sudoku when modelled as an exact cover problem
- `data`
    - `NYTimes_Sudoku_Dataset.csv`: Scraped NYTimes Sudoku dataset
- `comparison.ipynb`: Algorithm comparisons

## Datasets
We compare algorithms in two settings.
### NYTimes Sudoku Dataset
NYTimes publishes easy, medium, and hard classic 9x9 Sudokus daily. We scrape 
the website to prepare a Sudoku dataset using the following Google Apps Script 
code:
```javascript
function update() {
  let html = UrlFetchApp.fetch('https://www.nytimes.com/puzzles/sudoku/easy').getContentText();

  // Get and parse JSON string
  let startIndex = html.match(/window\.gameData/).index;
  let endIndex = html.match(/}}}/).index;
  let data = html.substring(startIndex + 18, endIndex + 3);
  let dataAsJSON = JSON.parse(data);

  // Extract easy, medium and hard sudokus from object
  let easyGrid = dataAsJSON.easy.puzzle_data.puzzle;
  let mediumGrid = dataAsJSON.medium.puzzle_data.puzzle;
  let hardGrid = dataAsJSON.hard.puzzle_data.puzzle;

  let sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('Dataset');

  // Find next blank row
  let rowIndex = 1;
  while (!sheet.getRange('A' + rowIndex.toString()).isBlank()) {
    rowIndex++;
  }

  // Add data to dataset
  sheet.getRange('A' + rowIndex.toString()).setValue(Utilities.formatDate(new Date(), 'Asia/Calcutta', 'dd/MM/yy'));
  sheet.getRange('B' + rowIndex.toString()).setValue(easyGrid.join());
  sheet.getRange('C' + rowIndex.toString()).setValue(mediumGrid.join());
  sheet.getRange('D' + rowIndex.toString()).setValue(hardGrid.join());
}
```
The script is setup to automatically update a spreadsheet daily.
### AI Escargot (*"The Most Difficult Sudoku Puzzle"*)
In late November 2006, a Finnish applied mathematician, Arto Inkala, claimed to have created the world's hardest Sudoku. In his words, *"I called the puzzle AI Escargot, because it looks like a snail. Solving it is like an intellectual culinary pleasure. AI are my initials"*, and *"Escargot demands those tackling it to consider eight casual relationships simultaneously, while the most complicated variants attempted by the public require people to think of only one or two combinations at any one time"*.

## Algorithms
- **Simple Backtracking**: Simple backtracking is perhaps the simplest sudoku solving algorithm and serves as a baseline. It entails iterating over the sudoku grid and assigning valid values to unfilled cells (a value is considered valid if there is no other cell with the same value in the row/column/3x3 box of the given cell). In case assignments lead to an unfeasible scenario, the algorithm backtracks and attempts alternative assignments to the unfilled cells.
- **Constraint Satisfaction Problem (CSP)**: Simple backtracking can be enhanced when modelling the sudoku as a [constraint satisfaction problem (CSP)](https://en.wikipedia.org/wiki/Constraint_satisfaction_problem). Of particular significance is the *Maintaining Arc Consistency (MAC) algorithm*, which trims the set of possible values for other unfilled cells whenever an unfilled cell is assigned. Additionally, the algorithm can be made more efficient using the *minimum-remaining-values (MRV) heuristic* (assign the next value to the unfilled cell with the fewest possible values), the *degree heuristic* (assign the next value to the unfilled cell that is involved in the highest no. of constraints with other unfilled cells), and the *least-constraining-value heuristic* (assign the next value that yields the highest number of consistent values of neighboring cells) (see for reference Chapter 6 of Russell, S. J., Norvig, P., & Davis, E. (2010). Artificial intelligence: a modern approach. 3rd ed. Upper Saddle River, NJ: Prentice Hall).
- **Algorithm X (DLX)**: The Sudoku can be modelled as an [exact cover problem](https://en.wikipedia.org/wiki/Exact_cover), which lends itself to solving using the [dancing links](https://en.wikipedia.org/wiki/Dancing_Links) implementation of [Donald Knuth's Algorithm X](https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X) (see for reference [here](https://arxiv.org/pdf/cs/0011047.pdf).