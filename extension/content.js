const board = document.getElementsByClassName("su-board")[0];
const unsolvedGrid = [...Array(9)].map(elem => Array(9).fill(0));
const solvedGrid = [...Array(9)].map(elem => Array(9).fill(0));
const encodings = new Set();

const getEncodings = (row, col, val) => {
  return new Set([
    `${row}(${val})`, 
    `(${val})${col}`, 
    `${Math.floor(row / 3)}(${val})${Math.floor(col / 3)}`
  ]);
}

const areDisjoint = (set1, set2) => {
    if (set1.size > set2.size) [set1, set2] = [set2, set1];
    for (const elem of set1) if (set2.has(elem)) return false;
    return true;
}

const backtrack = (row, col) => {
  if (row === 9 && col === 0) {
    return true;
  } else if (solvedGrid[row][col] !== 0) {
    return col < 8 ? backtrack(row, col + 1) : backtrack(row + 1, 0);
  } else {
    for (let val = 1; val < 10; val++) {
      const encs = getEncodings(row, col, val);
      if (!areDisjoint(encodings, encs)) continue;
      solvedGrid[row][col] = val;
      for (const elem of encs) encodings.add(elem);
      if (col < 8 ? backtrack(row, col + 1) : backtrack(row + 1, 0)) return true;
      solvedGrid[row][col] = 0;
      for (const elem of encs) encodings.delete(elem);
    }
    return false;
  }
}

for (let row = 0; row < 9; row++) {
  for (let col = 0; col < 9; col++) {
    let idx = 9 * row + col;
    if (board.childNodes[idx].getAttribute("class").includes("prefilled")) {
      let val = parseInt(board.childNodes[idx].getAttribute("aria-label"));
      unsolvedGrid[row][col] = val;
      solvedGrid[row][col] = val;
      for (const elem of getEncodings(row, col, val)) encodings.add(elem);
    }
  }
}

backtrack(0, 0);

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  sendResponse({
    "unsolvedGrid": unsolvedGrid.map(elem => elem.slice()),
    "solvedGrid": solvedGrid.map(elem => elem.slice())
  });
});