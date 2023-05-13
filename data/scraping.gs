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