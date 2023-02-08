// Generate dataset of tic-tac-toe games (3x3 board, 2 players, 9 squares)

type Player = "X" | "O";
type SquareState = Player | "EMPTY";

type Board = [
  [SquareState, SquareState, SquareState],
  [SquareState, SquareState, SquareState],
  [SquareState, SquareState, SquareState]
];

type GameState = "UNFINISHED" | "DRAW" | Player;

const determineWinner = (board: Board): GameState => {
  const rows = board;
  const columns = [
    [board[0][0], board[1][0], board[2][0]],
    [board[0][1], board[1][1], board[2][1]],
    [board[0][2], board[1][2], board[2][2]],
  ];
  const diagonals = [
    [board[0][0], board[1][1], board[2][2]],
    [board[0][2], board[1][1], board[2][0]],
  ];
  const lines = [...rows, ...columns, ...diagonals];

  for (const line of lines) {
    const [a, b, c] = line;
    if (a !== "EMPTY" && a === b && b === c) {
      return a;
    }
  }

  if (board.flat().includes("EMPTY")) {
    return "UNFINISHED";
  }

  return "DRAW";
};

const generateValidGame = (): [Board, GameState] => {
  const board: Board = [
    ["EMPTY", "EMPTY", "EMPTY"],
    ["EMPTY", "EMPTY", "EMPTY"],
    ["EMPTY", "EMPTY", "EMPTY"],
  ];

  let winner: GameState = "UNFINISHED";

  // Randomly fill the board until there is a winner
  // 3% chance to abort early, which cumulates to 24% chance to abort before the board is full
  const randomFillingOrder = Array(9)
    .fill(0)
    .map((_, i) => i)
    .sort(() => Math.random() - 0.5);

  let startingPlayer: Player = Math.random() < 0.5 ? "X" : "O";

  while (winner === "UNFINISHED" && Math.random() <= 0.97) {
    const idx = randomFillingOrder.pop()!;
    const x = Math.floor(idx / 3);
    const y = idx % 3;
    board[x][y] = startingPlayer;
    startingPlayer = startingPlayer === "X" ? "O" : "X";
    winner = determineWinner(board);
  }

  return [board, winner];
};

const generateTrainingData = (numGames: number) => {
  const trainingData: [Board, GameState][] = [];

  for (let i = 0; i < numGames; i++) {
    const [board, winner] = generateValidGame();
    trainingData.push([board, winner]);
  }

  return trainingData;
};

const trainingDataToCsv = (trainingData: [Board, GameState][]) => {
  // Each square should be encoded as a binary vector of length 3 (i.e. [1, 0, 0] for "X", [0, 1, 0] for "O", [0, 0, 1] for "EMPTY")
  // The outcome should be encoded as a binary vector of length 4
  // (i.e. [1, 0, 0, 0] for "X", [0, 1, 0, 0] for "O", [0, 0, 1, 0] for "DRAW", [0, 0, 0, 1] for "UNFINISHED")

  const squareToBinaryVector = (square: SquareState) => {
    switch (square) {
      case "X":
        return [1, 0, 0];
      case "O":
        return [0, 1, 0];
      case "EMPTY":
        return [0, 0, 1];
    }
  };

  const outcomeToBinaryVector = (outcome: GameState) => {
    switch (outcome) {
      case "X":
        return [1, 0, 0, 0];
      case "O":
        return [0, 1, 0, 0];
      case "DRAW":
        return [0, 0, 1, 0];
      case "UNFINISHED":
        return [0, 0, 0, 1];
    }
  };

  const trainingDataCsv = trainingData
    .map(([board, outcome]) => {
      const boardCsv = board.flat().map(squareToBinaryVector).flat().join(",");
      const outcomeCsv = outcomeToBinaryVector(outcome).join(",");
      return `${outcomeCsv},${boardCsv}`;
    })
    .join("\n");

  const header =
    "outcome_x,outcome_o,outcome_draw,outcome_unfinished,square_0_x,square_0_o,square_0_empty,square_1_x,square_1_o,square_1_empty,square_2_x,square_2_o,square_2_empty,square_3_x,square_3_o,square_3_empty,square_4_x,square_4_o,square_4_empty,square_5_x,square_5_o,square_5_empty,square_6_x,square_6_o,square_6_empty,square_7_x,square_7_o,square_7_empty,square_8_x,square_8_o,square_8_empty";

  return `${header}\n${trainingDataCsv}`;
};

const trainingData = generateTrainingData(10000);

const trainingDataCsv = trainingDataToCsv(trainingData);

// Write the training data to a CSV file
const fs = require("fs");

fs.writeFileSync("tictactoe.csv", trainingDataCsv);
