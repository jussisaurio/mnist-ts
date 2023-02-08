import * as fs from "fs";
import {
  MultiHiddenLayerNetwork,
  createMultiHiddenLayerNetwork,
  multiLayerGradientDescent,
  multiLayerForwardPropagate,
} from "./lib/network";
import { getRandomSample } from "./lib/util";

const ticTacToeDataset = fs.readFileSync("src/dataset/tictactoe.csv", "utf8");

// First four values per row are the output, a 4-length binary vector [x-win, o-win, draw, unfinished]
// The rest are squares, nine 3-length binary vectors [x, o, empty]

const parsedDataset = ticTacToeDataset
  .split("\n")
  .slice(1)
  .map((row) => row.split(",").map((value) => parseInt(value, 10)))
  .map((row) => ({
    input: row.slice(4),
    output: row.slice(0, 4),
  }));

const train = parsedDataset.slice(0, 8000);
const test = parsedDataset.slice(8000);

const command = process.argv[2];

if (command === "train") {
  const epochs = parseInt(process.argv[3], 10) || 2000;
  const learningRate = parseFloat(process.argv[4]) || 0.01;
  const batchSize = parseInt(process.argv[5], 10) || 32;

  const model = process.env.USE_EXISTING
    ? (JSON.parse(
        fs.readFileSync("src/tictactoe-model.json", "utf8")
      ) as MultiHiddenLayerNetwork)
    : createMultiHiddenLayerNetwork({
        inputSize: 27,
        hiddenSizes: [12, 12],
        outputSize: 4,
      });
  const finishedModel = multiLayerGradientDescent(model, {
    learningRate,
    epochs,
    trainingData: train,
    testData: test,
    batchSize,
  });

  const existingModel = (() => {
    try {
      return JSON.parse(fs.readFileSync("src/tictactoe-model.json", "utf8"));
    } catch (e) {
      return null;
    }
  })();

  if (existingModel && existingModel.accuracy > finishedModel.accuracy) {
    console.log(
      "Accuracy is: " +
        finishedModel.accuracy +
        ". Existing model is better with accuracy " +
        existingModel.accuracy
    );
    process.exit(0);
  }

  fs.writeFileSync("src/tictactoe-model.json", JSON.stringify(finishedModel));

  console.log("New model saved with accuracy " + finishedModel.accuracy);
} else if (command === "test") {
  const model = JSON.parse(fs.readFileSync("src/tictactoe-model.json", "utf8"));

  if (!model) throw new Error("No model found. Please train a model first");

  const samples = getRandomSample(test, 10);

  const OUTCOMES = ["X", "O", "Draw", "Unfinished"];

  const printGrid = (grid: number[]) => {
    // 27 length array, 9 squares, 3 values per square
    const partition = (arr: number[], size: number) => {
      const result: ("X" | "O" | ".")[] = [];
      for (let i = 0; i < arr.length; i += size) {
        const values = arr.slice(i, i + size);
        const index = values.indexOf(1);
        if (index === 0) {
          result.push("X");
        } else if (index === 1) {
          result.push("O");
        } else {
          result.push(".");
        }
      }

      return result;
    };

    const squares = partition(grid, 3);

    console.log(
      `  ${squares[0]} | ${squares[1]} | ${squares[2]}
 ---+---+---
  ${squares[3]} | ${squares[4]} | ${squares[5]}
 ---+---+---
  ${squares[6]} | ${squares[7]} | ${squares[8]}`
    );
    console.log("");
  };

  let correct = 0;
  samples.forEach(({ input, output }) => {
    const forward = multiLayerForwardPropagate([input], model.params);
    const lastLayerA = forward[forward.length - 1][1];
    const lastLayerTranspose = lastLayerA.map((arr) => arr[0]);
    const prediction =
      OUTCOMES[lastLayerTranspose.indexOf(Math.max(...lastLayerTranspose))];
    const actual = OUTCOMES[(output as number[]).indexOf(Math.max(...output))];
    printGrid(input);
    console.log(`Prediction: ${prediction} / Actual: ${actual}`);
    console.log(
      `Confidences: ${lastLayerTranspose
        .map((v, i) => `${OUTCOMES[i]}: ${v.toFixed(2)}`)
        .join(", ")}`
    );
    if (prediction === actual) {
      correct++;
      console.log("Correct!");
    } else {
      console.log("Incorrect!");
    }
    console.log("");
    console.log("------------------------");
    console.log("");
  });

  console.log(`Guessed ${correct} out of ${samples.length} correctly`);
} else {
  console.log(
    "Usage: npm run tictactoe:train <epochs> <learningRate> | npm run tictactoe:test"
  );
}
