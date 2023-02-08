import { mnistTestSet, mnistTrainingSet } from "./dataset/parseDataset";

import * as fs from "fs";
import {
  createSingleHiddenLayerNetwork,
  singleLayerForwardPropagate,
  singleLayerGradientDescent,
  SingleHiddenLayerNetwork,
} from "./lib/network";
import { getRandomSample } from "./lib/util";

const train = mnistTrainingSet;
const test = mnistTestSet;

const command = process.argv[2];

if (command === "train") {
  const epochs = parseInt(process.argv[3], 10) || 10;
  const learningRate = parseFloat(process.argv[4]) || 0.01;
  const batchSize = parseInt(process.argv[5], 10) || 10;

  const model = process.env.USE_EXISTING
    ? (JSON.parse(
        fs.readFileSync("src/mnist-model.json", "utf8")
      ) as SingleHiddenLayerNetwork)
    : createSingleHiddenLayerNetwork({
        inputSize: 784,
        hiddenSize: 16,
        outputSize: 10,
      });
  const finishedModel = singleLayerGradientDescent(model, {
    learningRate,
    epochs,
    trainingData: train,
    testData: test,
    batchSize,
  });

  const existingModel = (() => {
    try {
      return JSON.parse(fs.readFileSync("src/mnist-model.json", "utf8"));
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

  fs.writeFileSync("src/mnist-model.json", JSON.stringify(finishedModel));

  console.log("New model saved with accuracy " + finishedModel.accuracy);
} else if (command === "test") {
  const model = JSON.parse(fs.readFileSync("src/mnist-model.json", "utf8"));

  if (!model) throw new Error("No model found. Please train a model first");

  const samples = getRandomSample(test, 10);

  let correct = 0;
  samples.forEach(({ input, output }) => {
    const { a2 } = singleLayerForwardPropagate([input], model.params);
    const a2transpose = a2.map((arr) => arr[0]);
    const prediction = a2transpose.indexOf(Math.max(...a2transpose));
    const actual = (output as number[]).indexOf(Math.max(...output));
    console.log(`Prediction: ${prediction} Actual: ${actual}`);
    if (prediction === actual) correct++;
  });

  console.log(`Guessed ${correct} out of ${samples.length} correctly`);
} else {
  console.log(
    "Usage: npm run mnist:train <epochs> <learningRate> | npm run mnist:test"
  );
}
