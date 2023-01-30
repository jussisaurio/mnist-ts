import { mnistTestSet, mnistTrainingSet } from "./dataset/parseDataset";

import * as fs from "fs";
import { matrixMultiply, matrixTranspose } from "./matrix";
import { relu, reluDerivative, sigmoid } from "./activation";

const train = mnistTrainingSet;
const test = mnistTestSet;

type Network = {
  params: {
    weights1: number[][];
    weights2: number[][];
    biases1: number[][];
    biases2: number[][];
  };
  accuracy: number;
};

function createNetwork(): Network {
  const INPUT_SIZE = 784;
  const HIDDEN_SIZE = 16;
  const OUTPUT_SIZE = 10;

  const weights1 = Array(HIDDEN_SIZE)
    .fill(0)
    .map(() =>
      Array(INPUT_SIZE)
        .fill(0)
        .map(() => Math.random() - 0.5)
    );

  const weights2 = Array(OUTPUT_SIZE)
    .fill(0)
    .map(() =>
      Array(HIDDEN_SIZE)
        .fill(0)
        .map(() => Math.random() - 0.5)
    );

  const biases1 = Array(HIDDEN_SIZE)
    .fill(0)
    .map(() => [Math.random() * 2 - 1]);

  const biases2 = Array(OUTPUT_SIZE)
    .fill(0)
    .map(() => [Math.random() * 2 - 1]);

  return {
    params: {
      weights1,
      weights2,
      biases1,
      biases2,
    },
    accuracy: 0,
  };
}

function forwardPropagate(input: number[], params: Network["params"]) {
  const { weights1, weights2, biases1, biases2 } = params;

  // weights1 matrix dimensions (rows, cols) are [HIDDEN_SIZE, INPUT_SIZE]
  // input vector dimensions are [1, INPUT_SIZE] so we need to transpose it to [INPUT_SIZE, 1]
  // in matmul, the dimensions of the result are multiplier's rows and multiplicand's cols,
  // so the dimensions of z1 are [HIDDEN_SIZE, 1], which is what we expect:
  // each hidden node has a single output
  const transposedInput = matrixTranspose([input]);
  const z1 = matrixMultiply(weights1, transposedInput).map((arr, i) =>
    arr.map((v) => v + biases1[i][0])
  );
  // run the hidden layer through the activation function to get some nonlinearity up in this motha
  // a1 is the output of the hidden layer, so it has the same dimensions as z1
  const a1 = z1.map((arr) => arr.map(relu));

  // weights2 matrix dimensions (rows, cols) are [OUTPUT_SIZE, HIDDEN_SIZE]
  // a1 vector dimensions are [HIDDEN_SIZE, 1]
  // in matmul, the dimensions of the result are multiplier's rows and multiplicand's cols,
  // so the dimensions of z2 are [OUTPUT_SIZE, 1], which is what we expect:
  // each output node has a single output
  const z2 = matrixMultiply(weights2, a1).map((arr, i) =>
    arr.map((v) => v + biases2[i][0])
  );

  // run the output layer through the activation function to get some nonlinearity up in this motha
  // sigmoid clamps the output between 0 and 1, which is what we want for a probability
  const a2 = z2.map((arr) => arr.map((v) => sigmoid(v)));

  return {
    a1,
    a2,
    z1,
    z2,
  };
}

type BackpropParams = ReturnType<typeof forwardPropagate> & {
  weights2: number[][];
  input: number[];
  expected: number[];
};

function backPropagate(params: BackpropParams) {
  const { a1, a2, z1, z2, input, weights2, expected } = params;

  // the output layer has 'm' numbers (in this case 10)
  const m = expected.length;
  // Get the error for each output
  // dimensions of a2 are (rows, cols): [OUTPUT_SIZE, 1]
  const dZ2 = a2.map((arr, i) => arr.map((v) => v - expected[i]));
  // how much does each weight (connecting a hidden node to an output node) contribute to the error
  // dimensions of dZ2 are (rows, cols): [OUTPUT_SIZE, 1]
  // dimensions of a1 are [HIDDEN_SIZE, 1] so we need to transpose it to [1, HIDDEN_SIZE]
  // in matmul, because the dimensions are multiplier's rows * multiplicand's cols,
  // the result is [OUTPUT_SIZE, HIDDEN_SIZE], which is what we expect:
  // each output node (10) has a weight for each hidden node (16)
  const dW2 = matrixMultiply(dZ2, matrixTranspose(a1)).map((arr) =>
    arr.map((v) => v / m)
  );

  // bias for the output nodes is simply the error for that node
  const dB2 = dZ2.map((arr) => arr.map((v) => v / m));
  // Get the error for each hidden node
  // dimensions of weights2 when transposed are (rows, cols): [HIDDEN_SIZE, OUTPUT_SIZE]
  // dimensions of dZ2 are (rows, cols): [OUTPUT_SIZE, 1]
  // in matmul, because the dimensions are multiplier's rows * multiplicand's cols,
  // the result, dZ1, is [HIDDEN_SIZE, 1], which is what we expect:
  // each hidden node (16) has a single delta for each of its outputs;
  // the z1 output dimensions are naturally also [HIDDEN_SIZE, 1]
  const dZ1 = matrixMultiply(matrixTranspose(weights2), dZ2).map((arr, i) =>
    arr.map((v) => v * reluDerivative(z1[i][0]))
  );
  // how much does each weight (connecting an input node to a hidden node) contribute to the error
  const dW1 = matrixMultiply(dZ1, [input]).map((arr) => arr.map((v) => v / m));
  // bias for the hidden nodes is simply the activation error for that node
  const dB1 = dZ1.map((arr) => arr.map((v) => v / m));

  return {
    dW1,
    dW2,
    dB1,
    dB2,
  };
}

function updateWeightsAndBiases(
  params: Network["params"],
  dW1: Network["params"]["weights1"],
  dW2: Network["params"]["weights2"],
  dB1: Network["params"]["biases1"],
  dB2: Network["params"]["biases2"],
  learningRate: number
): void {
  const { weights1, weights2, biases1, biases2 } = params;

  const newWeights1 = weights1.map((arr, i) =>
    arr.map((v, j) => v - learningRate * dW1[i][j])
  );
  const newWeights2 = weights2.map((arr, i) =>
    arr.map((v, j) => v - learningRate * dW2[i][j])
  );
  const newBiases1 = biases1.map((v, i) => [v[0] - learningRate * dB1[i][0]]);
  const newBiases2 = biases2.map((v, i) => [v[0] - learningRate * dB2[i][0]]);

  params.weights1 = newWeights1;
  params.weights2 = newWeights2;
  params.biases1 = newBiases1;
  params.biases2 = newBiases2;
}

function getAccuracy(
  test: {
    input: number[];
    output: number[];
  }[],
  params: Network["params"]
) {
  const correct = test.reduce((acc, { input, output }) => {
    const { a2 } = forwardPropagate(input, params);
    const a2transpose = a2.map((arr) => arr[0]);
    const prediction = a2transpose.indexOf(Math.max(...a2transpose));
    const actual = output.indexOf(Math.max(...output));
    return prediction === actual ? acc + 1 : acc;
  }, 0);

  return correct / test.length;
}

type GradientDescentParams = {
  learningRate: number;
  epochs: number;
  trainingData: {
    input: number[];
    output: number[];
  }[];
  testData: {
    input: number[];
    output: number[];
  }[];
};

function getRandomSample<T extends any>(arr: T[], size: number) {
  const sample = [];
  for (let i = 0; i < size; i++) {
    const index = Math.floor(Math.random() * arr.length);
    sample.push(arr[index]);
  }
  return sample;
}

function gradientDescent(props: GradientDescentParams) {
  const model = process.env.USE_EXISTING
    ? (JSON.parse(fs.readFileSync("src/model.json", "utf8")) as ReturnType<
        typeof createNetwork
      >)
    : createNetwork();
  const { learningRate, epochs, trainingData, testData } = props;

  for (let i = 0; i < epochs; i++) {
    const trainingsample = getRandomSample(trainingData, 1000);

    for (let j = 0; j < trainingsample.length; j++) {
      const { input, output } = trainingsample[j];
      const forward = forwardPropagate(input, model.params);
      const backprop = backPropagate({
        ...forward,
        input,
        expected: output,
        weights2: model.params.weights2,
      });
      updateWeightsAndBiases(
        model.params,
        backprop.dW1,
        backprop.dW2,
        backprop.dB1,
        backprop.dB2,
        learningRate
      );
    }

    const accuracy = getAccuracy(getRandomSample(testData, 100), model.params);

    console.log(`Epoch ${i + 1} accuracy: ${accuracy}`);
  }

  return {
    ...model,
    accuracy: getAccuracy(testData, model.params),
  };
}

const command = process.argv[2];

if (command === "train") {
  const epochs = parseInt(process.argv[3], 10) || 100;
  const learningRate = parseFloat(process.argv[4]) || 0.01;
  const finishedModel = gradientDescent({
    learningRate,
    epochs,
    trainingData: train,
    testData: test,
  });

  const existingModel = (() => {
    try {
      return JSON.parse(fs.readFileSync("src/model.json", "utf8"));
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

  fs.writeFileSync("src/model.json", JSON.stringify(finishedModel));

  console.log("New model saved with accuracy " + finishedModel.accuracy);
} else if (command === "test") {
  const model = JSON.parse(fs.readFileSync("src/model.json", "utf8"));

  if (!model) throw new Error("No model found. Please train a model first");

  const samples = getRandomSample(test, 10);

  let correct = 0;
  samples.forEach(({ input, output }) => {
    const { a2 } = forwardPropagate(input, model.params);
    const a2transpose = a2.map((arr) => arr[0]);
    const prediction = a2transpose.indexOf(Math.max(...a2transpose));
    const actual = (output as number[]).indexOf(Math.max(...output));
    console.log(`Prediction: ${prediction} Actual: ${actual}`);
    if (prediction === actual) correct++;
  });

  console.log(`Guessed ${correct} out of ${samples.length} correctly`);
} else {
  console.log("Usage: npm run train <epochs> <learningRate> | npm run test");
}
