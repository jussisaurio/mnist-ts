import { mnistTestSet, mnistTrainingSet } from "./dataset/parseDataset";

import * as fs from "fs";
import { matrixMultiply, matrixTranspose } from "./matrix";
import { relu, reluDerivative, softmax } from "./activation";

const train = mnistTrainingSet;
const test = mnistTestSet;

type Network = {
  params: {
    weights1: number[][]; // [HIDDEN_SIZE, INPUT_SIZE]
    weights2: number[][]; // [OUTPUT_SIZE, HIDDEN_SIZE]
    biases1: number[][]; // [HIDDEN_SIZE, 1]
    biases2: number[][]; // [OUTPUT_SIZE, 1]
  };
  accuracy: number;
};

const matrixOf = ({
  rows,
  cols,
  initialValue,
}: {
  rows: number;
  cols: number;
  initialValue: () => number;
}) =>
  Array(rows)
    .fill(0)
    .map(() => Array(cols).fill(0).map(initialValue));

function createNetwork(): Network {
  const INPUT_SIZE = 784;
  const HIDDEN_SIZE = 16;
  const OUTPUT_SIZE = 10;

  const weights1 = matrixOf({
    rows: HIDDEN_SIZE,
    cols: INPUT_SIZE,
    initialValue: () => Math.random() - 0.5,
  });

  const weights2 = matrixOf({
    rows: OUTPUT_SIZE,
    cols: HIDDEN_SIZE,
    initialValue: () => Math.random() - 0.5,
  });

  const biases1 = matrixOf({
    rows: HIDDEN_SIZE,
    cols: 1,
    initialValue: () => Math.random() * 2 - 1,
  });

  const biases2 = matrixOf({
    rows: OUTPUT_SIZE,
    cols: 1,
    initialValue: () => Math.random() * 2 - 1,
  });

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

function forwardPropagate(input: number[][], params: Network["params"]) {
  const { weights1, weights2, biases1, biases2 } = params;

  // weights1 matrix dimensions (rows, cols) are [HIDDEN_SIZE, INPUT_SIZE]
  // input vector dimensions are [SAMPLE_SIZE, INPUT_SIZE] so we need to transpose it to [INPUT_SIZE, SAMPLE_SIZE]
  // in matmul, the dimensions of the result are multiplier's rows and multiplicand's cols,
  // so the dimensions of z1 are [HIDDEN_SIZE, SAMPLE_SIZE], which is what we expect:
  // each hidden node has a single output for each sample
  const samples = matrixTranspose(input);
  const z1 = matrixMultiply(weights1, samples).map((arr, i) =>
    arr.map((v) => v + biases1[i][0])
  );

  // run the hidden layer through the activation function to get some nonlinearity up in this motha
  // a1 is the output of the hidden layer, so it has the same dimensions as z1
  const a1 = z1.map((arr) => arr.map(relu));

  // weights2 matrix dimensions (rows, cols) are [OUTPUT_SIZE, HIDDEN_SIZE]
  // a1 vector dimensions are [HIDDEN_SIZE, SAMPLE_SIZE]
  // in matmul, the dimensions of the result are multiplier's rows and multiplicand's cols,
  // so the dimensions of z2 are [OUTPUT_SIZE, SAMPLE_SIZE], which is what we expect:
  // each output node has a single output for each sample
  const z2 = matrixMultiply(weights2, a1).map((arr, i) =>
    arr.map((v) => v + biases2[i][0])
  );

  // run the output layer through the activation function to get some nonlinearity up in this motha
  // the sum of the softmax values should be 1, so we can use it as a probability distribution
  const z2Transposed = matrixTranspose(z2);
  const a2 = matrixTranspose(z2Transposed.map(softmax));

  return {
    a1,
    a2,
    z1,
    z2,
  };
}

type BackpropParams = ReturnType<typeof forwardPropagate> & {
  weights2: number[][];
  input: number[][];
  expected: number[][];
};

function backPropagate(params: BackpropParams) {
  const { a1, a2, z1, input, weights2: w2, expected } = params;

  // expected is the expected output of the network (dimensions: [SAMPLE_SIZE, OUTPUT_SIZE])
  // a0, i.e. input activation is just the inputs (dimensions: [SAMPLE_SIZE, INPUT_SIZE])
  const a0 = input;
  // z1 is the unactivated output of the hidden layer (dimensions: [HIDDEN_SIZE, SAMPLE_SIZE])
  // a1 is the activation output of the hidden layer (dimensions: [HIDDEN_SIZE, SAMPLE_SIZE])
  // z2 is the unactivated output of the output layer (dimensions: [OUTPUT_SIZE, SAMPLE_SIZE])
  // a2 is the activation output, i.e. prediction of the output layer (dimensions: [OUTPUT_SIZE, SAMPLE_SIZE])
  // w2 is the weights matrix connecting the hidden layer to the output layer (dimensions: [OUTPUT_SIZE, HIDDEN_SIZE])

  // We are implicitly using cross-entropy loss with softmax here:
  // L = -∑expected * log(a2), where a2 is the actual output:
  // a2 = softmax(w2 * z2 + b2)
  // From this can be derived that the derivative of the loss with respect to the unactivated output of the output layer is just a2 - expected,
  // explained here: http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
  // So, how much does the loss change when we change the unactivated output?
  // dL/dZ2 = a2 - expected
  const dL_dZ2 = a2.map((arr, i) => arr.map((v, j) => v - expected[i][j]));

  // How much does the loss change when we change the weights connecting the hidden layer to the output layer?
  // Using the chain rule:
  // dL/dW2 = dL/dZ2 * dZ2/dW2
  //        = dl/dZ2 * d/dW2(a1 * w2 + b2)
  //        = dl/dZ2 * a1
  const dL_dW2 = matrixMultiply(dL_dZ2, matrixTranspose(a1)).map(
    (row) => row.map((v) => v / a1[0].length) // divide by the sample size to get the average
  );

  // How much does the loss change when we change the biases connecting the hidden layer to the output layer?
  // dL/dB2 = dL/dZ2 * dZ2/dB2
  //        = dL/dZ2 * d/dB2(a1 * w2 + b2)
  //        = dL/dZ2 * 1
  const dL_dB2 = dL_dZ2.map((row) => [row.reduce((acc, v) => acc + v, 0)]);

  // How much does the loss change when we change the unactivated output of the hidden layer?
  // dL/dZ1 = dL/dZ2 * dZ2/dA1 * dA1/dZ1
  //        = dL/dZ2 * d/dA1 (a1 * w2 + b2) * d/dZ1 (relu(z1))
  //        = dL/dZ2 * w2 * relu'(z1) * 1
  //        = dL/dZ2 * w2 * relu'(z1)
  const dL_dZ1 = matrixMultiply(matrixTranspose(w2), dL_dZ2).map((arr, i) =>
    arr.map((v) => v * reluDerivative(z1[i][0]))
  );

  // How much does the loss change when we change the weights connecting the input layer to the hidden layer?
  // dL/dW1 = dL/dZ1 * dZ1/dW1
  //        = dL/dZ1 * d/dW1(a0 * w1 + b1)
  //        = dL/dZ1 * a0
  const dL_dW1 = matrixMultiply(dL_dZ1, matrixTranspose(a0)).map(
    (row) => row.map((v) => v / a0[0].length) // divide by sample size to get the average
  );

  // How much does the loss change when we change the biases connecting the input layer to the hidden layer?
  // dL/dB1 = dL/dZ1 * dZ1/dB1
  //        = dL/dZ1 * d/dB1(a0 * w1 + b1)
  //        = dL/dZ1 * 1
  const dL_dB1 = dL_dZ1.map((row) => [row.reduce((acc, v) => acc + v, 0)]);

  return {
    dW1: dL_dW1,
    dW2: dL_dW2,
    dB1: dL_dB1,
    dB2: dL_dB2,
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
    const { a2 } = forwardPropagate([input], params);
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
  batchSize: number;
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
  const { learningRate, epochs, trainingData, testData, batchSize } = props;

  for (let i = 0; i < epochs; i++) {
    console.log(`Epoch ${i + 1} started, batch size: ${batchSize}`);
    let batches = 0;
    for (let j = 0; j < trainingData.length; j += batchSize) {
      batches++;
      const batch = trainingData.slice(j, j + batchSize);
      const input = batch.map((d) => d.input);
      const forward = forwardPropagate(input, model.params);

      const expected = batch.map((d) => d.output);
      const backprop = backPropagate({
        ...forward,
        input: matrixTranspose(input),
        expected: matrixTranspose(expected),
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

      // every now and then log the accuracy
      if (batches % 200 === 0) {
        const accuracy = getAccuracy(
          getRandomSample(testData, 100),
          model.params
        );

        // console.log(`Interim accuracy: ${accuracy}`);
      }
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
  const epochs = parseInt(process.argv[3], 10) || 10;
  const learningRate = parseFloat(process.argv[4]) || 0.01;
  const batchSize = parseInt(process.argv[5], 10) || 10;
  const finishedModel = gradientDescent({
    learningRate,
    epochs,
    trainingData: train,
    testData: test,
    batchSize,
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
    const { a2 } = forwardPropagate([input], model.params);
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
