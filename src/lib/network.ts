import { relu, reluDerivative, softmax } from "./activation";
import {
  getShape,
  matrixMultiply,
  matrixOf,
  matrixScalarMultiply,
  matrixSubtract,
  matrixTranspose,
} from "./matrix";
import { getRandomSample } from "./util";

export type SingleHiddenLayerNetwork = {
  params: {
    weights1: number[][]; // [HIDDEN_SIZE, INPUT_SIZE]
    weights2: number[][]; // [OUTPUT_SIZE, HIDDEN_SIZE]
    biases1: number[][]; // [HIDDEN_SIZE, 1]
    biases2: number[][]; // [OUTPUT_SIZE, 1]
  };
  accuracy: number;
};

export type SingleHiddenLayerNetworkProps = {
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
};

export function createSingleHiddenLayerNetwork(
  props: SingleHiddenLayerNetworkProps
): SingleHiddenLayerNetwork {
  const {
    inputSize: INPUT_SIZE,
    hiddenSize: HIDDEN_SIZE,
    outputSize: OUTPUT_SIZE,
  } = props;

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

export function singleLayerForwardPropagate(
  input: number[][],
  params: SingleHiddenLayerNetwork["params"]
) {
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

export type SingleLayerBackpropParams = ReturnType<
  typeof singleLayerForwardPropagate
> & {
  weights2: number[][];
  input: number[][];
  expected: number[][];
};

export function singleLayerBackpropagate(params: SingleLayerBackpropParams) {
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

export function singleLayerUpdateWeightsAndBiases(
  network: SingleHiddenLayerNetwork,
  dW1: SingleHiddenLayerNetwork["params"]["weights1"],
  dW2: SingleHiddenLayerNetwork["params"]["weights2"],
  dB1: SingleHiddenLayerNetwork["params"]["biases1"],
  dB2: SingleHiddenLayerNetwork["params"]["biases2"],
  learningRate: number
): void {
  const { params } = network;
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

export type CreateMultiHiddenLayerNetworkProps = {
  inputSize: number;
  hiddenSizes: number[];
  outputSize: number;
};

export type MultiHiddenLayerNetwork = {
  params: {
    weights: number[][][];
    biases: number[][][];
  };
  accuracy: number;
};

export const createMultiHiddenLayerNetwork = (
  props: CreateMultiHiddenLayerNetworkProps
): MultiHiddenLayerNetwork => {
  const {
    inputSize: INPUT_SIZE,
    hiddenSizes: HIDDEN_SIZES,
    outputSize: OUTPUT_SIZE,
  } = props;

  const LAYERS = [...HIDDEN_SIZES, OUTPUT_SIZE];
  const weights = LAYERS.reduce((acc, layerSize, i) => {
    const prevSize = i === 0 ? INPUT_SIZE : LAYERS[i - 1];
    const weight = matrixOf({
      rows: layerSize,
      cols: prevSize,
      initialValue: () => Math.random() - 0.5,
    });
    return [...acc, weight];
  }, [] as number[][][]);

  const biases = LAYERS.map((layerSize) =>
    matrixOf({
      rows: layerSize,
      cols: 1,
      initialValue: () => Math.random() * 2 - 1,
    })
  );

  return {
    params: {
      weights,
      biases,
    },
    accuracy: 0,
  };
};

export function multiLayerForwardPropagate(
  inputs: number[][],
  params: MultiHiddenLayerNetwork["params"]
) {
  const { weights, biases } = params;

  // inputs is the input layer
  // hidden layer count depends on the number of weights

  const outputsAndActivations = weights.reduce(
    (acc, weight, i) => {
      const [prevZ, prevA] = acc[i];
      const z = matrixMultiply(weight, prevA).map(
        (row, j) => row.map((v) => v + biases[i][j][0]) // add bias
      );

      const isOutputLayer = i === weights.length - 1;
      const a = (() => {
        if (isOutputLayer) {
          const zTranspose = matrixTranspose(z);
          return matrixTranspose(zTranspose.map(softmax));
        }
        return z.map((row) => row.map((v) => relu(v)));
      })();
      return [...acc, [z, a]];
    },
    [[matrixTranspose(inputs), matrixTranspose(inputs)]] as number[][][][]
  );

  return outputsAndActivations as [number[][], number[][]][]; // [z, a][]
}

export function multiLayerBackPropagate(
  outputsAndActivations: [number[][], number[][]][],
  targets: number[][],
  params: MultiHiddenLayerNetwork["params"]
) {
  const { weights } = params;

  // for (const [z, a] of outputsAndActivations) {
  //   console.log("Z shape", getShape(z));
  //   console.log("A shape", getShape(a));
  // }

  const dL_dZs = outputsAndActivations.reduceRight((acc, [z, a], i, arr) => {
    const prevDl_dZ = acc[0];
    const isOutputLayer = i === arr.length - 1;
    const dL_dZ = (() => {
      if (isOutputLayer) {
        return matrixSubtract(a, targets);
      }
      const prevWeight = weights[i];
      const prevDl_dZTranspose = matrixTranspose(prevDl_dZ);
      const prevWeightTranspose = matrixTranspose(prevWeight);
      const prevZTranspose = matrixTranspose(z);
      const dZ_dA = prevZTranspose.map((row) =>
        row.map((v) => (v > 0 ? 1 : 0))
      );
      const dL_dA = matrixMultiply(prevWeightTranspose, prevDl_dZTranspose);
      return matrixMultiply(dL_dA, dZ_dA);
    })();
    return [dL_dZ, ...acc];
  }, [] as number[][][]);

  const dL_dWs = dL_dZs.reduce((acc, dL_dZ, i) => {
    const prevA = outputsAndActivations[i][1];
    const dL_dW = matrixMultiply(dL_dZ, matrixTranspose(prevA)).map(
      (row) => row.map((v) => v / prevA[0].length) // divide by sample size to get the average
    );
    return [...acc, dL_dW];
  }, [] as number[][][]);

  const dL_dBs = dL_dZs.map((dL_dZ) =>
    dL_dZ.map((row) => [row.reduce((acc, v) => acc + v, 0)])
  );

  return {
    dWs: dL_dWs,
    dBs: dL_dBs,
  };
}

export function multiLayerUpdateParams(
  params: MultiHiddenLayerNetwork["params"],
  dWs: number[][][],
  dBs: number[][][],
  learningRate: number
) {
  const { weights, biases } = params;

  const newWeights = weights.map((weight, i) =>
    matrixSubtract(weight, matrixScalarMultiply(dWs[i], learningRate))
  );

  const newBiases = biases.map((bias, i) =>
    matrixSubtract(bias, matrixScalarMultiply(dBs[i], learningRate))
  );

  params.weights = newWeights;
  params.biases = newBiases;
}

export function getAccuracy(
  test: {
    input: number[];
    output: number[];
  }[],
  params: SingleHiddenLayerNetwork["params"]
) {
  const correct = test.reduce((acc, { input, output }) => {
    const { a2 } = singleLayerForwardPropagate([input], params);
    const a2transpose = a2.map((arr) => arr[0]);
    const prediction = a2transpose.indexOf(Math.max(...a2transpose));
    const actual = output.indexOf(Math.max(...output));
    return prediction === actual ? acc + 1 : acc;
  }, 0);

  return correct / test.length;
}

export function getAccuracyMulti(
  test: {
    input: number[];
    output: number[];
  }[],
  params: MultiHiddenLayerNetwork["params"]
) {
  const correct = test.reduce((acc, { input, output }) => {
    const outputsAndActivations = multiLayerForwardPropagate([input], params);
    const a2 = outputsAndActivations[outputsAndActivations.length - 1][1];
    const a2transpose = a2.map((arr) => arr[0]);
    const prediction = a2transpose.indexOf(Math.max(...a2transpose));
    const actual = output.indexOf(Math.max(...output));
    return prediction === actual ? acc + 1 : acc;
  }, 0);

  return correct / test.length;
}

export type GradientDescentParams = {
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

export function singleLayerGradientDescent(
  model: SingleHiddenLayerNetwork,
  props: GradientDescentParams
) {
  const { learningRate, epochs, trainingData, testData, batchSize } = props;

  for (let i = 0; i < epochs; i++) {
    console.log(`Epoch ${i + 1} started, batch size: ${batchSize}`);
    let batches = 0;
    for (let j = 0; j < trainingData.length; j += batchSize) {
      batches++;
      const batch = trainingData.slice(j, j + batchSize);
      const input = batch.map((d) => d.input);
      const forward = singleLayerForwardPropagate(input, model.params);

      const expected = batch.map((d) => d.output);
      const backprop = singleLayerBackpropagate({
        ...forward,
        input: matrixTranspose(input),
        expected: matrixTranspose(expected),
        weights2: model.params.weights2,
      });
      singleLayerUpdateWeightsAndBiases(
        model,
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

export function multiLayerGradientDescent(
  model: MultiHiddenLayerNetwork,
  props: GradientDescentParams
) {
  const { learningRate, epochs, trainingData, testData, batchSize } = props;

  for (let i = 0; i < epochs; i++) {
    console.log(`Epoch ${i + 1} started, batch size: ${batchSize}`);
    let batches = 0;
    for (let j = 0; j < trainingData.length; j += batchSize) {
      batches++;
      const batch = trainingData.slice(j, j + batchSize);
      const input = batch.map((d) => d.input);
      const forward = multiLayerForwardPropagate(input, model.params);

      const expected = batch.map((d) => d.output);
      const backprop = multiLayerBackPropagate(
        forward,
        matrixTranspose(expected),
        {
          weights: model.params.weights,
          biases: model.params.biases,
        }
      );
      multiLayerUpdateParams(
        model.params,
        backprop.dWs,
        backprop.dBs,
        learningRate
      );

      // every now and then log the accuracy
      if (batches % 200 === 0) {
        const accuracy = getAccuracyMulti(
          getRandomSample(testData, 100),
          model.params
        );

        // console.log(`Interim accuracy: ${accuracy}`);
      }
    }

    const accuracy = getAccuracyMulti(
      getRandomSample(testData, 100),
      model.params
    );

    console.log(`Epoch ${i + 1} accuracy: ${accuracy}`);
  }

  return {
    ...model,
    accuracy: getAccuracyMulti(testData, model.params),
  };
}
