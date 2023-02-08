import { relu, reluDerivative, softmax } from "./activation";
import { mMul, matrixOf, mSD, mSM, mSub, T } from "./matrix";

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

  const outputsAndActivations = weights.reduce(
    (acc, weight, i) => {
      const [_, prevA] = acc[i];
      // z = w * a + b
      const z = mMul(weight, prevA).map(
        (row, j) => row.map((v) => v + biases[i][j][0]) // add bias
      );

      const isOutputLayer = i === weights.length - 1;
      // a = softmax(z) if output layer, relu(z) otherwise.
      // softmax is used for the output layer because it's a classification problem, and we want the output to be a probability distribution.
      const a = (() => {
        if (isOutputLayer) {
          const zTranspose = T(z);
          return T(zTranspose.map(softmax));
        }
        return z.map((row) => row.map(relu));
      })();
      return [...acc, [z, a]];
    },
    // The input layer activation is just the input
    [[T(inputs), T(inputs)]] as number[][][][]
  );

  return outputsAndActivations as [number[][], number[][]][]; // [z, a][]
}

export function multiLayerBackPropagate(
  outputsAndActivations: [number[][], number[][]][],
  expected: number[][],
  params: MultiHiddenLayerNetwork["params"]
) {
  const { weights } = params;

  const batchSize = expected[0].length;

  const [_, inputLayerA] = outputsAndActivations[0];
  const hiddenLayers = outputsAndActivations.slice(
    1,
    outputsAndActivations.length - 1
  );
  const [__, outputLayerA] =
    outputsAndActivations[outputsAndActivations.length - 1];

  const dW: any[] = [];
  const dB: any[] = [];

  const dZOutput = mSub(outputLayerA, expected);

  const [___, lastHiddenLayerA] = hiddenLayers[hiddenLayers.length - 1];

  dW.unshift(mSD(mMul(dZOutput, T(lastHiddenLayerA)), batchSize));
  dB.unshift(mSD(dZOutput, batchSize));

  let dZPrev = dZOutput;
  for (let i = hiddenLayers.length - 1; i >= 0; i--) {
    const [hiddenZ] = hiddenLayers[i];
    const nextWeights = T(weights[i + 1]);
    const dZ = mMul(nextWeights, dZPrev).map((row, j, self) => {
      return row.map((v, k) => v * reluDerivative(hiddenZ[j][k]));
    });
    const prevA = i === 0 ? inputLayerA : hiddenLayers[i - 1][1];
    dW.unshift(mSD(mMul(dZ, T(prevA)), batchSize));
    dB.unshift(mSD(dZ, batchSize));
    dZPrev = dZ;
  }

  return {
    dWs: dW,
    dBs: dB,
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
    mSub(weight, mSM(dWs[i], learningRate))
  );

  const newBiases = biases.map((bias, i) =>
    mSub(bias, mSM(dBs[i], learningRate))
  );

  params.weights = newWeights;
  params.biases = newBiases;
}

export function getAccuracy(
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

export function multiLayerGradientDescent(
  model: MultiHiddenLayerNetwork,
  props: GradientDescentParams
) {
  const { learningRate, epochs, trainingData, testData, batchSize } = props;

  for (let i = 0; i < epochs; i++) {
    console.log(
      `Epoch ${i + 1} of ${epochs} started, batch size: ${batchSize}`
    );
    let batches = 0;
    for (let j = 0; j < trainingData.length; j += batchSize) {
      batches++;
      const batch = trainingData.slice(j, j + batchSize);
      const input = batch.map((d) => d.input);
      const forward = multiLayerForwardPropagate(input, model.params);

      const expected = batch.map((d) => d.output);
      const backprop = multiLayerBackPropagate(forward, T(expected), {
        weights: model.params.weights,
        biases: model.params.biases,
      });
      multiLayerUpdateParams(
        model.params,
        backprop.dWs,
        backprop.dBs,
        learningRate
      );
    }

    const accuracy = getAccuracy(testData, model.params);

    console.log(`Epoch ${i + 1} accuracy: ${accuracy}`);
  }

  return {
    ...model,
    accuracy: getAccuracy(testData, model.params),
  };
}
