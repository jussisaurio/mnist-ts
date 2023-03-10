import { relu, reluDerivative, softmax } from "./activation";
import { mMul, matrixOf, mSD, mSM, mSub, T, mMap, mHad } from "./matrix";

export type Matrix = number[][];

export type CreateNetworkProps = {
  inputSize: number;
  hiddenSizes: number[];
  outputSize: number;
};

export type Network = {
  params: {
    weights: Matrix[];
    biases: Matrix[];
  };
  accuracy: number;
};

export const createNetwork = (props: CreateNetworkProps): Network => {
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
  }, [] as Matrix[]);

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

export function forwardPropagate(inputs: Matrix, params: Network["params"]) {
  const { weights, biases } = params;

  const weightedSumsAndActivations = weights.reduce(
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
    [[T(inputs), T(inputs)]] as Matrix[][]
  );

  return weightedSumsAndActivations as [Matrix, Matrix][]; // [z, a][]
}

export function backPropagate(
  weightedSumsAndActivations: [Matrix, Matrix][],
  expected: Matrix,
  params: Network["params"]
) {
  const { weights } = params;

  const batchSize = expected[0].length;

  const A_Input = weightedSumsAndActivations[0][1];
  const hiddenLayers = weightedSumsAndActivations.slice(
    1,
    weightedSumsAndActivations.length - 1
  );
  const A_Output =
    weightedSumsAndActivations[weightedSumsAndActivations.length - 1][1];

  // We want to calculate the negative gradient of the cost function with respect to the weights and biases,
  // I.e. the derivative of the cost function with respect to the weights and biases that will reduce the cost function the most.
  // Using these, we can update the weights and biases to reduce the cost function, i.e. reduce how much our network is wrong.
  const dW: Matrix[] = [];
  const dB: Matrix[] = [];

  // To update the weights, we want to calculate the derivative of the cost function with respect to the weights.
  // Using the chain rule:
  // dLoss/dW_Output = dLoss/dZ_Output * dZ_Output/dW_Output
  //
  // About calculating dLoss/dZ_Output:
  // We are implicitly using Cross-Entropy loss here.
  // When using softmax, the derivative of the C-E loss function with respect to the output layer's weighted sum, Z_Output,
  // is obtained from a complicated calculation (http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/),
  // but in the end it simplifies to just the difference between the output and the expected output. Trust me bro, it's true.
  // For this reason we don't have to calculate anything except:
  const dL_dZOutput = mSub(A_Output, expected);

  // For dZ_Output/dW_Output, we can expand Z_Output to = W_Output * A_Prev + B_Output
  // So: dZ_Output/dW_Output = A_Prev
  let A_Prev = hiddenLayers[hiddenLayers.length - 1][1];

  // So: dLoss/dW_Output = dLoss/dZ_Output * A_Prev
  dW.unshift(mSD(mMul(dL_dZOutput, T(A_Prev)), batchSize)); // divide by batch size to get the average

  // dLoss/dB_Output = dLoss/dZ_Output * dZ_Output/dB_Output
  // For dZ_Output/dB_Output, we can expand Z_Output to = W_Output * A_Prev + B_Output
  // So: dZ_Output/dB_Output = 1
  // So: dLoss/dB_Output = dLoss/dZ_Output * 1
  dB.unshift(mSD(dL_dZOutput, batchSize)); // divide by batch size to get the average

  // By 'next', we mean the layer that is closer to the output layer. We are going backwards through the layers.
  let dZNext = dL_dZOutput;

  for (let i = hiddenLayers.length - 1; i >= 0; i--) {
    // For the other layers, we can use the same process as above, but we need to use the derivative of the activation function.
    // Again using the chain rule:
    // dLoss/dZ_Hidden = dLoss/dZ_Next * dZ_Next/dA_Hidden * dA_Hidden/dZ_Hidden

    // We already have dLoss/dZNext in the variable: dZNext.
    dZNext = dZNext;

    // For dZ_Next/dA_Hidden, we can expand Z_Next to = W_Next * A_Hidden + B_Next
    // So: dZ_Next/dA_Hidden = W_Next
    const W_Next = weights[i + 1];

    // For dA_Hidden/dZ_Hidden, we can use the derivative of the activation function.
    // Since A_Hidden = relu(Z_Hidden):
    // dA_Hidden/dZ_Hidden = relu'(Z_Hidden) // technically it's relu'(Z_Hidden) * 1 because of the chain rule f'(g(x)) * g'(x), but we can ignore the 1
    // So: dLoss/dZ_Hidden = dZNext * W_Next * relu'(Z_Hidden)
    const [Z_Hidden] = hiddenLayers[i];
    const reluDerivZHidden = mMap(Z_Hidden, reluDerivative);
    const dZ_Hidden = mHad(reluDerivZHidden, mMul(T(W_Next), dZNext));

    // So: dLoss/dW_Hidden = dLoss/dZ_Hidden * A_Prev
    A_Prev = i === 0 ? A_Input : hiddenLayers[i - 1][1];
    dW.unshift(mSD(mMul(dZ_Hidden, T(A_Prev)), batchSize)); // divide by batch size to get the average

    // dLoss/dB_Hidden = dLoss/dZ_Next * dZ_Next/dA_Hidden * dA_Hidden/dZ_Hidden * dZ_Hidden/dB_Hidden
    // For dZ_Hidden/dB_Hidden, we can expand Z_Hidden to = W_Hidden * A_Prev + B_Hidden
    // So: dZ_Hidden/dB_Hidden = 1
    // So: dLoss/dB_Hidden = dLoss/dZ_Hidden * 1

    dB.unshift(mSD(dZ_Hidden, batchSize)); // divide by batch size to get the average

    // For the next iteration, we want to calculate dLoss/dZ_Prev, so we set dZNext to dZ_Hidden.
    dZNext = dZ_Hidden;
  }

  return {
    dWs: dW,
    dBs: dB,
  };
}

export function step(
  params: Network["params"],
  dWs: Matrix[],
  dBs: Matrix[],
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
  params: Network["params"]
) {
  const correct = test.reduce((acc, { input, output }) => {
    const weightedSumsAndActivations = forwardPropagate([input], params);
    const a2 =
      weightedSumsAndActivations[weightedSumsAndActivations.length - 1][1];
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

export function gradientDescent(model: Network, props: GradientDescentParams) {
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
      const weightedSumsAndActivations = forwardPropagate(input, model.params);

      const expected = batch.map((d) => d.output);
      const deltaWeightsAndBiases = backPropagate(
        weightedSumsAndActivations,
        T(expected),
        {
          weights: model.params.weights,
          biases: model.params.biases,
        }
      );
      step(
        model.params,
        deltaWeightsAndBiases.dWs,
        deltaWeightsAndBiases.dBs,
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
