import * as fs from "fs";
import * as path from "path";

const MNIST_TRAINING_DATA_FILENAME = path.resolve(__dirname, "mnist_train.csv");
const MNIST_TEST_DATA_FILENAME = path.resolve(__dirname, "mnist_test.csv");

export function parseDataset(filename: string) {
  const data = fs.readFileSync(filename, "utf8");
  const lines = data.split("\n");

  const pixelValues = lines.slice(1, lines.length - 1).map((line) => {
    const values = line.split(",");
    return values.map((value, i) =>
      // First value is the label, so we skip it
      // Normalize the pixel values
      i == 0 ? parseInt(value) : parseInt(value) / 255 - 0.5
    );
  });

  return pixelValues.map((pixelValue) => {
    const output = pixelValue[0];
    const input = pixelValue.slice(1);

    return {
      input,
      output: Array(10)
        .fill(0)
        .map((_, i) => (i == output ? 1 : 0)),
    };
  });
}

export const mnistTrainingSet = parseDataset(MNIST_TRAINING_DATA_FILENAME);

export const mnistTestSet = parseDataset(MNIST_TEST_DATA_FILENAME);
