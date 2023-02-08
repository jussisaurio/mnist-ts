# Neural network from scratch in TypeScript

Zero-dependency neural network trained to recognize handwritten digits from the MNIST dataset. The stored src/mnist-model.json file has an accuracy of 94.8% on the test set. The training performance is not very good. :)

## Install

```bash
npm i
```

Download the MNIST dataset from [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and extract it like so:

```bash
src/dataset/mnist_train.csv
src/dataset/mnist_test.csv
```

### Train

```bash
npm run mnist:train <epochs> <learning_rate> <batch_size>
```

#### Retrain existing model

```bash
USE_EXISTING=true npm run mnist:train <epochs> <learning_rate> <batch_size>
```

### Test

```bash
npm run mnist:test
```
