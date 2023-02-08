export const matrixMultiply = (a: number[][], b: number[][]) => {
  const result: number[][] = [];

  //   console.log(
  //     `Multiplying matrix of size ${a[0].length}x${a.length} with matrix of size ${b[0].length}x${b.length}`
  //   );

  for (let i = 0; i < a.length; i++) {
    result[i] = [];

    for (let j = 0; j < b[0].length; j++) {
      let sum = 0;

      for (let k = 0; k < a[0].length; k++) {
        sum += a[i][k] * b[k][j];
      }

      result[i][j] = sum;
    }
  }

  // Example:
  // a = [
  //   [1, 2],
  //   [3, 4],
  //   [5, 6],
  // ];
  // b = [
  //   [7, 8, 9],
  //   [10, 11, 12],
  // ];
  // result = [
  //   [27, 30, 33],
  //   [61, 68, 75],
  //   [95, 106, 117],
  // ];
  // "b's columns, a's rows"
  // 7*1 + 10*3 + 13*5 = 27
  // 10*1 + 11*3 + 12*5 = 30
  // = [27, 30, 33]. etc

  return result;
};

export const mMul = matrixMultiply;

export const matrixTranspose = (a: number[][]) => {
  const result: number[][] = [];

  for (let i = 0; i < a[0].length; i++) {
    result[i] = [];

    for (let j = 0; j < a.length; j++) {
      result[i][j] = a[j][i];
    }
  }

  return result;
};

export const T = matrixTranspose;

export const matrixSubtract = (a: number[][], b: number[][]) => {
  const result: number[][] = [];

  for (let i = 0; i < a.length; i++) {
    result[i] = [];

    for (let j = 0; j < a[0].length; j++) {
      result[i][j] = a[i][j] - b[i][j];
    }
  }

  return result;
};

export const mSub = matrixSubtract;

export const matrixScalarMultiply = (a: number[][], b: number) => {
  const result: number[][] = [];

  for (let i = 0; i < a.length; i++) {
    result[i] = [];

    for (let j = 0; j < a[0].length; j++) {
      result[i][j] = a[i][j] * b;
    }
  }

  return result;
};

export const mSM = matrixScalarMultiply;

export const matrixScalarDivide = (a: number[][], b: number) => {
  const result: number[][] = [];

  for (let i = 0; i < a.length; i++) {
    result[i] = [];

    for (let j = 0; j < a[0].length; j++) {
      result[i][j] = a[i][j] / b;
    }
  }

  return result;
};

export const mSD = matrixScalarDivide;

export const matrixOf = ({
  rows,
  cols,
  initialValue = () => 0,
}: {
  rows: number;
  cols: number;
  initialValue: () => number;
}) =>
  Array(rows)
    .fill(0)
    .map(() => Array(cols).fill(0).map(initialValue));

export const getShape = (a: number[][]) => [a.length, a[0].length];
