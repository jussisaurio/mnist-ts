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

  return result;
};

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
