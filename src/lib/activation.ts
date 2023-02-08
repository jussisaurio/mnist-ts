export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export const softmax = (x: number[]) => {
  const exps = x.map((x) => Math.exp(x));
  const sum = exps.reduce((a, b) => a + b);
  return exps.map((x) => x / sum);
};

export const relu = (x: number) => {
  return Math.max(0, x);
};

export const reluDerivative = (x: number) => {
  return x > 0 ? 1 : 0;
};
