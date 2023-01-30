export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export const relu = (x: number) => {
  return Math.max(0, x);
};

export const reluDerivative = (x: number) => {
  return x > 0 ? 1 : 0;
};
