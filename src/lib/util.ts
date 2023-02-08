export function getRandomSample<T extends any>(arr: T[], size: number) {
  const sample = [];
  for (let i = 0; i < size; i++) {
    const index = Math.floor(Math.random() * arr.length);
    sample.push(arr[index]);
  }
  return sample;
}
