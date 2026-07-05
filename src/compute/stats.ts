/**
 * Statistical utility functions for the compute layer.
 * Pure TypeScript — no external dependencies.
 */

/**
 * Arithmetic mean of a number array.
 */
export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += values[i];
  return sum / values.length;
}

/**
 * Population variance of a number array.
 */
export function variance(values: number[]): number {
  if (values.length === 0) return 0;
  const m = mean(values);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const d = values[i] - m;
    sum += d * d;
  }
  return sum / values.length;
}

/**
 * Standardize each column of a 2D dataset to zero mean, unit variance.
 */
export function standardize(data: number[][]): number[][] {
  if (data.length === 0) return [];
  const cols = data[0].length;
  const result: number[][] = [];

  for (let i = 0; i < data.length; i++) {
    result.push(new Array(cols));
  }

  for (let c = 0; c < cols; c++) {
    const colValues: number[] = [];
    for (let r = 0; r < data.length; r++) colValues.push(data[r][c]);
    const m = mean(colValues);
    const v = variance(colValues);
    const std = Math.sqrt(v) || 1;
    for (let r = 0; r < data.length; r++) {
      result[r][c] = (data[r][c] - m) / std;
    }
  }

  return result;
}

/**
 * Euclidean distance between two vectors.
 */
export function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/**
 * Softmax converts a logit array to a probability distribution.
 */
export function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

/**
 * Sigmoid activation.
 */
export function sigmoid(x: number): number {
  if (x < -45) return 0;
  if (x > 45) return 1;
  return 1 / (1 + Math.exp(-x));
}

/**
 * ReLU activation.
 */
export function relu(x: number): number {
  return Math.max(0, x);
}

/**
 * ReLU derivative.
 */
export function reluDerivative(x: number): number {
  return x > 0 ? 1 : 0;
}

/**
 * Dot product of two vectors.
 */
export function dot(a: number[], b: number[]): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) sum += a[i] * b[i];
  return sum;
}

/**
 * Create a zero-filled 2D matrix.
 */
export function zeros2d(rows: number, cols: number): number[][] {
  const result: number[][] = [];
  for (let r = 0; r < rows; r++) {
    result.push(new Array(cols).fill(0));
  }
  return result;
}

/**
 * Random number from standard normal distribution (Box-Muller).
 */
export function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Column-wise means of a 2D array.
 */
export function columnMeans(data: number[][]): number[] {
  if (data.length === 0) return [];
  const cols = data[0].length;
  const means: number[] = [];
  for (let c = 0; c < cols; c++) {
    let sum = 0;
    for (let r = 0; r < data.length; r++) sum += data[r][c];
    means.push(sum / data.length);
  }
  return means;
}

/**
 * Column-wise variances of a 2D array.
 */
export function columnVariances(data: number[][]): number[] {
  if (data.length === 0) return [];
  const cols = data[0].length;
  const means = columnMeans(data);
  const vars: number[] = [];
  for (let c = 0; c < cols; c++) {
    let sum = 0;
    for (let r = 0; r < data.length; r++) {
      const d = data[r][c] - means[c];
      sum += d * d;
    }
    vars.push(sum / data.length);
  }
  return vars;
}
