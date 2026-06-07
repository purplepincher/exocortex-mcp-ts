/**
 * Tests for statistical utility functions.
 */

import { mean, variance, standardize, euclideanDistance, softmax, sigmoid, relu, reluDerivative, dot, zeros2d, randn, columnMeans, columnVariances } from '../src/compute/stats';

describe('Stats Utilities', () => {
  test('mean of empty array is 0', () => {
    expect(mean([])).toBe(0);
  });

  test('mean of [2, 4, 6] is 4', () => {
    expect(mean([2, 4, 6])).toBe(4);
  });

  test('variance of empty array is 0', () => {
    expect(variance([])).toBe(0);
  });

  test('variance of [1, 2, 3, 4, 5]', () => {
    // Population variance = 2
    expect(variance([1, 2, 3, 4, 5])).toBe(2);
  });

  test('standardize empty returns empty', () => {
    expect(standardize([])).toEqual([]);
  });

  test('standardize produces zero mean', () => {
    const data = [[1, 10], [2, 20], [3, 30]];
    const result = standardize(data);
    // Each column should have mean ≈ 0
    for (let c = 0; c < 2; c++) {
      const m = mean(result.map(r => r[c]));
      expect(Math.abs(m)).toBeLessThan(0.001);
    }
  });

  test('euclideanDistance [0,0] to [3,4] is 5', () => {
    expect(euclideanDistance([0, 0], [3, 4])).toBe(5);
  });

  test('euclideanDistance with different lengths', () => {
    expect(euclideanDistance([1, 2, 3], [1, 2])).toBe(0);
  });

  test('softmax sums to 1', () => {
    const probs = softmax([1, 2, 3]);
    const sum = probs.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(0.0001);
  });

  test('softmax handles extreme values', () => {
    const probs = softmax([1000, 1001]);
    expect(probs.length).toBe(2);
    expect(probs[0]).toBeLessThan(probs[1]);
  });

  test('sigmoid(0) is 0.5', () => {
    expect(sigmoid(0)).toBeCloseTo(0.5);
  });

  test('sigmoid of large negative is ~0', () => {
    expect(sigmoid(-100)).toBeCloseTo(0, 5);
  });

  test('sigmoid of large positive is ~1', () => {
    expect(sigmoid(100)).toBeCloseTo(1, 5);
  });

  test('relu of positive is identity', () => {
    expect(relu(5)).toBe(5);
  });

  test('relu of negative is 0', () => {
    expect(relu(-5)).toBe(0);
  });

  test('reluDerivative', () => {
    expect(reluDerivative(5)).toBe(1);
    expect(reluDerivative(-5)).toBe(0);
    expect(reluDerivative(0)).toBe(0);
  });

  test('dot product of orthogonal vectors is 0', () => {
    expect(dot([1, 0], [0, 1])).toBe(0);
  });

  test('dot product of parallel vectors', () => {
    expect(dot([2, 3], [4, 5])).toBe(23);
  });

  test('zeros2d creates correct shape', () => {
    const m = zeros2d(3, 4);
    expect(m.length).toBe(3);
    expect(m[0].length).toBe(4);
    expect(m[1][2]).toBe(0);
  });

  test('randn returns finite numbers', () => {
    for (let i = 0; i < 100; i++) {
      expect(isFinite(randn())).toBe(true);
    }
  });

  test('columnMeans of empty is empty', () => {
    expect(columnMeans([])).toEqual([]);
  });

  test('columnMeans calculates correctly', () => {
    const data = [[1, 10], [3, 30], [5, 50]];
    expect(columnMeans(data)).toEqual([3, 30]);
  });

  test('columnVariances calculates correctly', () => {
    const data = [[1], [2], [3]];
    const v = columnVariances(data);
    // Population variance of [1,2,3] = 2/3
    expect(v[0]).toBeCloseTo(2/3);
  });
});
