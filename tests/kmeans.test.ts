/**
 * Tests for KMeans clustering.
 */

import { KMeans } from '../src/compute/kmeans';

describe('KMeans', () => {
  test('creates instance with correct name', () => {
    const km = new KMeans();
    expect(km.name).toBe('kmeans');
  });

  test('clusters simple 2D data', () => {
    const km = new KMeans();
    const data = [
      [0, 0], [1, 0], [0, 1],  // cluster 0
      [10, 10], [11, 10], [10, 11],  // cluster 1
    ];

    const result = km.cluster(data, 2);
    expect(result.centroids.length).toBe(2);
    expect(result.assignments.length).toBe(6);
    expect(result.iterations).toBeGreaterThan(0);
  });

  test('points in same group get same cluster', () => {
    const km = new KMeans();
    const data = [
      [0, 0], [0.1, 0.1], [0.2, 0],
      [20, 20], [20.1, 20.1], [20.2, 20],
    ];

    const result = km.cluster(data, 2);
    // First 3 should be same cluster, last 3 same cluster
    expect(result.assignments[0]).toBe(result.assignments[1]);
    expect(result.assignments[1]).toBe(result.assignments[2]);
    expect(result.assignments[3]).toBe(result.assignments[4]);
    expect(result.assignments[4]).toBe(result.assignments[5]);
    // And the two groups should be different
    expect(result.assignments[0]).not.toBe(result.assignments[3]);
  });

  test('throws with insufficient data', async () => {
    const km = new KMeans();
    km.setK(5);
    await expect(km.train([[1, 2], [3, 4]])).rejects.toThrow();
  });

  test('train and predict workflow', async () => {
    const km = new KMeans();
    km.setK(2);
    const data = [
      [0, 0], [1, 1], [0.5, 0.5],
      [20, 20], [21, 21], [20.5, 20.5],
    ];

    const report = await km.train(data);
    expect(report.samples).toBe(6);
    expect(report.accuracy).toBeGreaterThan(0);

    const pred = await km.predict([0.5, 0.5]);
    expect(pred).toHaveProperty('class');
    expect(pred.class).toMatch(/^cluster-\d+$/);
  });

  test('throws when predicting before training', async () => {
    const km = new KMeans();
    await expect(km.predict([1, 2])).rejects.toThrow('trained');
  });

  test('serialize and deserialize', async () => {
    const km = new KMeans();
    km.setK(2);
    const data = [[0, 0], [10, 10]];
    await km.train(data);

    const buf = km.serialize();
    const restored = KMeans.deserialize(buf);
    expect(restored.name).toBe('kmeans');
  });

  test('cluster respects maxIterations', () => {
    const km = new KMeans();
    const data = [[0, 0], [1, 1], [10, 10], [11, 11]];
    const result = km.cluster(data, 2, 2);
    expect(result.iterations).toBeLessThanOrEqual(3);
  });
});
