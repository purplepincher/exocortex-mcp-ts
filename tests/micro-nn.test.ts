/**
 * Tests for MicroNN — 2-layer MLP.
 */

import { MicroNN } from '../src/compute/micro-nn';

describe('MicroNN', () => {
  test('creates instance with correct name', () => {
    const nn = new MicroNN();
    expect(nn.name).toBe('micro-nn');
  });

  test('creates instance with custom hidden dim', () => {
    const nn = new MicroNN(32);
    expect(nn.name).toBe('micro-nn');
  });

  test('trains on simple binary classification', async () => {
    const nn = new MicroNN(8);
    const data = [
      [0, 0], [0, 1], [1, 0],
      [5, 5], [5, 6], [6, 5],
    ];
    const labels = ['a', 'a', 'a', 'b', 'b', 'b'];
    const report = await nn.train(data, labels);

    expect(report.samples).toBe(6);
    expect(report.latencyMs).toBeGreaterThanOrEqual(0);
    expect(report.accuracy).toBeGreaterThan(0);
  });

  test('throws if no labels provided', async () => {
    const nn = new MicroNN();
    await expect(nn.train([[1, 2]])).rejects.toThrow('requires labels');
  });

  test('throws if data/labels length mismatch', async () => {
    const nn = new MicroNN();
    await expect(nn.train([[1, 2]], ['a', 'b'])).rejects.toThrow('same length');
  });

  test('predicts after training', async () => {
    const nn = new MicroNN(8);
    const data = [
      [0, 0], [0, 1], [1, 0],
      [10, 10], [10, 11], [11, 10],
    ];
    const labels = ['low', 'low', 'low', 'high', 'high', 'high'];
    await nn.train(data, labels);

    const pred = await nn.predict([0.5, 0.5]);
    expect(pred).toHaveProperty('class');
    expect(pred).toHaveProperty('confidence');
    expect(typeof pred.confidence).toBe('number');
  });

  test('throws when predicting before training', async () => {
    const nn = new MicroNN();
    await expect(nn.predict([1, 2])).rejects.toThrow('trained');
  });

  test('serialize and deserialize roundtrip', async () => {
    const nn = new MicroNN(4);
    const data = [[0, 0], [1, 1], [5, 5], [6, 6]];
    const labels = ['a', 'a', 'b', 'b'];
    await nn.train(data, labels);

    const buf = nn.serialize();
    expect(Buffer.isBuffer(buf)).toBe(true);

    const restored = MicroNN.deserialize(buf);
    expect(restored.name).toBe('micro-nn');

    const pred1 = await nn.predict([2, 2]);
    const pred2 = await restored.predict([2, 2]);
    expect(pred1.class).toBe(pred2.class);
  });

  test('handles multi-class classification', async () => {
    const nn = new MicroNN(8);
    const data = [
      [0, 0], [1, 0],
      [5, 5], [6, 5],
      [10, 10], [11, 10],
    ];
    const labels = ['a', 'a', 'b', 'b', 'c', 'c'];
    const report = await nn.train(data, labels);
    expect(report.samples).toBe(6);
  });
});
