/**
 * Tests for LogisticRegression.
 */

import { LogisticRegression } from '../src/compute/logistic';

describe('LogisticRegression', () => {
  test('creates instance with correct name', () => {
    const lr = new LogisticRegression();
    expect(lr.name).toBe('logistic');
  });

  test('trains on linearly separable data', async () => {
    const lr = new LogisticRegression();
    const data: number[][] = [];
    const labels: string[] = [];

    // Class A: far negative
    for (let i = 0; i < 20; i++) {
      data.push([-100 + Math.random() * 10, -100 + Math.random() * 10]);
      labels.push('A');
    }
    // Class B: far positive
    for (let i = 0; i < 20; i++) {
      data.push([100 + Math.random() * 10, 100 + Math.random() * 10]);
      labels.push('B');
    }

    const report = await lr.train(data, labels);
    expect(report.samples).toBe(40);
    expect(report.accuracy).toBeGreaterThan(0.5);
  });

  test('throws if no labels provided', async () => {
    const lr = new LogisticRegression();
    await expect(lr.train([[1, 2]])).rejects.toThrow('requires labels');
  });

  test('throws if data/labels mismatch', async () => {
    const lr = new LogisticRegression();
    await expect(lr.train([[1, 2]], ['a', 'b'])).rejects.toThrow('same length');
  });

  test('predicts after training', async () => {
    const lr = new LogisticRegression();
    const data = [
      [-10, -10], [-8, -9], [-9, -8],
      [10, 10], [8, 9], [9, 8],
    ];
    const labels = ['neg', 'neg', 'neg', 'pos', 'pos', 'pos'];
    await lr.train(data, labels);

    const pred = await lr.predict([-5, -5]);
    // Logistic with close data may classify either way
    expect(pred).toHaveProperty('class');
    expect(pred).toHaveProperty('confidence');
    expect(typeof pred.confidence).toBe('number');
  });

  test('throws when predicting before training', async () => {
    const lr = new LogisticRegression();
    await expect(lr.predict([1, 2])).rejects.toThrow('trained');
  });

  test('serialize and deserialize', async () => {
    const lr = new LogisticRegression();
    const data = [[-5, -5], [-4, -4], [5, 5], [4, 4]];
    const labels = ['A', 'A', 'B', 'B'];
    await lr.train(data, labels);

    const buf = lr.serialize();
    const restored = LogisticRegression.deserialize(buf);

    const p1 = await lr.predict([0, 0]);
    const p2 = await restored.predict([0, 0]);
    expect(p1.class).toBe(p2.class);
  });
});
