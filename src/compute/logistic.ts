/**
 * LogisticRegression — Binary classification via sigmoid + gradient descent.
 */

import { ComputeKernel, DataPoint, Prediction, TrainReport } from '../types';
import { sigmoid, dot } from './stats';

export class LogisticRegression implements ComputeKernel {
  readonly name = 'logistic';

  private weights: number[] = [];
  private bias = 0;
  private trained = false;
  private positiveClass = '';
  private negativeClass = '';

  async train(data: DataPoint[], labels?: string[]): Promise<TrainReport> {
    if (!labels || labels.length === 0) {
      throw new Error('LogisticRegression requires labels');
    }
    if (data.length !== labels.length) {
      throw new Error('Data and labels must have same length');
    }

    const start = Date.now();
    const dim = data[0].length;
    const uniqueLabels = [...new Set(labels)].sort();
    this.positiveClass = uniqueLabels[uniqueLabels.length - 1];
    this.negativeClass = uniqueLabels[0];

    // Binary targets
    const targets = labels.map(l => l === this.positiveClass ? 1 : 0);

    // Initialize weights
    this.weights = new Array(dim).fill(0);
    this.bias = 0;

    const epochs = 500;
    const lr = 0.05;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let s = 0; s < data.length; s++) {
        const x = data[s];
        const z = dot(this.weights, x) + this.bias;
        const pred = sigmoid(z);
        const error = targets[s] - pred;

        for (let i = 0; i < dim; i++) {
          this.weights[i] += lr * error * x[i];
        }
        this.bias += lr * error;
      }
    }

    this.trained = true;

    // Accuracy
    let correct = 0;
    for (let s = 0; s < data.length; s++) {
      const pred = await this.predict(data[s]);
      if (pred.class === labels[s]) correct++;
    }

    return {
      accuracy: correct / data.length,
      samples: data.length,
      latencyMs: Date.now() - start,
    };
  }

  async predict(input: DataPoint): Promise<Prediction> {
    if (!this.trained) {
      throw new Error('Model must be trained first');
    }

    const z = dot(this.weights, input) + this.bias;
    const prob = sigmoid(z);

    if (prob >= 0.5) {
      return { class: this.positiveClass, confidence: prob };
    } else {
      return { class: this.negativeClass, confidence: 1 - prob };
    }
  }

  serialize(): Buffer {
    return Buffer.from(JSON.stringify({
      name: this.name,
      weights: this.weights,
      bias: this.bias,
      positiveClass: this.positiveClass,
      negativeClass: this.negativeClass,
      trained: this.trained,
    }));
  }

  static deserialize(buf: Buffer): LogisticRegression {
    const data = JSON.parse(buf.toString());
    const lr = new LogisticRegression();
    lr.weights = data.weights;
    lr.bias = data.bias;
    lr.positiveClass = data.positiveClass;
    lr.negativeClass = data.negativeClass || 'other';
    lr.trained = data.trained;
    return lr;
  }
}
