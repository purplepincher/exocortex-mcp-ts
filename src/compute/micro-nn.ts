/**
 * MicroNN — 2-layer MLP (Multi-Layer Perceptron).
 *
 * Architecture: input → hidden (ReLU) → output (softmax).
 * Training: Backpropagation with stochastic gradient descent.
 * Pure TypeScript math — no TensorFlow, ONNX, or external libs.
 */

import { ComputeKernel, DataPoint, Prediction, TrainReport } from '../types';
import { softmax, relu, reluDerivative, dot, zeros2d, randn, mean } from './stats';

export class MicroNN implements ComputeKernel {
  readonly name = 'micro-nn';

  private weights1: number[][] = [];  // input → hidden
  private weights2: number[][] = [];  // hidden → output
  private bias1: number[] = [];
  private bias2: number[] = [];
  private classes: string[] = [];
  private inputDim = 0;
  private hiddenDim = 16;
  private outputDim = 0;
  private trained = false;

  // Caches for backprop
  private lastHidden: number[] = [];
  private lastOutput: number[] = [];

  constructor(hiddenDim?: number) {
    if (hiddenDim) this.hiddenDim = hiddenDim;
  }

  async train(data: DataPoint[], labels?: string[]): Promise<TrainReport> {
    if (!labels || labels.length === 0) {
      throw new Error('MicroNN requires labels for training');
    }
    if (data.length !== labels.length) {
      throw new Error('Data and labels must have the same length');
    }

    const start = Date.now();
    this.inputDim = data[0].length;
    this.classes = [...new Set(labels)].sort();
    this.outputDim = this.classes.length;

    // Xavier initialization
    const scale1 = Math.sqrt(2 / this.inputDim);
    const scale2 = Math.sqrt(2 / this.hiddenDim);

    this.weights1 = zeros2d(this.inputDim, this.hiddenDim);
    this.bias1 = new Array(this.hiddenDim).fill(0);

    for (let i = 0; i < this.inputDim; i++) {
      for (let j = 0; j < this.hiddenDim; j++) {
        this.weights1[i][j] = randn() * scale1;
      }
    }

    this.weights2 = zeros2d(this.hiddenDim, this.outputDim);
    this.bias2 = new Array(this.outputDim).fill(0);

    for (let i = 0; i < this.hiddenDim; i++) {
      for (let j = 0; j < this.outputDim; j++) {
        this.weights2[i][j] = randn() * scale2;
      }
    }

    // Training loop
    const epochs = 100;
    const lr = 0.01;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let s = 0; s < data.length; s++) {
        const input = data[s];
        const targetIdx = this.classes.indexOf(labels[s]);

        // Forward pass
        const hidden = new Array(this.hiddenDim);
        for (let j = 0; j < this.hiddenDim; j++) {
          let z = this.bias1[j];
          for (let i = 0; i < this.inputDim; i++) {
            z += input[i] * this.weights1[i][j];
          }
          hidden[j] = relu(z);
        }

        const logits = new Array(this.outputDim);
        for (let k = 0; k < this.outputDim; k++) {
          let z = this.bias2[k];
          for (let j = 0; j < this.hiddenDim; j++) {
            z += hidden[j] * this.weights2[j][k];
          }
          logits[k] = z;
        }

        const probs = softmax(logits);

        // Backprop: output gradient (softmax + cross-entropy)
        const outputGrad = new Array(this.outputDim);
        for (let k = 0; k < this.outputDim; k++) {
          outputGrad[k] = probs[k] - (k === targetIdx ? 1 : 0);
        }

        // Update weights2, bias2
        for (let j = 0; j < this.hiddenDim; j++) {
          for (let k = 0; k < this.outputDim; k++) {
            this.weights2[j][k] -= lr * hidden[j] * outputGrad[k];
          }
        }
        for (let k = 0; k < this.outputDim; k++) {
          this.bias2[k] -= lr * outputGrad[k];
        }

        // Backprop: hidden gradient
        for (let j = 0; j < this.hiddenDim; j++) {
          let grad = 0;
          for (let k = 0; k < this.outputDim; k++) {
            grad += this.weights2[j][k] * outputGrad[k];
          }
          // ReLU derivative
          grad *= reluDerivative(hidden[j]);

          // Update weights1, bias1
          for (let i = 0; i < this.inputDim; i++) {
            this.weights1[i][j] -= lr * input[i] * grad;
          }
          this.bias1[j] -= lr * grad;
        }
      }
    }

    this.trained = true;

    // Compute accuracy
    let correct = 0;
    for (let s = 0; s < data.length; s++) {
      const pred = await this.predict(data[s]);
      if (pred.class === labels[s]) correct++;
    }
    const accuracy = correct / data.length;

    return {
      accuracy,
      samples: data.length,
      latencyMs: Date.now() - start,
    };
  }

  async predict(input: DataPoint): Promise<Prediction> {
    if (!this.trained) {
      throw new Error('Model must be trained before prediction');
    }

    // Forward pass
    const hidden = new Array(this.hiddenDim);
    for (let j = 0; j < this.hiddenDim; j++) {
      let z = this.bias1[j];
      for (let i = 0; i < this.inputDim; i++) {
        z += (input[i] || 0) * this.weights1[i][j];
      }
      hidden[j] = relu(z);
    }

    const logits = new Array(this.outputDim);
    for (let k = 0; k < this.outputDim; k++) {
      let z = this.bias2[k];
      for (let j = 0; j < this.hiddenDim; j++) {
        z += hidden[j] * this.weights2[j][k];
      }
      logits[k] = z;
    }

    const probs = softmax(logits);
    let maxIdx = 0;
    let maxProb = probs[0];
    for (let k = 1; k < this.outputDim; k++) {
      if (probs[k] > maxProb) {
        maxProb = probs[k];
        maxIdx = k;
      }
    }

    this.lastHidden = hidden;
    this.lastOutput = probs;

    return {
      class: this.classes[maxIdx],
      confidence: maxProb,
    };
  }

  serialize(): Buffer {
    const data = {
      name: this.name,
      inputDim: this.inputDim,
      hiddenDim: this.hiddenDim,
      outputDim: this.outputDim,
      classes: this.classes,
      weights1: this.weights1,
      weights2: this.weights2,
      bias1: this.bias1,
      bias2: this.bias2,
      trained: this.trained,
    };
    return Buffer.from(JSON.stringify(data));
  }

  static deserialize(buf: Buffer): MicroNN {
    const data = JSON.parse(buf.toString());
    const nn = new MicroNN(data.hiddenDim);
    nn.inputDim = data.inputDim;
    nn.outputDim = data.outputDim;
    nn.classes = data.classes;
    nn.weights1 = data.weights1;
    nn.weights2 = data.weights2;
    nn.bias1 = data.bias1;
    nn.bias2 = data.bias2;
    nn.trained = data.trained;
    return nn;
  }
}
