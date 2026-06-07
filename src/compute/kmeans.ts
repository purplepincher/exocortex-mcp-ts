/**
 * KMeans — K-means clustering with K-means++ initialization.
 */

import { ComputeKernel, DataPoint, Prediction, TrainReport, ClusterResponse } from '../types';
import { euclideanDistance, mean } from './stats';

export class KMeans implements ComputeKernel {
  readonly name = 'kmeans';

  private centroids: DataPoint[] = [];
  private k = 3;
  private trained = false;
  private iterationsUsed = 0;

  async train(data: DataPoint[], labels?: string[]): Promise<TrainReport> {
    if (data.length < this.k) {
      throw new Error(`Need at least ${this.k} data points for ${this.k} clusters`);
    }

    const start = Date.now();
    const maxIter = 100;

    // K-means++ initialization
    this.centroids = this.kmeansPlusPlusInit(data, this.k);

    let assignments = new Array(data.length).fill(0);

    for (let iter = 0; iter < maxIter; iter++) {
      // Assignment step
      const newAssignments = data.map(point => this.nearestCentroid(point));

      // Check convergence
      let converged = true;
      for (let i = 0; i < newAssignments.length; i++) {
        if (newAssignments[i] !== assignments[i]) {
          converged = false;
          break;
        }
      }

      assignments = newAssignments;
      this.iterationsUsed = iter + 1;

      if (converged) break;

      // Update step
      for (let c = 0; c < this.k; c++) {
        const members = data.filter((_, i) => assignments[i] === c);
        if (members.length > 0) {
          this.centroids[c] = this.computeCentroid(members);
        }
      }
    }

    this.trained = true;

    // Compute "accuracy" as within-cluster coherence (1 - avg distance / max distance)
    const avgDist = this.averageDistance(data, assignments);
    const maxPossibleDist = Math.sqrt(data[0].length) * 10; // rough upper bound

    return {
      accuracy: Math.max(0, 1 - avgDist / maxPossibleDist),
      samples: data.length,
      latencyMs: Date.now() - start,
    };
  }

  async predict(input: DataPoint): Promise<Prediction> {
    if (!this.trained) {
      throw new Error('Model must be trained first');
    }

    const idx = this.nearestCentroid(input);
    const dist = euclideanDistance(input, this.centroids[idx]);
    const maxDist = Math.sqrt(input.length) * 10;
    const confidence = Math.max(0, 1 - dist / maxDist);

    return {
      class: `cluster-${idx}`,
      confidence,
    };
  }

  /**
   * Run full clustering and return centroids + assignments.
   */
  cluster(data: DataPoint[], k: number, maxIterations?: number): ClusterResponse {
    this.k = k;
    // We run train synchronously-ish by calling the internal logic
    this.centroids = this.kmeansPlusPlusInit(data, k);
    let assignments = new Array(data.length).fill(0);
    const maxIter = maxIterations || 100;

    for (let iter = 0; iter < maxIter; iter++) {
      const newAssignments = data.map(point => this.nearestCentroid(point));
      let converged = true;
      for (let i = 0; i < newAssignments.length; i++) {
        if (newAssignments[i] !== assignments[i]) { converged = false; break; }
      }
      assignments = newAssignments;
      this.iterationsUsed = iter + 1;
      if (converged) break;
      for (let c = 0; c < k; c++) {
        const members = data.filter((_, i) => assignments[i] === c);
        if (members.length > 0) {
          this.centroids[c] = this.computeCentroid(members);
        }
      }
    }

    this.trained = true;
    return {
      centroids: this.centroids.map(c => [...c]),
      assignments: [...assignments],
      iterations: this.iterationsUsed,
    };
  }

  serialize(): Buffer {
    return Buffer.from(JSON.stringify({
      name: this.name,
      centroids: this.centroids,
      k: this.k,
      trained: this.trained,
    }));
  }

  static deserialize(buf: Buffer): KMeans {
    const data = JSON.parse(buf.toString());
    const km = new KMeans();
    km.centroids = data.centroids;
    km.k = data.k;
    km.trained = data.trained;
    return km;
  }

  setK(k: number): void {
    this.k = k;
  }

  // ─── Private helpers ────────────────────────────────────────

  private kmeansPlusPlusInit(data: DataPoint[], k: number): DataPoint[] {
    const centroids: DataPoint[] = [];
    // First centroid: random
    centroids.push([...data[Math.floor(Math.random() * data.length)]]);

    for (let c = 1; c < k; c++) {
      // Compute distances to nearest centroid
      const distances = data.map(point => {
        const minDist = Math.min(...centroids.map(ct => euclideanDistance(point, ct)));
        return minDist * minDist;
      });
      const totalDist = distances.reduce((a, b) => a + b, 0);

      // Weighted random selection
      let r = Math.random() * totalDist;
      for (let i = 0; i < data.length; i++) {
        r -= distances[i];
        if (r <= 0) {
          centroids.push([...data[i]]);
          break;
        }
      }
      // Fallback
      if (centroids.length <= c) {
        centroids.push([...data[Math.floor(Math.random() * data.length)]]);
      }
    }

    return centroids;
  }

  private nearestCentroid(point: DataPoint): number {
    let minDist = Infinity;
    let minIdx = 0;
    for (let i = 0; i < this.centroids.length; i++) {
      const d = euclideanDistance(point, this.centroids[i]);
      if (d < minDist) {
        minDist = d;
        minIdx = i;
      }
    }
    return minIdx;
  }

  private computeCentroid(points: DataPoint[]): DataPoint {
    if (points.length === 0) return [];
    const dim = points[0].length;
    const centroid: DataPoint = new Array(dim).fill(0);
    for (const point of points) {
      for (let i = 0; i < dim; i++) {
        centroid[i] += point[i];
      }
    }
    for (let i = 0; i < dim; i++) centroid[i] /= points.length;
    return centroid;
  }

  private averageDistance(data: DataPoint[], assignments: number[]): number {
    let totalDist = 0;
    for (let i = 0; i < data.length; i++) {
      totalDist += euclideanDistance(data[i], this.centroids[assignments[i]]);
    }
    return totalDist / data.length;
  }
}
