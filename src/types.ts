/**
 * Exocortex MCP TypeScript — Shared Types
 *
 * Core type definitions for the Exocortex notebook runtime.
 * All public types are defined here as TypeScript interfaces.
 */

// ─── Data Primitives ─────────────────────────────────────────────

/** A numeric feature vector */
export type DataPoint = number[];

/** Result of a classification/regression prediction */
export interface Prediction {
  class: string;
  confidence: number;
}

/** Training completion report */
export interface TrainReport {
  accuracy: number;
  samples: number;
  latencyMs: number;
}

/** A stored insight with metadata */
export interface Insight {
  id: string;
  text: string;
  tags: string[];
  confidence: number;
  timestamp: number;
}

/** A vector embedding */
export type EmbeddingVector = number[];

/** A labelled training sample */
export interface LabelledSample {
  data: DataPoint;
  label: string;
}

// ─── Compute Kernel ──────────────────────────────────────────────

/** Pluggable computation backend */
export interface ComputeKernel {
  readonly name: string;
  train(data: DataPoint[], labels?: string[]): Promise<TrainReport>;
  predict(input: DataPoint): Promise<Prediction>;
  serialize(): Buffer;
}

// ─── Memory Store ────────────────────────────────────────────────

/** Persistent insight storage with semantic recall */
export interface MemoryStore {
  remember(insight: Insight): Promise<void>;
  recall(query: string, topK: number): Promise<Insight[]>;
}

// ─── REST / MCP Shared ──────────────────────────────────────────

export interface TrainRequest {
  algorithm: 'micro-nn' | 'logistic' | 'kmeans';
  data: DataPoint[];
  labels?: string[];
  epochs?: number;
  learningRate?: number;
  k?: number; // for kmeans
}

export interface PredictRequest {
  algorithm: 'micro-nn' | 'logistic' | 'kmeans';
  input: DataPoint;
}

export interface ClusterRequest {
  data: DataPoint[];
  k: number;
  maxIterations?: number;
}

export interface EmbedRequest {
  texts: string[];
  dimensions?: number;
}

export interface AnalyzeRequest {
  data: DataPoint[];
}

export interface AnalyzeResponse {
  mean: number[];
  variance: number[];
  standardized: DataPoint[];
  dimensions: number;
  sampleCount: number;
}

export interface ClusterResponse {
  centroids: DataPoint[];
  assignments: number[];
  iterations: number;
}

export interface QueryRequest {
  question: string;
  topK?: number;
}

export interface QueryResponse {
  insights: Insight[];
  query: string;
}

export interface HealthResponse {
  status: 'ok' | 'degraded';
  version: string;
  uptime: number;
  kernels: string[];
}
