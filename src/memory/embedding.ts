/**
 * Embedding index using cosine similarity.
 *
 * Uses hash-based random projection for ternary embeddings:
 * fast, deterministic, and requires no ML model.
 */

import { EmbeddingVector } from '../types';

/**
 * Cosine similarity between two vectors.
 */
export function cosineSimilarity(a: EmbeddingVector, b: EmbeddingVector): number {
  let dotProd = 0;
  let normA = 0;
  let normB = 0;
  const len = Math.min(a.length, b.length);

  for (let i = 0; i < len; i++) {
    dotProd += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;
  return dotProd / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Simple hash-based random projection embedding.
 * Maps text to a fixed-dimension vector using hash functions
 * to select ternary projections (+1, 0, -1).
 */
export function embedText(text: string, dimensions: number = 64): EmbeddingVector {
  const vector = new Array(dimensions).fill(0);

  // Split into character n-grams and words
  const tokens = text.toLowerCase().split(/\s+/);

  for (const token of tokens) {
    for (let d = 0; d < dimensions; d++) {
      const hash = simpleHash(`${token}:${d}`);
      // Ternary projection: +1, 0, or -1
      if (hash % 3 === 0) vector[d] += 1;
      else if (hash % 3 === 1) vector[d] -= 1;
    }
  }

  // Normalize
  let norm = 0;
  for (let i = 0; i < dimensions; i++) norm += vector[i] * vector[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dimensions; i++) vector[i] /= norm;

  return vector;
}

/**
 * Embedding index for fast top-K retrieval.
 */
export class EmbeddingIndex {
  private entries: Array<{ id: string; vector: EmbeddingVector; text: string }> = [];
  private dimensions: number;

  constructor(dimensions: number = 64) {
    this.dimensions = dimensions;
  }

  /**
   * Add a vector to the index.
   */
  addVector(id: string, vector: EmbeddingVector, text: string = ''): void {
    this.entries.push({ id, vector, text });
  }

  /**
   * Add text by auto-embedding it.
   */
  addText(id: string, text: string): void {
    const vector = embedText(text, this.dimensions);
    this.entries.push({ id, vector, text });
  }

  /**
   * Query the index for top-K most similar entries.
   */
  query(queryVector: EmbeddingVector, topK: number = 5): Array<{ id: string; similarity: number; text: string }> {
    const scored = this.entries.map(entry => ({
      id: entry.id,
      similarity: cosineSimilarity(queryVector, entry.vector),
      text: entry.text,
    }));

    scored.sort((a, b) => b.similarity - a.similarity);
    return scored.slice(0, topK);
  }

  /**
   * Query by text (auto-embeds the query).
   */
  queryByText(query: string, topK: number = 5): Array<{ id: string; similarity: number; text: string }> {
    const vector = embedText(query, this.dimensions);
    return this.query(vector, topK);
  }

  get size(): number {
    return this.entries.length;
  }

  clear(): void {
    this.entries = [];
  }
}

/**
 * Simple deterministic hash function (djb2 variant).
 */
function simpleHash(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) & 0x7fffffff;
  }
  return hash;
}
