/**
 * Insight management — in-memory implementation of MemoryStore.
 */

import { Insight, MemoryStore } from '../types';
import { EmbeddingIndex } from './embedding';

export class InsightStore implements MemoryStore {
  private insights: Map<string, Insight> = new Map();
  private index: EmbeddingIndex;

  constructor(embeddingDimensions: number = 64) {
    this.index = new EmbeddingIndex(embeddingDimensions);
  }

  async remember(insight: Insight): Promise<void> {
    this.insights.set(insight.id, insight);
    this.index.addText(insight.id, `${insight.text} ${insight.tags.join(' ')}`);
  }

  async recall(query: string, topK: number): Promise<Insight[]> {
    const results = this.index.queryByText(query, topK);
    return results
      .map(r => this.insights.get(r.id))
      .filter((i): i is Insight => i !== undefined);
  }

  get size(): number {
    return this.insights.size;
  }

  clear(): void {
    this.insights.clear();
    this.index.clear();
  }

  getAll(): Insight[] {
    return Array.from(this.insights.values());
  }
}
