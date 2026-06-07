/**
 * Tests for embedding and insight memory modules.
 */

import { cosineSimilarity, embedText, EmbeddingIndex } from '../src/memory/embedding';
import { InsightStore } from '../src/memory/insight';

describe('Embedding', () => {
  test('cosineSimilarity of identical vectors is 1', () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1);
  });

  test('cosineSimilarity of orthogonal vectors is 0', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
  });

  test('cosineSimilarity of opposite vectors is -1', () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1);
  });

  test('cosineSimilarity with zero vector returns 0', () => {
    expect(cosineSimilarity([0, 0], [1, 2])).toBe(0);
  });

  test('embedText returns vector of correct dimensions', () => {
    const vec = embedText('hello world', 32);
    expect(vec.length).toBe(32);
  });

  test('embedText returns normalized vector', () => {
    const vec = embedText('test embedding', 16);
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    expect(Math.abs(norm - 1)).toBeLessThan(0.001);
  });

  test('similar texts produce similar embeddings', () => {
    const v1 = embedText('machine learning algorithm', 64);
    const v2 = embedText('machine learning algorithm', 64);
    const v3 = embedText('completely unrelated cooking recipe for italian pasta', 64);

    const sim12 = cosineSimilarity(v1, v2);
    const sim13 = cosineSimilarity(v1, v3);
    // Identical texts should be more similar than unrelated ones
    expect(sim12).toBeCloseTo(1, 5);
    expect(sim12).toBeGreaterThan(sim13);
  });
});

describe('EmbeddingIndex', () => {
  test('addVector and query', () => {
    const index = new EmbeddingIndex(4);
    index.addVector('a', [1, 0, 0, 0]);
    index.addVector('b', [0, 1, 0, 0]);
    index.addVector('c', [1, 0, 0, 0]);

    const results = index.query([1, 0, 0, 0], 2);
    expect(results.length).toBe(2);
    expect(results[0].similarity).toBeCloseTo(1);
  });

  test('addText and queryByText', () => {
    const index = new EmbeddingIndex(32);
    index.addText('1', 'machine learning');
    index.addText('2', 'deep neural networks');
    index.addText('3', 'cooking recipes');

    const results = index.queryByText('ML algorithms', 3);
    expect(results.length).toBe(3);
  });

  test('size tracks entries', () => {
    const index = new EmbeddingIndex(4);
    expect(index.size).toBe(0);
    index.addVector('a', [1, 0]);
    expect(index.size).toBe(1);
  });

  test('clear empties the index', () => {
    const index = new EmbeddingIndex(4);
    index.addVector('a', [1, 0]);
    index.clear();
    expect(index.size).toBe(0);
  });
});

describe('InsightStore', () => {
  test('remember and recall', async () => {
    const store = new InsightStore(32);
    await store.remember({
      id: 'i1',
      text: 'TypeScript provides static type checking',
      tags: ['typescript', 'types'],
      confidence: 0.9,
      timestamp: Date.now(),
    });

    const results = await store.recall('type checking', 5);
    expect(results.length).toBe(1);
    expect(results[0].id).toBe('i1');
  });

  test('getAll returns all insights', async () => {
    const store = new InsightStore(16);
    await store.remember({ id: 'a', text: 'alpha', tags: [], confidence: 1, timestamp: 1 });
    await store.remember({ id: 'b', text: 'beta', tags: [], confidence: 1, timestamp: 2 });

    expect(store.size).toBe(2);
    expect(store.getAll().length).toBe(2);
  });

  test('clear removes everything', async () => {
    const store = new InsightStore(16);
    await store.remember({ id: 'x', text: 'test', tags: [], confidence: 1, timestamp: 1 });
    store.clear();
    expect(store.size).toBe(0);
  });
});
