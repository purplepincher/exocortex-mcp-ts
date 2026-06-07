/**
 * MCP tool definitions and handlers.
 *
 * All exocortex operations exposed as MCP tools:
 * notebook_query, notebook_embed, notebook_train, notebook_predict,
 * notebook_analyze, notebook_cluster, notebook_remember, notebook_recall.
 */

import { Tool, ToolCall, ToolResult } from './protocol';
import { MicroNN } from '../compute/micro-nn';
import { LogisticRegression } from '../compute/logistic';
import { KMeans } from '../compute/kmeans';
import { InsightStore } from '../memory/insight';
import { standardize, columnMeans, columnVariances } from '../compute/stats';
import { EmbeddingIndex, embedText, cosineSimilarity } from '../memory/embedding';
import { ComputeKernel, DataPoint, Insight } from '../types';

// Singleton state
const kernels: Map<string, ComputeKernel> = new Map();
const insightStore = new InsightStore(64);
const embeddingIndex = new EmbeddingIndex(64);

// ─── Tool Definitions ────────────────────────────────────────────

export const toolDefinitions: Tool[] = [
  {
    name: 'notebook_query',
    description: 'Query the notebook memory for relevant insights',
    inputSchema: {
      type: 'object',
      properties: {
        question: { type: 'string', description: 'Natural language query' },
        topK: { type: 'number', description: 'Number of results (default 5)' },
      },
      required: ['question'],
    },
  },
  {
    name: 'notebook_embed',
    description: 'Create embeddings for texts using random projection',
    inputSchema: {
      type: 'object',
      properties: {
        texts: { type: 'array', items: { type: 'string' }, description: 'Texts to embed' },
        dimensions: { type: 'number', description: 'Embedding dimensions (default 64)' },
      },
      required: ['texts'],
    },
  },
  {
    name: 'notebook_train',
    description: 'Train a compute kernel on labelled data',
    inputSchema: {
      type: 'object',
      properties: {
        algorithm: { type: 'string', enum: ['micro-nn', 'logistic', 'kmeans'] },
        data: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
        labels: { type: 'array', items: { type: 'string' } },
        epochs: { type: 'number' },
        learningRate: { type: 'number' },
        k: { type: 'number' },
      },
      required: ['algorithm', 'data'],
    },
  },
  {
    name: 'notebook_predict',
    description: 'Run inference with a trained model',
    inputSchema: {
      type: 'object',
      properties: {
        algorithm: { type: 'string', enum: ['micro-nn', 'logistic', 'kmeans'] },
        input: { type: 'array', items: { type: 'number' } },
      },
      required: ['algorithm', 'input'],
    },
  },
  {
    name: 'notebook_analyze',
    description: 'Compute statistics over a dataset',
    inputSchema: {
      type: 'object',
      properties: {
        data: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
      },
      required: ['data'],
    },
  },
  {
    name: 'notebook_cluster',
    description: 'Cluster data points using K-means',
    inputSchema: {
      type: 'object',
      properties: {
        data: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
        k: { type: 'number' },
        maxIterations: { type: 'number' },
      },
      required: ['data', 'k'],
    },
  },
  {
    name: 'notebook_remember',
    description: 'Store an insight in notebook memory',
    inputSchema: {
      type: 'object',
      properties: {
        id: { type: 'string' },
        text: { type: 'string' },
        tags: { type: 'array', items: { type: 'string' } },
        confidence: { type: 'number' },
      },
      required: ['id', 'text'],
    },
  },
  {
    name: 'notebook_recall',
    description: 'Recall insights from notebook memory',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string' },
        topK: { type: 'number' },
      },
      required: ['query'],
    },
  },
];

// ─── Tool Handlers ───────────────────────────────────────────────

function makeTextResult(text: string): ToolResult {
  return { content: [{ type: 'text', text }] };
}

function makeErrorResult(message: string): ToolResult {
  return { content: [{ type: 'text', text: `Error: ${message}` }], isError: true };
}

function getOrCreateKernel(algorithm: string): ComputeKernel {
  if (!kernels.has(algorithm)) {
    switch (algorithm) {
      case 'micro-nn': kernels.set(algorithm, new MicroNN()); break;
      case 'logistic': kernels.set(algorithm, new LogisticRegression()); break;
      case 'kmeans': kernels.set(algorithm, new KMeans()); break;
      default: throw new Error(`Unknown algorithm: ${algorithm}`);
    }
  }
  return kernels.get(algorithm)!;
}

export async function handleToolCall(call: ToolCall): Promise<ToolResult> {
  try {
    const args = call.arguments;

    switch (call.name) {
      case 'notebook_query': {
        const question = args.question as string;
        const topK = (args.topK as number) || 5;
        const insights = await insightStore.recall(question, topK);
        return makeTextResult(JSON.stringify({ insights, query: question }));
      }

      case 'notebook_embed': {
        const texts = args.texts as string[];
        const dimensions = (args.dimensions as number) || 64;
        const vectors = texts.map(t => embedText(t, dimensions));
        return makeTextResult(JSON.stringify({ embeddings: vectors, dimensions }));
      }

      case 'notebook_train': {
        const algorithm = args.algorithm as string;
        const data = args.data as DataPoint[];
        const labels = args.labels as string[] | undefined;
        const kernel = getOrCreateKernel(algorithm);
        const report = await kernel.train(data, labels);
        return makeTextResult(JSON.stringify(report));
      }

      case 'notebook_predict': {
        const algorithm = args.algorithm as string;
        const input = args.input as DataPoint;
        const kernel = getOrCreateKernel(algorithm);
        const prediction = await kernel.predict(input);
        return makeTextResult(JSON.stringify(prediction));
      }

      case 'notebook_analyze': {
        const data = args.data as DataPoint[];
        const result = {
          mean: columnMeans(data),
          variance: columnVariances(data),
          standardized: standardize(data),
          dimensions: data[0]?.length || 0,
          sampleCount: data.length,
        };
        return makeTextResult(JSON.stringify(result));
      }

      case 'notebook_cluster': {
        const data = args.data as DataPoint[];
        const k = args.k as number;
        const maxIterations = args.maxIterations as number | undefined;
        const km = new KMeans();
        km.setK(k);
        const result = km.cluster(data, k, maxIterations);
        return makeTextResult(JSON.stringify(result));
      }

      case 'notebook_remember': {
        const insight: Insight = {
          id: args.id as string,
          text: args.text as string,
          tags: (args.tags as string[]) || [],
          confidence: (args.confidence as number) || 1.0,
          timestamp: Date.now(),
        };
        await insightStore.remember(insight);
        return makeTextResult(JSON.stringify({ ok: true, id: insight.id }));
      }

      case 'notebook_recall': {
        const query = args.query as string;
        const topK = (args.topK as number) || 5;
        const insights = await insightStore.recall(query, topK);
        return makeTextResult(JSON.stringify({ insights }));
      }

      default:
        return makeErrorResult(`Unknown tool: ${call.name}`);
    }
  } catch (err: any) {
    return makeErrorResult(err.message || 'Internal error');
  }
}

// Export for testing
export { kernels, insightStore, embeddingIndex };
