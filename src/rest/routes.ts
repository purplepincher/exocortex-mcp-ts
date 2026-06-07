/**
 * REST routes — mirrors all MCP tools as HTTP endpoints.
 *
 * Uses Node's built-in http module. Zero runtime dependencies.
 */

import { IncomingMessage, ServerResponse } from 'http';
import { handleToolCall } from '../mcp/tools';
import { toolDefinitions } from '../mcp/tools';
import { DataPoint } from '../types';

/**
 * Parse JSON body from a request.
 */
function parseBody(req: IncomingMessage): Promise<any> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk: string) => { body += chunk; });
    req.on('end', () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (e) {
        reject(new Error('Invalid JSON'));
      }
    });
    req.on('error', reject);
  });
}

/**
 * Send a JSON response.
 */
function sendJson(res: ServerResponse, statusCode: number, data: any): void {
  const body = JSON.stringify(data);
  res.writeHead(statusCode, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

/**
 * Route a request to the appropriate handler.
 */
export async function routeRequest(
  method: string,
  url: string,
  req: IncomingMessage,
  res: ServerResponse
): Promise<void> {
  try {
    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    const path = url.split('?')[0];

    // Health check
    if (method === 'GET' && path === '/health') {
      sendJson(res, 200, {
        status: 'ok',
        version: '1.0.0',
        uptime: process.uptime(),
        kernels: ['micro-nn', 'logistic', 'kmeans'],
      });
      return;
    }

    // Tool discovery
    if (method === 'GET' && path === '/tools') {
      sendJson(res, 200, { tools: toolDefinitions });
      return;
    }

    // POST endpoints
    if (method !== 'POST') {
      sendJson(res, 405, { error: 'Method not allowed' });
      return;
    }

    const body = await parseBody(req);

    switch (path) {
      case '/query': {
        const result = await handleToolCall({
          name: 'notebook_query',
          arguments: { question: body.question, topK: body.topK || 5 },
        });
        sendJson(res, 200, JSON.parse(result.content[0].text));
        break;
      }

      case '/embed': {
        const result = await handleToolCall({
          name: 'notebook_embed',
          arguments: { texts: body.texts, dimensions: body.dimensions || 64 },
        });
        sendJson(res, 200, JSON.parse(result.content[0].text));
        break;
      }

      case '/train': {
        const result = await handleToolCall({
          name: 'notebook_train',
          arguments: {
            algorithm: body.algorithm,
            data: body.data,
            labels: body.labels,
            epochs: body.epochs,
            learningRate: body.learningRate,
            k: body.k,
          },
        });
        sendJson(res, 200, JSON.parse(result.content[0].text));
        break;
      }

      case '/predict': {
        const result = await handleToolCall({
          name: 'notebook_predict',
          arguments: { algorithm: body.algorithm, input: body.input },
        });
        sendJson(res, 200, JSON.parse(result.content[0].text));
        break;
      }

      case '/analyze': {
        const result = await handleToolCall({
          name: 'notebook_analyze',
          arguments: { data: body.data },
        });
        sendJson(res, 200, JSON.parse(result.content[0].text));
        break;
      }

      case '/cluster': {
        const result = await handleToolCall({
          name: 'notebook_cluster',
          arguments: { data: body.data, k: body.k, maxIterations: body.maxIterations },
        });
        sendJson(res, 200, JSON.parse(result.content[0].text));
        break;
      }

      case '/remember': {
        const result = await handleToolCall({
          name: 'notebook_remember',
          arguments: { id: body.id, text: body.text, tags: body.tags, confidence: body.confidence },
        });
        sendJson(res, 200, JSON.parse(result.content[0].text));
        break;
      }

      case '/recall': {
        const result = await handleToolCall({
          name: 'notebook_recall',
          arguments: { query: body.query, topK: body.topK },
        });
        sendJson(res, 200, JSON.parse(result.content[0].text));
        break;
      }

      default:
        sendJson(res, 404, { error: `Not found: ${path}` });
    }
  } catch (err: any) {
    sendJson(res, 500, { error: err.message || 'Internal server error' });
  }
}
