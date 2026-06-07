/**
 * Tests for MCP server — protocol parsing, tool dispatch.
 */

import { handleRequest, parseRequest } from '../src/mcp/server';
import { makeError, makeResult, JSONRPCRequest } from '../src/mcp/protocol';
import { handleToolCall } from '../src/mcp/tools';

describe('MCP Protocol', () => {
  test('parseRequest handles valid JSON-RPC', () => {
    const req = parseRequest('{"jsonrpc":"2.0","id":1,"method":"ping"}');
    expect(req).toBeTruthy();
    expect(req!.method).toBe('ping');
    expect(req!.id).toBe(1);
  });

  test('parseRequest returns null for invalid JSON', () => {
    expect(parseRequest('not json')).toBeNull();
  });

  test('parseRequest returns null for non-JSON-RPC', () => {
    expect(parseRequest('{"foo":"bar"}')).toBeNull();
  });

  test('makeError creates proper error response', () => {
    const resp = makeError(1, -32600, 'Invalid request');
    expect(resp.jsonrpc).toBe('2.0');
    expect(resp.id).toBe(1);
    expect(resp.error?.code).toBe(-32600);
  });

  test('makeResult creates proper result response', () => {
    const resp = makeResult(2, { ok: true });
    expect(resp.jsonrpc).toBe('2.0');
    expect(resp.result).toEqual({ ok: true });
  });
});

describe('MCP Server', () => {
  test('initialize returns server info', async () => {
    const resp = await handleRequest({
      jsonrpc: '2.0', id: 1, method: 'initialize',
    });
    expect(resp.result).toBeDefined();
    const result = resp.result as any;
    expect(result.serverInfo.name).toBe('exocortex-mcp-ts');
    expect(result.protocolVersion).toBe('2024-11-05');
  });

  test('tools/list returns tool definitions', async () => {
    const resp = await handleRequest({
      jsonrpc: '2.0', id: 2, method: 'tools/list',
    });
    const result = resp.result as any;
    expect(result.tools.length).toBe(8);
    expect(result.tools[0].name).toBe('notebook_query');
  });

  test('ping returns empty result', async () => {
    const resp = await handleRequest({
      jsonrpc: '2.0', id: 3, method: 'ping',
    });
    expect(resp.result).toEqual({});
  });

  test('unknown method returns error', async () => {
    const resp = await handleRequest({
      jsonrpc: '2.0', id: 4, method: 'nonexistent',
    });
    expect(resp.error).toBeDefined();
    expect(resp.error!.code).toBe(-32601);
  });

  test('tools/call dispatches notebook_analyze', async () => {
    const resp = await handleRequest({
      jsonrpc: '2.0', id: 5, method: 'tools/call',
      params: { name: 'notebook_analyze', arguments: { data: [[1, 2], [3, 4], [5, 6]] } },
    });
    expect(resp.result).toBeDefined();
    const result = resp.result as any;
    const parsed = JSON.parse(result.content[0].text);
    expect(parsed.sampleCount).toBe(3);
    expect(parsed.dimensions).toBe(2);
  });
});

describe('MCP Tools', () => {
  test('notebook_remember and notebook_recall', async () => {
    await handleToolCall({
      name: 'notebook_remember',
      arguments: { id: 'test-1', text: 'TypeScript is great for type safety', tags: ['typescript', 'types'] },
    });

    const result = await handleToolCall({
      name: 'notebook_recall',
      arguments: { query: 'type safety in TypeScript', topK: 5 },
    });

    expect(result.isError).toBeFalsy();
    const parsed = JSON.parse(result.content[0].text);
    expect(parsed.insights.length).toBeGreaterThan(0);
    expect(parsed.insights[0].id).toBe('test-1');
  });

  test('notebook_embed creates embeddings', async () => {
    const result = await handleToolCall({
      name: 'notebook_embed',
      arguments: { texts: ['hello world', 'foo bar'], dimensions: 32 },
    });

    const parsed = JSON.parse(result.content[0].text);
    expect(parsed.embeddings.length).toBe(2);
    expect(parsed.embeddings[0].length).toBe(32);
    expect(parsed.dimensions).toBe(32);
  });

  test('notebook_train and notebook_predict', async () => {
    await handleToolCall({
      name: 'notebook_train',
      arguments: {
        algorithm: 'logistic',
        data: [[-5, -5], [-4, -4], [5, 5], [4, 4]],
        labels: ['A', 'A', 'B', 'B'],
      },
    });

    const result = await handleToolCall({
      name: 'notebook_predict',
      arguments: { algorithm: 'logistic', input: [-3, -3] },
    });

    const parsed = JSON.parse(result.content[0].text);
    expect(parsed).toHaveProperty('class');
    expect(parsed).toHaveProperty('confidence');
  });

  test('notebook_cluster returns centroids', async () => {
    const result = await handleToolCall({
      name: 'notebook_cluster',
      arguments: {
        data: [[0, 0], [1, 1], [20, 20], [21, 21]],
        k: 2,
      },
    });

    const parsed = JSON.parse(result.content[0].text);
    expect(parsed.centroids.length).toBe(2);
    expect(parsed.assignments.length).toBe(4);
  });

  test('unknown tool returns error', async () => {
    const result = await handleToolCall({
      name: 'notebook_nonexistent',
      arguments: {},
    });
    expect(result.isError).toBe(true);
  });
});
