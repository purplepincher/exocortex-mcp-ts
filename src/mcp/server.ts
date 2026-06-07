/**
 * MCP Server — JSON-RPC 2.0 over stdio.
 *
 * Reads JSON-RPC requests from stdin, dispatches to tool handlers,
 * and writes JSON-RPC responses to stdout.
 */

import { JSONRPCRequest, JSONRPCResponse, makeError, makeResult, METHOD_NOT_FOUND, PARSE_ERROR, INVALID_REQUEST } from './protocol';
import { toolDefinitions, handleToolCall } from './tools';

/**
 * Process a single JSON-RPC request.
 */
export async function handleRequest(request: JSONRPCRequest): Promise<JSONRPCResponse> {
  const id = request.id ?? null;
  const { method, params } = request;

  switch (method) {
    case 'initialize':
      return makeResult(id, {
        protocolVersion: '2024-11-05',
        capabilities: { tools: { listChanged: false } },
        serverInfo: { name: 'exocortex-mcp-ts', version: '1.0.0' },
      });

    case 'tools/list':
      return makeResult(id, { tools: toolDefinitions });

    case 'tools/call': {
      const toolName = params?.name as string;
      const toolArgs = (params?.arguments as Record<string, unknown>) || {};
      const result = await handleToolCall({ name: toolName, arguments: toolArgs });
      return makeResult(id, result);
    }

    case 'ping':
      return makeResult(id, {});

    default:
      return makeError(id, METHOD_NOT_FOUND, `Method not found: ${method}`);
  }
}

/**
 * Parse a line of input as a JSON-RPC request.
 */
export function parseRequest(line: string): JSONRPCRequest | null {
  try {
    const obj = JSON.parse(line);
    if (obj && typeof obj === 'object' && obj.jsonrpc === '2.0' && typeof obj.method === 'string') {
      return obj as JSONRPCRequest;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Run the MCP server on stdio.
 */
export function runMcpServer(): void {
  let buffer = '';

  process.stdin.on('data', async (chunk: Buffer) => {
    buffer += chunk.toString();

    // Process complete lines
    const lines = buffer.split('\n');
    buffer = lines.pop() || ''; // Keep incomplete line in buffer

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      const request = parseRequest(trimmed);
      if (!request) {
        const response = makeError(null, PARSE_ERROR, 'Parse error');
        process.stdout.write(JSON.stringify(response) + '\n');
        continue;
      }

      const response = await handleRequest(request);
      process.stdout.write(JSON.stringify(response) + '\n');
    }
  });

  process.stdin.on('end', () => {
    process.exit(0);
  });
}
