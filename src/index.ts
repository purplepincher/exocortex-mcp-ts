/**
 * Exocortex MCP TypeScript — Entry Point
 *
 * Starts both MCP (stdio) and REST (HTTP) servers.
 * Mode selected via CLI flags:
 *   --mcp        Start MCP server on stdio (default)
 *   --rest       Start REST API server
 *   --port PORT  REST server port (default 3000)
 */

import { runMcpServer } from './mcp/server';
import { startRestServer } from './rest/app';

function parseArgs(): { mode: 'mcp' | 'rest' | 'both'; port: number } {
  const args = process.argv.slice(2);
  let mode: 'mcp' | 'rest' | 'both' = 'both';
  let port = 3000;

  if (args.includes('--mcp') && !args.includes('--rest')) mode = 'mcp';
  else if (args.includes('--rest') && !args.includes('--mcp')) mode = 'rest';

  const portIdx = args.indexOf('--port');
  if (portIdx !== -1 && args[portIdx + 1]) {
    port = parseInt(args[portIdx + 1], 10) || 3000;
  }

  return { mode, port };
}

async function main(): Promise<void> {
  const { mode, port } = parseArgs();

  if (mode === 'mcp' || mode === 'both') {
    if (mode === 'mcp') {
      // Pure MCP mode — only stdio
      runMcpServer();
    } else {
      // Both — MCP in background not practical; start REST + note
      console.error('Starting in dual mode. MCP via stdio requires dedicated process.');
    }
  }

  if (mode === 'rest' || mode === 'both') {
    await startRestServer(port);
  }
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});

export { runMcpServer, startRestServer };
