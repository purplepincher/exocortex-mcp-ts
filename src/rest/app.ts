/**
 * REST application — Node http server with zero runtime dependencies.
 *
 * "The web is the universal deployment target."
 */

import { createServer, IncomingMessage, ServerResponse, Server } from 'http';
import { routeRequest } from './routes';

export function createApp(): Server {
  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    routeRequest(req.method || 'GET', req.url || '/', req, res);
  });

  return server;
}

/**
 * Start the REST server on the given port.
 */
export function startRestServer(port: number = 3000): Promise<Server> {
  return new Promise((resolve) => {
    const server = createApp();
    server.listen(port, () => {
      console.log(`Exocortex REST API listening on port ${port}`);
      resolve(server);
    });
  });
}
