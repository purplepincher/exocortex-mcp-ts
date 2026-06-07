/**
 * Tests for REST API routes.
 */

import * as http from 'http';

function makeRequest(method: string, path: string, body?: any): Promise<{ statusCode: number; body: string; headers: Record<string, string | string[] | undefined> }> {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: '127.0.0.1',
      port: 9876,
      path,
      method,
      headers: { 'Content-Type': 'application/json' },
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        resolve({
          statusCode: res.statusCode || 0,
          body: data,
          headers: res.headers as Record<string, string | string[] | undefined>,
        });
      });
    });

    req.on('error', reject);
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

import { createApp } from '../src/rest/app';
import { Server } from 'http';

let server: Server;

beforeAll(async () => {
  server = createApp();
  await new Promise<void>((resolve) => {
    server.listen(9876, () => resolve());
  });
});

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

describe('REST API', () => {
  test('GET /health returns status', async () => {
    const res = await makeRequest('GET', '/health');
    expect(res.statusCode).toBe(200);
    const data = JSON.parse(res.body);
    expect(data.status).toBe('ok');
    expect(data.version).toBe('1.0.0');
    expect(data.kernels).toContain('micro-nn');
  });

  test('GET /tools returns tool list', async () => {
    const res = await makeRequest('GET', '/tools');
    expect(res.statusCode).toBe(200);
    const data = JSON.parse(res.body);
    expect(data.tools.length).toBe(8);
  });

  test('OPTIONS returns CORS headers', async () => {
    const res = await makeRequest('OPTIONS', '/health');
    expect(res.statusCode).toBe(204);
    expect(res.headers['access-control-allow-origin']).toBe('*');
  });

  test('POST /analyze returns statistics', async () => {
    const res = await makeRequest('POST', '/analyze', { data: [[1, 2], [3, 4], [5, 6]] });
    expect(res.statusCode).toBe(200);
    const data = JSON.parse(res.body);
    expect(data.sampleCount).toBe(3);
    expect(data.dimensions).toBe(2);
    expect(data.mean.length).toBe(2);
  });

  test('POST /embed returns embeddings', async () => {
    const res = await makeRequest('POST', '/embed', { texts: ['hello', 'world'], dimensions: 16 });
    expect(res.statusCode).toBe(200);
    const data = JSON.parse(res.body);
    expect(data.embeddings.length).toBe(2);
  });

  test('POST /remember stores insight', async () => {
    const res = await makeRequest('POST', '/remember', { id: 'r1', text: 'test insight', tags: ['test'] });
    expect(res.statusCode).toBe(200);
    const data = JSON.parse(res.body);
    expect(data.ok).toBe(true);
  });

  test('POST /recall retrieves insights', async () => {
    await makeRequest('POST', '/remember', { id: 'r2', text: 'machine learning models' });
    const res = await makeRequest('POST', '/recall', { query: 'machine learning', topK: 5 });
    expect(res.statusCode).toBe(200);
    const data = JSON.parse(res.body);
    expect(data.insights.length).toBeGreaterThan(0);
  });

  test('GET unknown path returns 405', async () => {
    const res = await makeRequest('GET', '/unknown');
    expect(res.statusCode).toBe(405);
  });

  test('POST unknown path returns 404', async () => {
    const res = await makeRequest('POST', '/unknown', {});
    expect(res.statusCode).toBe(404);
  });

  test('POST /cluster returns clustering result', async () => {
    const res = await makeRequest('POST', '/cluster', {
      data: [[0, 0], [1, 1], [20, 20], [21, 21]],
      k: 2,
    });
    expect(res.statusCode).toBe(200);
    const data = JSON.parse(res.body);
    expect(data.centroids.length).toBe(2);
    expect(data.assignments.length).toBe(4);
  });

  test('POST /train and /predict workflow', async () => {
    const trainRes = await makeRequest('POST', '/train', {
      algorithm: 'logistic',
      data: [[-5, -5], [-4, -4], [5, 5], [4, 4]],
      labels: ['A', 'A', 'B', 'B'],
    });
    expect(trainRes.statusCode).toBe(200);

    const predRes = await makeRequest('POST', '/predict', {
      algorithm: 'logistic',
      input: [-3, -3],
    });
    expect(predRes.statusCode).toBe(200);
    const data = JSON.parse(predRes.body);
    expect(data).toHaveProperty('class');
    expect(data).toHaveProperty('confidence');
  });
});
