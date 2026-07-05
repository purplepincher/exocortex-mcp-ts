# exocortex-mcp-ts

> TypeScript implementation of the Exocortex MCP server + REST API — the web-native interface to the notebook runtime.

## Overview

**exocortex-mcp-ts** is a pure TypeScript notebook runtime exposing both an **MCP (Model Context Protocol)** server and a **REST API**. It proves the protocol is language-independent: the same operations are available via JSON-RPC 2.0 over stdio and over plain HTTP.

The runtime provides:

- **Three compute kernels** — MicroNN (2-layer MLP), Logistic Regression, and K-Means clustering
- **Semantic memory** — Hash-based random projection embeddings with cosine similarity search
- **Dual interfaces** — JSON-RPC 2.0 over stdio (MCP) and HTTP REST endpoints
- **Zero runtime dependencies** — Only TypeScript and Jest as dev dependencies
- **87 tests** across 7 test suites covering all modules

## Installation

```bash
git clone https://github.com/purplepincher/exocortex-mcp-ts.git
cd exocortex-mcp-ts
npm install
npm run build
```

## Quick Start

### REST API

Start the server:

```bash
node dist/index.js --rest --port 3000
```

Train a logistic regression classifier:

```bash
curl -X POST http://localhost:3000/train \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "logistic",
    "data": [
      [-10, -10], [-8, -9], [-9, -8], [-7, -7],
      [10, 10], [8, 9], [9, 8], [7, 7]
    ],
    "labels": ["low", "low", "low", "low", "high", "high", "high", "high"]
  }'
```

Response:

```json
{"accuracy":1,"samples":8,"latencyMs":4}
```

Make a prediction:

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"algorithm":"logistic","input":[-5,-5]}'
```

Response:

```json
{"class":"low","confidence":0.998}
```

### MCP Server

```bash
node dist/index.js --mcp
```

Send JSON-RPC requests via stdin:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize"}
```

Response:

```json
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"exocortex-mcp-ts","version":"1.0.0"}}}
```

## Runnable Examples

### Example 1: MCP Server JSON-RPC

```bash
node dist/index.js --mcp
```

```json
{"jsonrpc":"2.0","id":1,"method":"initialize"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"notebook_analyze","arguments":{"data":[[1.0,2.0],[3.0,4.0],[5.0,6.0]]}}}
```

Response for `notebook_analyze`:

```json
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"mean\":[3,4],\"variance\":[2.6667,2.6667],\"dimensions\":2,\"sampleCount\":3}"}]}}
```

### Example 2: Training a Classifier via REST API

See [Quick Start](#quick-start).

### Example 3: Embedding Search

```bash
curl -X POST http://localhost:3000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "machine learning for classification",
      "neural network architectures",
      "baking chocolate chip cookies"
    ],
    "dimensions": 32
  }'

curl -X POST http://localhost:3000/remember \
  -H "Content-Type: application/json" \
  -d '{
    "id": "ml-001",
    "text": "Random forests are an ensemble method for classification and regression",
    "tags": ["ml", "ensemble", "classification"],
    "confidence": 0.95
  }'

curl -X POST http://localhost:3000/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "classification methods", "topK": 5}'
```

### Example 4: Full Pipeline

```bash
# Step 1: Analyze
curl -X POST http://localhost:3000/analyze \
  -H "Content-Type: application/json" \
  -d '{"data": [[1,2],[2,3],[3,4],[8,9],[9,10],[10,11]]}'
# → mean: [5.5, 6.5], variance: [12.917, 12.917]

# Step 2: Cluster (k-means labels may be swapped depending on initialization)
curl -X POST http://localhost:3000/cluster \
  -H "Content-Type: application/json" \
  -d '{"data": [[1,2],[2,3],[3,4],[8,9],[9,10],[10,11]], "k": 2}'
# → two centroids near [2, 3] and [9, 10]

# Step 3: Train a classifier on the discovered groups
curl -X POST http://localhost:3000/train \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "micro-nn",
    "data": [[1,2],[2,3],[3,4],[8,9],[9,10],[10,11]],
    "labels": ["group-a","group-a","group-a","group-b","group-b","group-b"]
  }'

# Step 4: Predict
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "micro-nn", "input": [2.5, 3.5]}'
# → {"class": "group-a", "confidence": ~1.0}

# Step 5: Remember
curl -X POST http://localhost:3000/remember \
  -H "Content-Type: application/json" \
  -d '{
    "id": "pipeline-001",
    "text": "2D data naturally clusters into group-a (low) and group-b (high) with clear separation around x=5",
    "tags": ["pipeline", "clustering", "classification"],
    "confidence": 0.92
  }'

# Step 6: Query memory
curl -X POST http://localhost:3000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what groups exist in the data?"}'
```

## API Reference

### MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `notebook_query` | Query notebook memory | `question: string`, `topK?: number` |
| `notebook_embed` | Create text embeddings | `texts: string[]`, `dimensions?: number` |
| `notebook_train` | Train a compute kernel | `algorithm`, `data`, `labels?`, `epochs?`, `learningRate?`, `k?` |
| `notebook_predict` | Run inference | `algorithm`, `input: number[]` |
| `notebook_analyze` | Compute statistics | `data: number[][]` |
| `notebook_cluster` | K-means clustering | `data: number[][]`, `k: number`, `maxIterations?: number` |
| `notebook_remember` | Store an insight | `id`, `text`, `tags?`, `confidence?` |
| `notebook_recall` | Recall insights | `query: string`, `topK?: number` |

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/tools` | List available tools |
| POST | `/query` | Query notebook memory |
| POST | `/embed` | Create embeddings |
| POST | `/train` | Train a kernel |
| POST | `/predict` | Make a prediction |
| POST | `/analyze` | Analyze data |
| POST | `/cluster` | Cluster data |
| POST | `/remember` | Store insight |
| POST | `/recall` | Recall insights |

## Development

```bash
npm run build
npm test
```

## License

MIT © purplepincher
