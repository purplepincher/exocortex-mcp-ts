# exocortex-mcp-ts

TypeScript implementation of the Exocortex MCP server + REST API — a notebook runtime that exposes the same operations over JSON-RPC 2.0 (stdio) and plain HTTP.

## Quickstart

```bash
git clone https://github.com/purplepincher/exocortex-mcp-ts.git
cd exocortex-mcp-ts
npm install
npm run build
```

### REST API

Start the server on port 3000:

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

## Usage

### Example 1: MCP Server JSON-RPC

Run the MCP server then send the following requests on separate lines:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize"}
```

```json
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
```

```json
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"notebook_analyze","arguments":{"data":[[1.0,2.0],[3.0,4.0],[5.0,6.0]]}}}
```

Response for `notebook_analyze`:

```json
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"mean\":[3,4],\"variance\":[2.6667,2.6667],\"dimensions\":2,\"sampleCount\":3}"}]}}
```

### Example 2: Embedding Search

Create embeddings:

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
```

Store an insight:

```bash
curl -X POST http://localhost:3000/remember \
  -H "Content-Type: application/json" \
  -d '{
    "id": "ml-001",
    "text": "Random forests are an ensemble method for classification and regression",
    "tags": ["ml", "ensemble", "classification"],
    "confidence": 0.95
  }'
```

Recall insights by semantic similarity:

```bash
curl -X POST http://localhost:3000/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "classification methods", "topK": 5}'
```

### Example 3: Full Pipeline

Analyze, cluster, train, predict, and query memory:

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

## How it works

Three compute kernels operate on numeric data points (`number[]`):

* **MicroNN** — A two-layer MLP (input → hidden 16 units → output) with sigmoid activations. No automatic differentiation; gradients are computed manually. The hidden dimension can be set via constructor argument.
* **LogisticRegression** — Binary logistic classifier trained with gradient descent. Only binary labels are supported; the first seen label is treated as the positive class.
* **KMeans** — K-means with k-means++ initialization and Euclidean distance. Convergence is checked by assignment stability; maximum iterations defaults to 100.

Semantic memory is built from character n-gram hashing to a configurable‑dimensionality embedding vector (default 64), followed by cosine‑similarity search. The `EmbeddingIndex` stores vectors in‑memory; there is no persistence to disk.

All endpoints are served by a built‑in Node.js HTTP server (no Express or other runtime dependencies). The MCP server reads JSON‑RPC 2.0 from `stdin` and writes responses to `stdout`.

## Configuration / options

| Flag         | Description                                      | Default |
|--------------|--------------------------------------------------|---------|
| `--rest`     | Enable the REST API server                        | off     |
| `--port`     | Port for the REST API                             | 3000    |
| `--mcp`      | Enable the MCP server (JSON‑RPC over stdio)       | off     |

You can pass both flags simultaneously; the process will run both servers.

## Real constraints & limitations

* **Memory** is entirely in‑process. Restarting the server loses all stored insights, trained models, and embeddings.
* **MicroNN** has one hidden layer only; accuracy on complex nonlinear patterns is limited. No support for convolutional or recurrent architectures.
* **LogisticRegression** only handles binary classification. Multi‑class problems require multiple runs or a different kernel.
* **KMeans** uses Euclidean distance and does not support categorical data directly. Random initialization can produce different centroids across runs.
* **Embeddings** use a simple hash‑based projection, not a trained embedding model. Semantic quality is low; the method is appropriate only for rough similarity matching with short text.
* **87 tests** across 7 suites validate the core modules, but there is no integration test for long‑running or concurrent requests.

## License

MIT © purplepincher

## Contributing

Send pull requests to the repository referenced above. If you find a bug or want to request a feature, open an issue.
