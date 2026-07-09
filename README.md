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

## Status & capabilities

Verified by building (`npm run build`), running the suite (`npm test` →
**87 tests across 7 suites, all passing**), and driving both the REST API
and the MCP stdio server with the examples below.

- ✅ **Two transports, one handler core:** 8 MCP tools (`notebook_query`,
  `notebook_embed`, `notebook_train`, `notebook_predict`, `notebook_analyze`,
  `notebook_cluster`, `notebook_remember`, `notebook_recall`) defined in
  `src/mcp/tools.ts`; the REST routes in `src/rest/routes.ts` delegate to the
  same `handleToolCall`.
- ✅ **Three compute kernels:** `micro-nn`, `logistic`, `kmeans` — pure
  TypeScript, zero runtime dependencies.
- ✅ **JSON-RPC 2.0 MCP server** over stdio (`initialize`, `tools/list`,
  `tools/call`); `protocolVersion` reports `2024-11-05`.
- ✅ **REST API** on Node's built-in `http` (no Express), with permissive CORS
  (`*`, `OPTIONS` preflight → `204`).
- ✅ **Two extra GET endpoints** (not shown in the examples above):
  - `GET /health` → `{ status, version, uptime, kernels: ["micro-nn","logistic","kmeans"] }`
  - `GET /tools`  → `{ tools: <toolDefinitions> }` (same payload as `tools/list`)
- ⚠️ **REST + MCP cannot share one process:** see the mode-selection note above.

### What it does NOT do (yet)

- 🔮 **No persistence.** Trained models, the insight store, and the embedding
  index all live in process memory and are lost on restart. (Kernels expose
  `serialize()`/`deserialize()`, but nothing wires them to disk.)
- 🔮 **No authentication, TLS, or rate limiting** on the REST API.
- 🔮 **No concurrency / load tests.** The 87 tests are unit/integration-level;
  there is no integration test for long-running or concurrent requests.
- 🔮 **MicroNN is shallow** (one hidden layer, fixed 100 epochs / lr 0.01);
  no conv/recurrent architectures, no early stopping, no regularization.
- 🔮 **LogisticRegression is binary-only.** Multi-class needs multiple runs.
- 🔮 **Embeddings are not learned** — deterministic hash projection of
  whitespace tokens; semantic quality is low and only suited to rough
  short-text similarity.
- 🔮 **`notebook_query` vs `notebook_recall`:** both call `insightStore.recall()`
  and return the stored insights; `query` additionally echoes the `query`
  string. Neither does any LLM/RAG-style answering.

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
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"mean\":[3,4],\"variance\":[2.6666666666666665,2.6666666666666665],\"standardized\":[[-1.224744871391589,-1.224744871391589],[0,0],[1.224744871391589,1.224744871391589]],\"dimensions\":2,\"sampleCount\":3}"}]}}
```

(The `analyze` / `notebook_analyze` result includes `mean`, `variance`,
`standardized`, `dimensions`, and `sampleCount`.)

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

* **MicroNN** — A two-layer MLP (input → hidden 16 units → output) with **ReLU** activation on the hidden layer and **softmax** on the output. Training is backpropagation with stochastic gradient descent (cross-entropy loss); gradients are computed manually (no autodiff framework). The hidden dimension can be set via constructor argument; epochs/learning rate are fixed at 100 / 0.01.
* **LogisticRegression** — Binary logistic classifier trained with gradient descent (500 epochs, lr 0.05). Only binary labels are supported; the **lexicographically-last** label after sorting (`[...new Set(labels)].sort()`) is treated as the positive class — not the first-seen label, so label *names* matter.
* **KMeans** — K-means with k-means++ initialization and Euclidean distance. Convergence is checked by assignment stability; maximum iterations defaults to 100.

Semantic memory is built from **whitespace-token hashing** to a configurable-dimensionality embedding vector (default 64). Each `(token, dimension)` pair is hashed to a ternary contribution (`+1`/`0`/`-1`); the resulting vector is L2-normalised and searched by cosine similarity. (The projection is deterministic, not a trained model — note the code comments say "character n-gram", but the implementation splits on `\s+` into whole word tokens.) The `EmbeddingIndex` stores vectors in-memory; there is no persistence to disk.

All endpoints are served by a built‑in Node.js HTTP server (no Express or other runtime dependencies). The MCP server reads JSON‑RPC 2.0 from `stdin` and writes responses to `stdout`.

## Configuration / options

| Flag         | Description                                      | Default |
|--------------|--------------------------------------------------|---------|
| `--rest`     | Enable the REST API server                        | off     |
| `--port`     | Port for the REST API                             | 3000    |
| `--mcp`      | Enable the MCP server (JSON‑RPC over stdio)       | off     |

Mode selection (in `src/index.ts`): `--mcp` alone → MCP-only over stdio;
`--rest` alone → REST-only; **both flags (or neither)** → "both" mode. In
"both" mode only the REST server actually starts — the MCP stdio loop is
**not** run (it prints `Starting in dual mode. MCP via stdio requires
dedicated process.` to stderr), because a single process can't serve stdio
JSON-RPC and an HTTP listener interchangeably. Run MCP in its own process
(`npm run start:mcp`).

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
