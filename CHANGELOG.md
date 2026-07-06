# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- GitHub Actions CI workflow (`.github/workflows/ci.yml`) testing Node.js 18, 20, and 22.
- MIT `LICENSE` file.
- `.aider*` to `.gitignore`.

### Changed

- `package.json` author updated from `SuperInstance` to `purplepincher`.
- `package.json` repository URL updated to the `purplepincher/exocortex-mcp-ts` fork.
- `README.md` trimmed to verified facts and rewritten in an instructional style.
- `npm test` script now runs `jest --verbose --forceExit`.
- CI test job timeout set to 5 minutes.

### Removed

- `AGENT.md` and `memory/JOURNAL.md` ensign-scaffold boilerplate.

### Fixed

- Box-Muller transform in `src/compute/stats.ts` (`randn`) now uses `Math.cos(2.0 * Math.PI * v)` instead of an incorrect multiplication.
- REST test suite no longer hangs on keep-alive sockets: disabled HTTP agent pooling and force-closes idle connections in `tests/rest.test.ts`.

## [1.0.0] - 2026-06-06

### Added

- Initial release of **exocortex-mcp-ts**.
- MCP server implementing JSON-RPC 2.0 over stdio.
- REST API built on Node.js built-in `http` module.
- Shared tool handlers used by both MCP and REST interfaces.
- Compute kernels:
  - `MicroNN` — 2-layer MLP classifier.
  - `LogisticRegression` — binary logistic classifier.
  - `KMeans` — k-means clustering with k-means++ initialization.
- Memory layer:
  - Hash-based random projection embeddings.
  - In-process insight store with cosine-similarity recall.
- 87 tests across 7 Jest test suites covering compute, memory, MCP, and REST modules.
- Zero runtime production dependencies; only TypeScript/Jest dev dependencies.
- 817-line `README.md` with overview, examples, architecture, glossary, and references.
