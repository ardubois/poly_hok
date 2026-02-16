# Black-Scholes Implementation Guide

This guide documents how the Black-Scholes call-price application is implemented in this repository, including both the CUDA reference and the PolyHok version used for validation.

## Scope

In scope:

- CUDA implementation for call-price computation.
- PolyHok port of the same call-price kernel.
- CPU-vs-GPU numerical validation method.

Out of scope:

- PUT-price implementation in PolyHok.
- Throughput and latency benchmarking claims.
- Fusion comparison experiments not directly tied to Black-Scholes.

## Code Map

- CUDA reference: `experiments/black-scholes-cuda/blackscholes.cu`
- PolyHok experiment: `experiments/black_scholes_polyhok.exs`
- PolyHok SKE variant (map3): `experiments/black_scholes_skeimpl.exs`

## CUDA Reference Implementation

`experiments/black-scholes-cuda/blackscholes.cu` implements a baseline with:

- Input generation using seed `5347` and ranges:
  - stock price: `[5.0, 30.0]`
  - strike price: `[1.0, 100.0]`
  - option years: `[0.25, 10.0]`
- CPU reference functions:
  - `CND(double d)`
  - `BlackScholesBodyCPU(...)`
  - `BlackScholesCPU(...)`
- GPU functions:
  - `norm_cdf(float x)`
  - `blackscholes_body(float S, float X, float T, float r, float v)`
  - `black_scholes(...)` kernel
- Launch configuration:
  - `numThreads = 256`
  - `numBlocks = (OPT_N + numThreads - 1) / numThreads`

Validation output includes:

- `L1 norm`
- `Max absolute error`
- `[BlackScholes] - Test Summary`

## PolyHok Implementation

`experiments/black_scholes_polyhok.exs` mirrors the CUDA logic in the PolyHok DSL.

### Device and Kernel Definitions

`PolyHok.defmodule BlackScholesExp` defines:

- `defd norm_cdf/1`
- `defd blackscholes_body/5`
- `defk black_scholes/7`

The kernel computes `opt = threadIdx.x + blockIdx.x * blockDim.x` and writes one call-price output per option when `opt < n`.

### Host-Side Flow

`BlackScholesExperiment.run/0` performs:

1. Deterministic input generation (same seed and ranges as CUDA).
2. Transfer to GPU (`PolyHok.new_gnx/1`).
3. Kernel launch via `PolyHok.spawn/4` with `@threads_per_block 256`.
4. Result retrieval (`PolyHok.get_gnx/1`).
5. CPU reference recomputation.
6. Error metrics (`L1 norm`, `Max absolute error`) and threshold check.

### Input Contract

`run_kernel/5` enforces:

- matching tensor shapes
- tensor type `{:f, 32}`

If these constraints are not met, the experiment raises an error.

### Pass Condition

- `@l1_tolerance` is `1.0e-6`.
- Experiment prints `Test passed.` when `L1 norm <= 1.0e-6`.

## Reproduction

From repo root:

```bash
# Build PolyHok NIFs (if not already built)
make

# Run PolyHok Black-Scholes validation
mix run experiments/black_scholes_polyhok.exs

# Optional: run the SKE map3 variant
mix run experiments/black_scholes_skeimpl.exs

# Build and run CUDA reference
nvcc -O3 experiments/black-scholes-cuda/blackscholes.cu -o experiments/black-scholes-cuda/run
./experiments/black-scholes-cuda/run
```

## Known Limitations

- PolyHok experiment computes call price only.
- Current PolyHok script is float32-only (`{:f, 32}`).
- No benchmark claims are part of this implementation guide.
