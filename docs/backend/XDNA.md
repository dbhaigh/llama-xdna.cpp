# AMD XDNA Backend

The XDNA backend is an experimental backend for AMD Ryzen AI NPUs. It discovers XDNA devices through XRT and provides XRT-backed host-visible buffers. It currently supports one narrowly bounded Q4_0 matrix-vector multiplication; all other computation must fall back to the CPU backend.

## Supported operation

`GGML_OP_MUL_MAT` is supported only for contiguous `Q4_0` weights and contiguous `F32` input/output, with `N=1`, default precision, no hint, and all trailing dimensions equal to 1. The supported `(M, K)` pairs are:

- `(16, 256)`
- `(256, 1536)`
- `(1536, 1536)`
- `(8960, 1536)`
- `(1536, 8960)`

For each pair, the weight shape is `[K, M, 1, 1]`, the input shape is `[K, 1, 1, 1]`, and the output shape is `[M, 1, 1, 1]`.

Other shapes, batches, types, and operations are rejected by `supports_op` and remain on the CPU. `GGML_OP_NONE` graph leaves are accepted as no-ops; they do not execute on the NPU. The five supported shapes, including the baseline `(16, 256)`, passed targeted `test-backend-ops` comparisons against the CPU reference on a Strix NPU using XRT 2.21. The Qwen vocabulary projection `(M=151936, K=1536)` timed out experimentally and remains unsupported. The Peano build reported missing stack-size metadata for three software-float helpers; the estimated requirement is at least 576 bytes. The validated operations completed, but this toolchain warning remains.

## Q4NX model loading

The llama model loader accepts a `.q4nx` file or its package directory when it contains `model.q4nx`, `config.json`, `tokenizer.json`, and `tokenizer_config.json`. The current loader supports `Qwen2ForCausalLM` packages with BF16 tensors and packed Q4NX matrices whose minimum plane is zero. BF16 tensors are converted to F32 and the packed matrices are unpacked to GGML Q4_0. Unsupported architectures, tensor types, or non-zero minimum planes are rejected.

The FastFlowLM `Qwen2.5-3B-Instruct-NPU2` package was validated with CPU-only inference. Its decoded Q4NX Q projection, FFN gate/down projections, and output head were also compared against the matching Q5_K_M GGUF tensors. The Q4NX input channels beyond the GGUF FFN width are zero padding.

Loading a Q4NX package does not imply that its operations run on the XDNA backend. The Qwen2.5-3B model's matrix shapes do not match the five XDNA Q4_0 kernel shapes above, so those operations remain on CPU. Native Q4NX kernels and XDNA compute for this model are not implemented or validated.

## Phase D benchmark

The matched benchmark used `bartowski/Qwen2.5-1.5B-Instruct-GGUF`, file `Qwen2.5-1.5B-Instruct-Q4_0.gguf` (SHA-256 `a5ab77ebbd1d6d70250ff415339935bebac619b7acc508f71545f08231369dc1`). It ran on a Ryzen AI 9 HX 370 with 12 threads, batch size 2048, ubatch size 512, and five repetitions per case. Both configurations used `-ngl 99 -nkvo 1`; CPU-only selected `-dev none`, and the XDNA-requested run selected `-dev XDNA0`.

- Prompt processing, 512 tokens: CPU 361.59 +/- 9.13 t/s; XDNA requested 333.85 +/- 7.25 t/s (-7.7%).
- Generation, 128 tokens: CPU 66.10 +/- 0.96 t/s; XDNA requested 61.14 +/- 0.13 t/s (-7.5%).
- Combined prompt/generation, 512/128 tokens: CPU 175.77 +/- 3.53 t/s; XDNA requested 161.37 +/- 1.35 t/s (-8.2%).

Unlike the earlier no-go run, this backend version has model-compatible Q4_0 kernels for the Qwen projection shapes. Generation uses single-token operations where those exact shapes can run on the NPU; prompt batches (`N>1`), the unsupported vocabulary projection `(M=151936, K=1536)`, and all other unsupported operations remain on CPU. Despite this NPU work, the XDNA-requested configuration was slower in all three end-to-end measurements, so it still fails the Phase D throughput gate. XRT reports NPU power as unavailable, so energy could not be compared. `-nkvo` is required for this run because the scheduler otherwise places a KV-cache view on XDNA for unsupported `SET_ROWS`.

Do not proceed to Phase E on the basis of these results. Revisit the performance gate only after an implementation change and a new matched benchmark demonstrate an end-to-end throughput or energy improvement.

## RPC and layer sharding

Build the XDNA host with both RPC and XDNA enabled, then explicitly select the XDNA device when starting the server. The server's default selection omits generic `ACCEL` devices such as XDNA.

```bash
cmake -B build-rpc -DGGML_RPC=ON -DGGML_XDNA=ON
cmake --build build-rpc --config Release --target ggml-rpc-server
build-rpc/bin/ggml-rpc-server --host 192.168.1.20 --port 50052 --device XDNA0
```

On the inference host, build with `GGML_RPC=ON` and run with the server endpoint and a one-device split:

```bash
cmake -B build-client -DGGML_RPC=ON -DGGML_BUILD_TESTS=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-client --config Release --target llama-cli
build-client/bin/llama-cli -m model.gguf --rpc 192.168.1.20:50052 --tensor-split 1 -ngl 99 --no-kv-offload --fit off
```

RPC protocol 8.1.0 forwards `supports_op` checks to the selected server device, so unsupported XDNA operations are rejected during scheduling and can fall back to CPU. Build the client and server from compatible versions. The standalone `test-rpc-supports-op` target checks the five supported Q4_0 shapes and confirms the timed-out vocabulary shape is rejected:

```bash
build-client/bin/test-rpc-supports-op 192.168.1.20:50052
```

Local loopback validation exposed XDNA as `RPC0`; `llama-cli --rpc ... --tensor-split 1 -ngl 99` assigned all 29 Qwen2.5-1.5B layers to that RPC device and completed inference. The RPC shape test passed from a client built without XRT, verifying that the XDNA host can serve a client without an NPU. This verifies layer placement and CPU fallback, not end-to-end NPU compute or performance.

The RPC server is unauthenticated and insecure. Bind it only on a trusted private network and restrict access with the host firewall; never expose it to the public Internet.

## Buffers

Each XDNA buffer is backed by an XRT host-only buffer object and uses its mapped host address for ggml tensor storage. Tensor get, set, 2D transfers, memset, and clear operations synchronize the relevant byte ranges with XRT. Copies between XDNA buffers use the mapped memory and synchronize both sides; other copies use ggml's generic get/set fallback.

The buffer reports the CPU backend's tensor alignment and checks each XRT mapping against it. XRT does not expose a fixed maximum allocation size for these host-only buffers, so the buffer type reports no fixed limit (`SIZE_MAX`); actual allocations remain constrained by host memory and XRT, and allocation failures are reported.

## Build

Install the XRT development package for your platform, then configure llama-xdna.cpp with:

```sh
cmake -B build -DGGML_XDNA=ON
cmake --build build
```

If CMake cannot find the XRT package, set `XILINX_XRT` to the SDK root; the backend checks `$XILINX_XRT/share/cmake/XRT`.

Runtime device discovery also requires compatible XRT driver modules. On Windows, `xrt_core.dll` must be available in the XRT installation root; the development headers and `xrt_coreutil.dll` alone are not sufficient.

The backend is disabled by default and does not require XRT unless `GGML_XDNA=ON`.

## Current limitations

- Only the five exact Q4_0/F32 shape pairs above are tested and executed on the NPU; every other type, batch size, and operation falls back to CPU. In particular, prompt batches with `N>1` do not use these kernels.
- The Qwen vocabulary projection is rejected after an experimental timeout. Practical context-length and parallel-sequence limits remain unvalidated; other model quantizations are unsupported by these kernels.
- XDNA RPC discovery and tensor-split layer placement were tested over loopback. Execution across separate hosts remains unvalidated.
- XRT devices are listed only when their reported name or platform identifies them as XDNA, NPU, or Ryzen AI.
