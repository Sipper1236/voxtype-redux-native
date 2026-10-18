# Native parakeet.cpp / Redux extension

This fork adds an optional `parakeet-cpp` Cargo feature to Voxtype 1.1.1. It dynamically loads the parakeet.cpp C API and passes the captured mono 16 kHz float PCM directly to the resident native model. It does not require the external adapter.

The separate [stock-Voxtype adapter repository](https://github.com/Sipper1236/voxtype-redux-adapter) contains the adapter, benchmark runners, archived results, and a pinned submodule for this fork. The adapter is the selected deployment; this fork preserves the original in-process implementation for comparison and development.

## Build

Install upstream Voxtype's Rust/native build prerequisites, including libclang/bindgen and the ALSA development headers. Build a recent parakeet.cpp shared library with C API ABI 10+ and obtain a compatible Redux GGUF separately.

```sh
cargo build --release --features parakeet-cpp --bin voxtype
```

Configure an isolated profile:

```toml
engine = "parakeet"
[hotkey]
enabled = false
[parakeet]
model = "/absolute/path/to/redux-packed.gguf"
```

```sh
export VOXTYPE_PARAKEET_CPP_LIBRARY=/absolute/path/to/libparakeet.so
./target/release/voxtype --config /absolute/path/to/native.toml transcribe /absolute/path/to/speech.wav
```

The shared library must be able to find its ggml dependencies. Other library search candidates are `~/.local/lib/voxtype-parakeet/libparakeet.so`, the system loader's `libparakeet.so`, `/usr/local/lib/libparakeet.so`, and `/usr/lib/libparakeet.so`.

## Benchmark helper

`examples/bench_worker.rs` loads a configured transcriber once, then reads newline-delimited JSON requests from stdin:

```sh
cargo build --release --features parakeet-cpp,moonshine --example bench_worker
printf '{"file":"/absolute/path/to/speech.wav"}\n' | \
  ./target/release/examples/bench_worker /absolute/path/to/native.toml
```

The first output line reports readiness and loading time. Each request returns text and inference time. ONNX backends need a compatible dynamic ONNX Runtime (API 24+); set `ORT_DYLIB_PATH` when required. GGUF C API inference does not initialize ONNX Runtime.

## Scope and validation

- `.gguf` paths select the C API; existing ONNX Parakeet paths remain available.
- One mutex protects the native context, and the shared library remains loaded until the context is freed.
- Transcript strings are copied and freed with the native allocator's matching function.
- Timed transcription currently returns one segment spanning the recording. It does not provide native word or sentence alignment.
- This original in-process path does not expose the adapter's pool-thread or quiet-chunk tuning. Its original default benchmark result is therefore not an interface-only comparison with the tuned adapter.

The implementation passed a feature check/build and native transcription on an i5-8350U. The original laptop build used an isolated manifest with the unrelated OpenVINO git patch removed to avoid its fetch, reused bundled Whisper bindings, and kept temporary build output off a quota-limited `/tmp`. Those environment workarounds are not committed here. A fresh full-manifest build is not claimed by that validation; upstream Cargo may fetch its configured OpenVINO patch even when that feature is unused.

Upstream Voxtype remains MIT-licensed. parakeet.cpp and model weights are separate dependencies with their own licenses; Redux weights are CC BY 4.0. No models or native libraries are bundled in this fork.
