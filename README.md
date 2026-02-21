# ffreis-rust-onnx-model-serving

Rust implementation of the same serving contract used in `ffreis-python-onnx-model-serving`.

## HTTP surface

- `GET /live`
- `GET /healthz`
- `GET /ready`
- `GET /readyz`
- `GET /ping`
- `POST /invocations`

`/invocations` accepts JSON (`application/json`) and CSV (`text/csv`) payloads and returns JSON predictions:

```json
{"predictions":[0,0,0]}
```

For readiness and invocation handling, the service expects an ONNX model path:
- `SM_MODEL_DIR/MODEL_FILENAME`
- defaults to `/opt/ml/model/model.onnx`

OpenAPI transport contract:

- `docs/openapi.yaml`

## gRPC surface

Proto: `app/proto/onnx_serving_grpc/inference.proto`

gRPC server reflection is intentionally not enabled in runtime paths.

- `Live(LiveRequest) -> StatusReply`
- `Ready(ReadyRequest) -> StatusReply`
- `Predict(PredictRequest) -> PredictReply`

## Run locally

```bash
make -C app test
SERVE_MODE=http HOST=0.0.0.0 PORT=8080 cargo run --manifest-path app/Cargo.toml
SERVE_MODE=grpc HOST=0.0.0.0 PORT=50052 cargo run --manifest-path app/Cargo.toml
```

## Consistency gates

- `make grpc-check`
- `make test-grpc-parity`
- `make smoke-api-grpc`
