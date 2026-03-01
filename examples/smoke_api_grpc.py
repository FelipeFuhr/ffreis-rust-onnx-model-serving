from __future__ import annotations

import json
import os
import time
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlsplit

import grpc


def _validate_http_base(api_base: str) -> tuple[str, str]:
    parsed = urlsplit(api_base)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"API_BASE must use http or https scheme; got {parsed.scheme or '<empty>'}"
        )
    if not parsed.netloc:
        raise ValueError("API_BASE must include network location (host[:port])")
    return parsed.scheme, parsed.netloc


def _http_request(
    scheme: str,
    netloc: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 5.0,
) -> tuple[int, bytes]:
    conn_cls = HTTPSConnection if scheme == "https" else HTTPConnection
    conn = conn_cls(netloc, timeout=timeout_seconds)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    status = response.status
    payload = response.read()
    conn.close()
    return status, payload


def _wait_http_ok(scheme: str, netloc: str, path: str, timeout_seconds: float = 30.0) -> bytes:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            status, payload = _http_request(
                scheme, netloc, path, timeout_seconds=3.0
            )
            if status == 200:
                return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for HTTP 200 at {path}: {last_error}")


def _assert_http(api_base: str) -> None:
    scheme, netloc = _validate_http_base(api_base)
    for path in ("/live", "/healthz", "/ready", "/readyz", "/ping"):
        _ = _wait_http_ok(scheme, netloc, path)

    status, body = _http_request(
        scheme,
        netloc,
        "/invocations",
        method="POST",
        body=json.dumps({"instances": [[1.0, 2.0], [3.0, 4.0]]}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout_seconds=5.0,
    )
    if status != 200:
        raise RuntimeError(f"unexpected status from /invocations: {status}")
    payload = json.loads(body.decode("utf-8"))
    if isinstance(payload, dict):
        predictions = payload.get("predictions")
    else:
        predictions = payload
    assert predictions == [[2.0, 3.0], [4.0, 5.0]], payload


def _assert_grpc(target: str) -> None:
    deadline = time.time() + 30.0
    last_error: Exception | None = None

    # Retry gRPC connection until it succeeds or timeout
    while time.time() < deadline:
        try:
            with grpc.insecure_channel(target) as channel:
                live_rpc = channel.unary_unary(
                    "/onnxserving.grpc.InferenceService/Live",
                    request_serializer=lambda _: b"",
                    response_deserializer=lambda data: data,
                )
                ready_rpc = channel.unary_unary(
                    "/onnxserving.grpc.InferenceService/Ready",
                    request_serializer=lambda _: b"",
                    response_deserializer=lambda data: data,
                )
                _ = live_rpc(b"", timeout=5.0)
                _ = ready_rpc(b"", timeout=5.0)
                return  # Success
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)

    raise RuntimeError(f"timed out waiting for gRPC at {target}: {last_error}")


def main() -> None:
    api_base = os.getenv("API_BASE", "http://serving-api:8080")
    grpc_target = os.getenv("GRPC_TARGET", "serving-grpc:50052")
    _assert_http(api_base)
    _assert_grpc(grpc_target)
    print("rust serving API+gRPC smoke passed")


if __name__ == "__main__":
    main()
