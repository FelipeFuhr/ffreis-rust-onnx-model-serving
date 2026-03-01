use std::env;
use std::time::Duration;
use std::{fs, path::PathBuf};

use app::grpc::inference_service_client::InferenceServiceClient;
use app::grpc::{LiveRequest, PredictRequest, ReadyRequest};
use app::{serve_grpc, serve_http, AppConfig};
use tempfile::TempDir;
use tokio::net::TcpListener;

/// Maximum number of retry attempts for server connection/readiness checks
const MAX_RETRY_ATTEMPTS: u32 = 100;
/// Delay between retry attempts in milliseconds
const RETRY_DELAY_MS: u64 = 50;
const OPENAPI_SPEC_PATH_ENV_KEY: &str = "OPENAPI_SPEC_PATH";

struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = env::var(key).ok();
        env::set_var(key, value);
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(previous) = &self.previous {
            env::set_var(self.key, previous);
        } else {
            env::remove_var(self.key);
        }
    }
}

async fn start_http_server(
    cfg: AppConfig,
) -> (String, tokio::task::JoinHandle<Result<(), std::io::Error>>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral http port");
    let addr = listener.local_addr().expect("http local addr");
    let handle = tokio::spawn(serve_http(listener, cfg));
    (format!("http://{}", addr), handle)
}

async fn start_grpc_server(
    cfg: AppConfig,
) -> (
    String,
    tokio::task::JoinHandle<Result<(), Box<dyn std::error::Error + Send + Sync>>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral grpc port");
    let addr = listener.local_addr().expect("grpc local addr");
    let handle = tokio::spawn(serve_grpc(listener, cfg));
    (format!("http://{}", addr), handle)
}

/// Poll HTTP readiness endpoint until server is ready or timeout
async fn poll_until_ready(client: &reqwest::Client, base_url: &str) -> bool {
    for _ in 0..MAX_RETRY_ATTEMPTS {
        match client.get(format!("{base_url}/readyz")).send().await {
            Ok(response) if response.status() == reqwest::StatusCode::OK => {
                return true;
            }
            _ => {
                tokio::time::sleep(Duration::from_millis(RETRY_DELAY_MS)).await;
            }
        }
    }
    false
}

/// Retry gRPC client connection until success or timeout
async fn retry_grpc_connect(
    url: String,
) -> Option<InferenceServiceClient<tonic::transport::Channel>> {
    for _ in 0..MAX_RETRY_ATTEMPTS {
        match InferenceServiceClient::connect(url.clone()).await {
            Ok(client) => return Some(client),
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(RETRY_DELAY_MS)).await;
            }
        }
    }
    None
}

async fn fetch_status(client: &reqwest::Client, url: String) -> reqwest::StatusCode {
    client
        .get(url)
        .send()
        .await
        .expect("http get response")
        .status()
}

async fn post_json_invocations(
    client: &reqwest::Client,
    http_base: &str,
    body: Vec<u8>,
) -> reqwest::Response {
    client
        .post(format!("{http_base}/invocations"))
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .body(body)
        .send()
        .await
        .expect("http response")
}

fn build_cfg_with_temp_model() -> (TempDir, AppConfig) {
    let tmp = tempfile::tempdir().expect("temp dir");
    let model_path: PathBuf = tmp.path().join("model.onnx");
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("model.onnx");
    let fixture = fs::read(&fixture_path).expect("read fixture model");
    fs::write(&model_path, fixture).expect("write fixture model");
    let cfg = AppConfig {
        model_type: "onnx".to_string(),
        model_dir: tmp.path().to_string_lossy().to_string(),
        model_filename: "model.onnx".to_string(),
        max_records: 1000,
        ..AppConfig::default()
    };
    (tmp, cfg)
}

#[tokio::test]
async fn http_health_and_ready_endpoints_are_available() {
    let (_tmp, cfg) = build_cfg_with_temp_model();
    let (base_url, handle) = start_http_server(cfg).await;
    let client = reqwest::Client::new();

    let ready = poll_until_ready(&client, &base_url).await;
    assert!(ready, "HTTP server did not become ready in time");

    for path in ["/live", "/healthz", "/ready", "/readyz", "/ping"] {
        let response = client
            .get(format!("{base_url}{path}"))
            .send()
            .await
            .expect("http endpoint should answer");
        assert_eq!(response.status(), reqwest::StatusCode::OK);
    }
    let metrics = client
        .get(format!("{base_url}/metrics"))
        .send()
        .await
        .expect("metrics endpoint should answer");
    assert_eq!(metrics.status(), reqwest::StatusCode::OK);

    handle.abort();
}

#[tokio::test]
async fn http_and_grpc_predict_parity_for_json_and_csv() {
    let (_tmp, cfg) = build_cfg_with_temp_model();
    let (http_base, http_handle) = start_http_server(cfg.clone()).await;
    let (grpc_url, grpc_handle) = start_grpc_server(cfg).await;

    let client = reqwest::Client::new();
    let mut grpc_client = retry_grpc_connect(grpc_url)
        .await
        .expect("gRPC client should connect after retries");

    let payloads = vec![
        (
            br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec(),
            "application/json",
        ),
        (b"1,2\n3,4\n".to_vec(), "text/csv"),
    ];

    for (payload, content_type) in payloads {
        let http_response = client
            .post(format!("{http_base}/invocations"))
            .header("content-type", content_type)
            .header("accept", "application/json")
            .body(payload.clone())
            .send()
            .await
            .expect("http invocation response");
        assert_eq!(http_response.status(), reqwest::StatusCode::OK);
        assert!(http_response.headers().contains_key("x-trace-id"));
        assert!(http_response.headers().contains_key("x-span-id"));
        let http_body = http_response.bytes().await.expect("http bytes");

        let grpc_reply = grpc_client
            .predict(PredictRequest {
                payload,
                content_type: content_type.to_string(),
                accept: "application/json".to_string(),
            })
            .await
            .expect("grpc predict should pass")
            .into_inner();

        assert_eq!(grpc_reply.content_type, "application/json");
        assert_eq!(http_body.as_ref(), grpc_reply.body.as_slice());
    }

    let live = grpc_client
        .live(LiveRequest {})
        .await
        .expect("grpc live")
        .into_inner();
    assert!(live.ok);
    assert_eq!(live.status, "live");

    let ready = grpc_client
        .ready(ReadyRequest {})
        .await
        .expect("grpc ready")
        .into_inner();
    assert!(ready.ok);
    assert_eq!(ready.status, "ready");

    http_handle.abort();
    grpc_handle.abort();
}

#[tokio::test]
async fn invalid_json_maps_to_http_400_and_grpc_invalid_argument() {
    let (_tmp, cfg) = build_cfg_with_temp_model();
    let (http_base, http_handle) = start_http_server(cfg.clone()).await;
    let (grpc_url, grpc_handle) = start_grpc_server(cfg).await;

    let bad_payload = b"{not-json".to_vec();
    let client = reqwest::Client::new();
    let http_response = client
        .post(format!("{http_base}/invocations"))
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .body(bad_payload.clone())
        .send()
        .await
        .expect("http bad response");
    assert_eq!(http_response.status(), reqwest::StatusCode::BAD_REQUEST);

    let mut grpc_client = retry_grpc_connect(grpc_url)
        .await
        .expect("gRPC client should connect after retries");
    let grpc_error = grpc_client
        .predict(PredictRequest {
            payload: bad_payload,
            content_type: "application/json".to_string(),
            accept: "application/json".to_string(),
        })
        .await
        .expect_err("grpc should reject invalid json");
    assert_eq!(grpc_error.code(), tonic::Code::InvalidArgument);

    http_handle.abort();
    grpc_handle.abort();
}

#[tokio::test]
async fn record_limit_violation_maps_to_http_400_and_grpc_invalid_argument() {
    let (_tmp, mut cfg) = build_cfg_with_temp_model();
    cfg.max_records = 1;
    let (http_base, http_handle) = start_http_server(cfg.clone()).await;
    let (grpc_url, grpc_handle) = start_grpc_server(cfg).await;

    // Wait for servers to be ready using retry mechanisms
    let client = reqwest::Client::new();
    assert!(
        poll_until_ready(&client, &http_base).await,
        "HTTP server should start"
    );
    retry_grpc_connect(grpc_url.clone())
        .await
        .expect("gRPC server should start");

    let payload = br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec();
    let http_response = client
        .post(format!("{http_base}/invocations"))
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .body(payload.clone())
        .send()
        .await
        .expect("http response");
    assert_eq!(http_response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body: serde_json::Value = http_response.json().await.expect("json body");
    assert!(body["error"]
        .as_str()
        .unwrap_or_default()
        .contains("too_many_records"));

    let mut grpc_client = retry_grpc_connect(grpc_url)
        .await
        .expect("gRPC client should connect after retries");
    let grpc_error = grpc_client
        .predict(PredictRequest {
            payload,
            content_type: "application/json".to_string(),
            accept: "application/json".to_string(),
        })
        .await
        .expect_err("grpc should reject record overflow");
    assert_eq!(grpc_error.code(), tonic::Code::InvalidArgument);
    assert!(grpc_error.message().contains("too_many_records"));

    http_handle.abort();
    grpc_handle.abort();
}

#[tokio::test]
async fn oversized_payload_maps_to_http_413_and_grpc_invalid_argument() {
    let (_tmp, mut cfg) = build_cfg_with_temp_model();
    cfg.max_body_bytes = 8;
    let (http_base, http_handle) = start_http_server(cfg.clone()).await;
    let (grpc_url, grpc_handle) = start_grpc_server(cfg).await;
    tokio::time::sleep(Duration::from_millis(75)).await;

    let payload = b"1,2,3\n4,5,6\n".to_vec();
    let client = reqwest::Client::new();
    let http_response = client
        .post(format!("{http_base}/invocations"))
        .header("content-type", "text/csv")
        .header("accept", "application/json")
        .body(payload.clone())
        .send()
        .await
        .expect("http response");
    assert_eq!(
        http_response.status(),
        reqwest::StatusCode::PAYLOAD_TOO_LARGE
    );

    let mut grpc_client = retry_grpc_connect(grpc_url)
        .await
        .expect("gRPC client should connect after retries");
    let grpc_error = grpc_client
        .predict(PredictRequest {
            payload,
            content_type: "text/csv".to_string(),
            accept: "application/json".to_string(),
        })
        .await
        .expect_err("grpc should reject oversized payload");
    assert_eq!(grpc_error.code(), tonic::Code::InvalidArgument);
    assert!(grpc_error.message().contains("payload too large"));

    http_handle.abort();
    grpc_handle.abort();
}

#[tokio::test]
async fn grpc_predict_uses_defaults_when_content_type_and_accept_are_missing() {
    let (_tmp, mut cfg) = build_cfg_with_temp_model();
    cfg.default_content_type = "application/json".to_string();
    cfg.default_accept = "application/json".to_string();
    let (grpc_url, grpc_handle) = start_grpc_server(cfg).await;

    let mut grpc_client = retry_grpc_connect(grpc_url)
        .await
        .expect("gRPC client should connect after retries");
    let reply = grpc_client
        .predict(PredictRequest {
            payload: br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec(),
            content_type: "".to_string(),
            accept: "".to_string(),
        })
        .await
        .expect("grpc predict should use defaults")
        .into_inner();

    assert_eq!(reply.content_type, "application/json");
    assert_eq!(reply.metadata.get("batch_size"), Some(&"2".to_string()));

    grpc_handle.abort();
}

#[tokio::test]
async fn grpc_predict_returns_internal_when_model_fails_to_load() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let cfg = AppConfig {
        model_type: "onnx".to_string(),
        model_dir: tmp.path().to_string_lossy().to_string(),
        model_filename: "missing.onnx".to_string(),
        ..AppConfig::default()
    };
    let (grpc_url, grpc_handle) = start_grpc_server(cfg).await;

    let mut grpc_client = retry_grpc_connect(grpc_url)
        .await
        .expect("gRPC client should connect after retries");
    let grpc_error = grpc_client
        .predict(PredictRequest {
            payload: b"1,2\n".to_vec(),
            content_type: "text/csv".to_string(),
            accept: "application/json".to_string(),
        })
        .await
        .expect_err("grpc predict should fail with load error");
    assert_eq!(grpc_error.code(), tonic::Code::Internal);
    assert!(grpc_error.message().contains("ONNX model not found"));

    grpc_handle.abort();
}

#[tokio::test]
async fn payload_too_large_returns_413() {
    let (_tmp, mut cfg) = build_cfg_with_temp_model();
    cfg.max_body_bytes = 8;
    let (http_base, http_handle) = start_http_server(cfg).await;
    tokio::time::sleep(Duration::from_millis(75)).await;

    let client = reqwest::Client::new();
    let response = post_json_invocations(
        &client,
        &http_base,
        br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec(),
    )
    .await;
    assert_eq!(response.status(), reqwest::StatusCode::PAYLOAD_TOO_LARGE);

    http_handle.abort();
}

#[tokio::test]
async fn swagger_routes_respect_toggle_flag() {
    let (_tmp, mut cfg) = build_cfg_with_temp_model();
    let (http_base_off, http_handle_off) = start_http_server(cfg.clone()).await;
    let client = reqwest::Client::new();
    assert_eq!(
        fetch_status(&client, format!("{http_base_off}/docs")).await,
        reqwest::StatusCode::NOT_FOUND
    );
    assert_eq!(
        fetch_status(&client, format!("{http_base_off}/openapi.yaml")).await,
        reqwest::StatusCode::NOT_FOUND
    );
    http_handle_off.abort();

    let openapi_tmp = tempfile::tempdir().expect("temp dir");
    let openapi_path = openapi_tmp.path().join("openapi.yaml");
    fs::write(&openapi_path, "openapi: 3.1.0\n").expect("write openapi spec");
    let _env_guard = EnvVarGuard::set(
        OPENAPI_SPEC_PATH_ENV_KEY,
        openapi_path.to_string_lossy().as_ref(),
    );

    cfg.swagger_enabled = true;
    let (http_base_on, http_handle_on) = start_http_server(cfg).await;
    assert_eq!(
        fetch_status(&client, format!("{http_base_on}/docs")).await,
        reqwest::StatusCode::OK
    );
    assert_eq!(
        fetch_status(&client, format!("{http_base_on}/openapi.yaml")).await,
        reqwest::StatusCode::OK
    );
    http_handle_on.abort();
}

#[tokio::test]
async fn multi_input_missing_key_maps_to_http_400_and_grpc_invalid_argument() {
    let (_tmp, mut cfg) = build_cfg_with_temp_model();
    cfg.onnx_input_map_json = r#"{"a":"input_a","b":"input_b"}"#.to_string();
    let (http_base, http_handle) = start_http_server(cfg.clone()).await;
    let (grpc_url, grpc_handle) = start_grpc_server(cfg).await;

    let payload = br#"{"instances":[{"a":[1.0,2.0]}]}"#.to_vec();
    let client = reqwest::Client::new();
    let http_response = post_json_invocations(&client, &http_base, payload.clone()).await;
    assert_eq!(http_response.status(), reqwest::StatusCode::BAD_REQUEST);
    let http_error: serde_json::Value = http_response.json().await.expect("json body");
    assert!(http_error["error"]
        .as_str()
        .unwrap_or_default()
        .contains("Missing key 'b'"));

    let mut grpc_client = retry_grpc_connect(grpc_url)
        .await
        .expect("gRPC client should connect after retries");
    let grpc_error = grpc_client
        .predict(PredictRequest {
            payload,
            content_type: "application/json".to_string(),
            accept: "application/json".to_string(),
        })
        .await
        .expect_err("grpc should reject missing key");
    assert_eq!(grpc_error.code(), tonic::Code::InvalidArgument);
    assert!(grpc_error.message().contains("Missing key 'b'"));

    http_handle.abort();
    grpc_handle.abort();
}

#[tokio::test]
async fn readiness_endpoints_return_500_when_model_dir_missing() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let cfg = AppConfig {
        model_type: "onnx".to_string(),
        model_dir: tmp.path().to_string_lossy().to_string(),
        model_filename: "missing.onnx".to_string(),
        ..AppConfig::default()
    };
    let (http_base, http_handle) = start_http_server(cfg).await;
    let client = reqwest::Client::new();

    for path in ["/ready", "/readyz", "/ping"] {
        assert_eq!(
            fetch_status(&client, format!("{http_base}{path}")).await,
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    http_handle.abort();
}
