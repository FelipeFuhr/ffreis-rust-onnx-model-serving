use std::net::TcpListener;
use std::process::{Child, Command};
use std::thread;
use std::time::Duration;
use std::{fs, path::PathBuf};

use app::grpc::inference_service_client::InferenceServiceClient;
use app::grpc::PredictRequest;
use app::{serve_http, AppConfig};

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
}

#[cfg(unix)]
fn stop_child_gracefully(child: &mut Child) {
    let pid = child.id().to_string();
    let _ = Command::new("kill").arg("-TERM").arg(pid).status();
    for _ in 0..20 {
        if child.try_wait().ok().flatten().is_some() {
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(not(unix))]
fn stop_child_gracefully(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn make_temp_model_dir() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().expect("temp dir");
    let model_path: PathBuf = tmp.path().join("model.onnx");
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("model.onnx");
    let fixture = fs::read(&fixture_path).expect("read fixture model");
    fs::write(&model_path, fixture).expect("write fixture model");
    tmp
}

fn spawn_http_binary(
    exe: &str,
    port: u16,
    model_dir: Option<&std::path::Path>,
    max_body_bytes: Option<usize>,
) -> Child {
    let mut cmd = Command::new(exe);
    cmd.env("SERVE_MODE", "http")
        .env("HOST", "127.0.0.1")
        .env("PORT", port.to_string());
    if let Some(dir) = model_dir {
        cmd.env("MODEL_TYPE", "onnx")
            .env("SM_MODEL_DIR", dir.to_string_lossy().to_string());
    }
    if let Some(limit) = max_body_bytes {
        cmd.env("MAX_BODY_BYTES", limit.to_string());
    }
    cmd.spawn().expect("spawn app binary")
}

fn spawn_grpc_binary(exe: &str, port: u16, model_dir: &std::path::Path) -> Child {
    let mut cmd = Command::new(exe);
    cmd.env("SERVE_MODE", "grpc")
        .env("HOST", "127.0.0.1")
        .env("PORT", port.to_string())
        .env("MODEL_TYPE", "onnx")
        .env("SM_MODEL_DIR", model_dir.to_string_lossy().to_string());
    cmd.spawn().expect("spawn app binary")
}

fn post_json_invocations(port: u16, body: Vec<u8>) -> reqwest::blocking::Response {
    post_invocations(port, body, "application/json", "application/json")
}

fn post_invocations(
    port: u16,
    body: Vec<u8>,
    content_type: &str,
    accept: &str,
) -> reqwest::blocking::Response {
    let client = reqwest::blocking::Client::new();
    let url = format!("http://127.0.0.1:{port}/invocations");
    client
        .post(url)
        .header("content-type", content_type)
        .header("accept", accept)
        .body(body)
        .send()
        .expect("invocations request")
}

#[test]
fn binary_starts_http_service_and_answers_healthz() {
    let Some(exe) = option_env!("CARGO_BIN_EXE_app") else {
        return; // Skip when binary is not available in this context.
    };
    let port = free_port();
    let mut child = spawn_http_binary(exe, port, None, None);

    let url = format!("http://127.0.0.1:{port}/healthz");
    let mut response = None;
    for _ in 0..100 {
        match reqwest::blocking::get(&url) {
            Ok(r) if r.status() == reqwest::StatusCode::OK => {
                response = Some(r);
                break;
            }
            _ => thread::sleep(Duration::from_millis(50)),
        }
    }
    assert_eq!(
        response.expect("healthz did not succeed in time").status(),
        reqwest::StatusCode::OK
    );

    stop_child_gracefully(&mut child);
}

#[test]
fn binary_handles_grpc_predict_happy_path() {
    let Some(exe) = option_env!("CARGO_BIN_EXE_app") else {
        return;
    };
    let model_dir = make_temp_model_dir();
    let port = free_port();
    let mut child = spawn_grpc_binary(exe, port, model_dir.path());

    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let reply = runtime.block_on(async {
        let endpoint = format!("http://127.0.0.1:{port}");
        for _ in 0..60 {
            if let Ok(mut client) = InferenceServiceClient::connect(endpoint.clone()).await {
                if let Ok(response) = client
                    .predict(PredictRequest {
                        payload: br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec(),
                        content_type: "application/json".to_string(),
                        accept: "application/json".to_string(),
                    })
                    .await
                {
                    return response.into_inner();
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("grpc server did not become ready in time");
    });

    assert_eq!(reply.content_type, "application/json");
    assert_eq!(
        reply.metadata.get("batch_size").map(String::as_str),
        Some("2")
    );
    let payload: serde_json::Value = serde_json::from_slice(&reply.body).expect("json body");
    assert!(payload.is_array(), "prediction body should be a JSON array");

    stop_child_gracefully(&mut child);
}

#[test]
fn binary_returns_400_for_invalid_json_payload() {
    let Some(exe) = option_env!("CARGO_BIN_EXE_app") else {
        return;
    };
    let model_dir = make_temp_model_dir();
    let port = free_port();
    let mut child = spawn_http_binary(exe, port, Some(model_dir.path()), None);

    thread::sleep(Duration::from_millis(350));
    let response = post_json_invocations(port, b"{not-json".to_vec());
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);

    stop_child_gracefully(&mut child);
}

#[test]
fn binary_supports_csv_response_format() {
    let Some(exe) = option_env!("CARGO_BIN_EXE_app") else {
        return;
    };
    let model_dir = make_temp_model_dir();
    let port = free_port();
    let mut child = spawn_http_binary(exe, port, Some(model_dir.path()), None);

    thread::sleep(Duration::from_millis(350));
    let response = post_invocations(port, b"1,2\n3,4\n".to_vec(), "text/csv", "text/csv");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("text/csv")
    );
    let body = response.text().expect("csv body");
    assert!(body.contains('\n') || !body.is_empty());

    stop_child_gracefully(&mut child);
}

#[test]
fn binary_ready_returns_500_when_model_missing() {
    let Some(exe) = option_env!("CARGO_BIN_EXE_app") else {
        return;
    };
    let empty_model_dir = tempfile::tempdir().expect("temp dir");
    let port = free_port();
    let mut child = spawn_http_binary(exe, port, Some(empty_model_dir.path()), None);

    thread::sleep(Duration::from_millis(350));
    let url = format!("http://127.0.0.1:{port}/ready");
    let response = reqwest::blocking::get(url).expect("ready request");
    assert_eq!(
        response.status(),
        reqwest::StatusCode::INTERNAL_SERVER_ERROR
    );

    stop_child_gracefully(&mut child);
}

#[test]
fn binary_handles_invocations_happy_path() {
    let Some(exe) = option_env!("CARGO_BIN_EXE_app") else {
        return;
    };
    let model_dir = make_temp_model_dir();
    let port = free_port();
    let mut child = spawn_http_binary(exe, port, Some(model_dir.path()), None);

    thread::sleep(Duration::from_millis(350));
    let response = post_json_invocations(port, br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec());
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body = response.text().expect("body text");
    assert!(!body.is_empty(), "prediction body should not be empty");

    stop_child_gracefully(&mut child);
}

#[test]
fn binary_returns_413_when_payload_too_large() {
    let Some(exe) = option_env!("CARGO_BIN_EXE_app") else {
        return;
    };
    let model_dir = make_temp_model_dir();
    let port = free_port();
    let mut child = spawn_http_binary(exe, port, Some(model_dir.path()), Some(8));

    thread::sleep(Duration::from_millis(350));
    let response = post_json_invocations(port, br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec());
    assert_eq!(response.status(), reqwest::StatusCode::PAYLOAD_TOO_LARGE);

    stop_child_gracefully(&mut child);
}

#[tokio::test]
async fn inprocess_http_server_handles_health_and_invocations() {
    let model_dir = make_temp_model_dir();
    let cfg = AppConfig {
        model_type: "onnx".to_string(),
        model_dir: model_dir.path().to_string_lossy().to_string(),
        model_filename: "model.onnx".to_string(),
        max_records: 1000,
        ..AppConfig::default()
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral http port");
    let addr = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(serve_http(listener, cfg));

    let client = reqwest::Client::new();
    for _ in 0..60 {
        let ok = client
            .get(format!("http://{addr}/readyz"))
            .send()
            .await
            .map(|r| r.status() == reqwest::StatusCode::OK)
            .unwrap_or(false);
        if ok {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    let health = client
        .get(format!("http://{addr}/healthz"))
        .send()
        .await
        .expect("health response");
    assert_eq!(health.status(), reqwest::StatusCode::OK);

    let predict = client
        .post(format!("http://{addr}/invocations"))
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .body(br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec())
        .send()
        .await
        .expect("predict response");
    assert_eq!(predict.status(), reqwest::StatusCode::OK);
    assert!(!predict.bytes().await.expect("predict bytes").is_empty());

    handle.abort();
}
