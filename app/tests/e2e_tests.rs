use std::net::TcpListener;
use std::process::{Child, Command};
use std::thread;
use std::time::Duration;
use std::{fs, path::PathBuf};

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

#[test]
fn binary_starts_http_service_and_answers_healthz() {
    let Some(exe) = option_env!("CARGO_BIN_EXE_app") else {
        return; // Skip when binary is not available in this context.
    };
    let port = free_port();
    let mut child = Command::new(exe)
        .env("SERVE_MODE", "http")
        .env("HOST", "127.0.0.1")
        .env("PORT", port.to_string())
        .spawn()
        .expect("spawn app binary");

    thread::sleep(Duration::from_millis(250));
    let url = format!("http://127.0.0.1:{port}/healthz");
    let response = reqwest::blocking::get(url).expect("healthz request");
    assert_eq!(response.status(), reqwest::StatusCode::OK);

    stop_child_gracefully(&mut child);
}

#[test]
fn binary_handles_invocations_happy_path() {
    let Some(exe) = option_env!("CARGO_BIN_EXE_app") else {
        return;
    };
    let model_dir = make_temp_model_dir();
    let port = free_port();
    let mut child = Command::new(exe)
        .env("SERVE_MODE", "http")
        .env("HOST", "127.0.0.1")
        .env("PORT", port.to_string())
        .env("MODEL_TYPE", "onnx")
        .env(
            "SM_MODEL_DIR",
            model_dir.path().to_string_lossy().to_string(),
        )
        .spawn()
        .expect("spawn app binary");

    thread::sleep(Duration::from_millis(350));
    let client = reqwest::blocking::Client::new();
    let url = format!("http://127.0.0.1:{port}/invocations");
    let response = client
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .body(br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec())
        .send()
        .expect("invocations request");
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
    let mut child = Command::new(exe)
        .env("SERVE_MODE", "http")
        .env("HOST", "127.0.0.1")
        .env("PORT", port.to_string())
        .env("MODEL_TYPE", "onnx")
        .env(
            "SM_MODEL_DIR",
            model_dir.path().to_string_lossy().to_string(),
        )
        .env("MAX_BODY_BYTES", "8")
        .spawn()
        .expect("spawn app binary");

    thread::sleep(Duration::from_millis(350));
    let client = reqwest::blocking::Client::new();
    let url = format!("http://127.0.0.1:{port}/invocations");
    let response = client
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json")
        .body(br#"{"instances":[[1.0,2.0],[3.0,4.0]]}"#.to_vec())
        .send()
        .expect("invocations request");
    assert_eq!(response.status(), reqwest::StatusCode::PAYLOAD_TOO_LARGE);

    stop_child_gracefully(&mut child);
}
