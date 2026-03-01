use std::env;
use std::fmt::Display;
use std::future::Future;

use app::{init_telemetry, run_grpc_server, run_http_server, AppConfig};

const SERVE_MODE_ENV_KEY: &str = "SERVE_MODE";
const HOST_ENV_KEY: &str = "HOST";
const PORT_ENV_KEY: &str = "PORT";

fn resolve_runtime_settings(
    mode_raw: Option<String>,
    host_raw: Option<String>,
    port_raw: Option<String>,
) -> (String, String, u16) {
    let mode = mode_raw.unwrap_or_else(|| "http".to_string());
    let host = host_raw.unwrap_or_else(|| "0.0.0.0".to_string());
    let port = port_raw
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or_else(|| if mode == "grpc" { 50052 } else { 8080 });
    (mode, host, port)
}

async fn run_with<InitFn, HttpFn, GrpcFn, HttpFut, GrpcFut, HttpErr, GrpcErr>(
    mode_raw: Option<String>,
    host_raw: Option<String>,
    port_raw: Option<String>,
    cfg: AppConfig,
    init: InitFn,
    http_runner: HttpFn,
    grpc_runner: GrpcFn,
) -> Result<(), String>
where
    InitFn: Fn(&AppConfig) -> Result<(), String>,
    HttpFn: Fn(String, u16, AppConfig) -> HttpFut,
    GrpcFn: Fn(String, u16, AppConfig) -> GrpcFut,
    HttpFut: Future<Output = Result<(), HttpErr>>,
    GrpcFut: Future<Output = Result<(), GrpcErr>>,
    HttpErr: Display,
    GrpcErr: Display,
{
    let (mode, host, port) = resolve_runtime_settings(mode_raw, host_raw, port_raw);
    init(&cfg).map_err(|err| format!("telemetry initialization failed: {err}"))?;

    match mode.as_str() {
        "grpc" => grpc_runner(host, port, cfg)
            .await
            .map_err(|err| format!("grpc server failed: {err}"))?,
        _ => http_runner(host, port, cfg)
            .await
            .map_err(|err| format!("http server failed: {err}"))?,
    }
    Ok(())
}

async fn run() -> Result<(), String> {
    run_with(
        env::var(SERVE_MODE_ENV_KEY).ok(),
        env::var(HOST_ENV_KEY).ok(),
        env::var(PORT_ENV_KEY).ok(),
        AppConfig::default(),
        init_telemetry,
        |host, port, cfg| async move { run_http_server(&host, port, cfg).await },
        |host, port, cfg| async move { run_grpc_server(&host, port, cfg).await },
    )
    .await
}

#[tokio::main]
async fn main() {
    if let Err(err) = run().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_runtime_settings;
    use super::run_with;
    use app::AppConfig;
    use std::io;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn resolve_runtime_settings_defaults_to_http() {
        let (mode, host, port) = resolve_runtime_settings(None, None, None);
        assert_eq!(mode, "http");
        assert_eq!(host, "0.0.0.0");
        assert_eq!(port, 8080);
    }

    #[test]
    fn resolve_runtime_settings_uses_grpc_default_port() {
        let (mode, host, port) = resolve_runtime_settings(Some("grpc".to_string()), None, None);
        assert_eq!(mode, "grpc");
        assert_eq!(host, "0.0.0.0");
        assert_eq!(port, 50052);
    }

    #[test]
    fn resolve_runtime_settings_uses_explicit_port_when_valid() {
        let (mode, host, port) = resolve_runtime_settings(
            Some("http".to_string()),
            Some("127.0.0.1".to_string()),
            Some("9000".to_string()),
        );
        assert_eq!(mode, "http");
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 9000);
    }

    #[test]
    fn resolve_runtime_settings_falls_back_when_port_is_invalid() {
        let (mode, host, port) = resolve_runtime_settings(
            Some("grpc".to_string()),
            Some("localhost".to_string()),
            Some("invalid".to_string()),
        );
        assert_eq!(mode, "grpc");
        assert_eq!(host, "localhost");
        assert_eq!(port, 50052);
    }

    #[test]
    fn resolve_runtime_settings_unknown_mode_uses_http_default_port() {
        let (mode, host, port) = resolve_runtime_settings(
            Some("custom".to_string()),
            Some("0.0.0.0".to_string()),
            None,
        );
        assert_eq!(mode, "custom");
        assert_eq!(host, "0.0.0.0");
        assert_eq!(port, 8080);
    }

    #[tokio::test]
    async fn run_with_calls_http_runner_by_default() {
        let cfg = AppConfig::default();
        let http_calls = Arc::new(AtomicUsize::new(0));
        let grpc_calls = Arc::new(AtomicUsize::new(0));
        let http_calls_ref = http_calls.clone();
        let grpc_calls_ref = grpc_calls.clone();

        let result = run_with(
            None,
            Some("127.0.0.1".to_string()),
            Some("8088".to_string()),
            cfg,
            |_| Ok(()),
            move |_, _, _| {
                http_calls_ref.fetch_add(1, Ordering::SeqCst);
                async { Ok::<(), io::Error>(()) }
            },
            move |_, _, _| {
                grpc_calls_ref.fetch_add(1, Ordering::SeqCst);
                async { Ok::<(), io::Error>(()) }
            },
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(http_calls.load(Ordering::SeqCst), 1);
        assert_eq!(grpc_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn run_with_calls_grpc_runner_for_grpc_mode() {
        let cfg = AppConfig::default();
        let http_calls = Arc::new(AtomicUsize::new(0));
        let grpc_calls = Arc::new(AtomicUsize::new(0));
        let http_calls_ref = http_calls.clone();
        let grpc_calls_ref = grpc_calls.clone();

        let result = run_with(
            Some("grpc".to_string()),
            None,
            None,
            cfg,
            |_| Ok(()),
            move |_, _, _| {
                http_calls_ref.fetch_add(1, Ordering::SeqCst);
                async { Ok::<(), io::Error>(()) }
            },
            move |_, _, _| {
                grpc_calls_ref.fetch_add(1, Ordering::SeqCst);
                async { Ok::<(), io::Error>(()) }
            },
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(http_calls.load(Ordering::SeqCst), 0);
        assert_eq!(grpc_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn run_with_returns_telemetry_error() {
        let cfg = AppConfig::default();
        let result = run_with(
            None,
            None,
            None,
            cfg,
            |_| Err("boom".to_string()),
            |_, _, _| async { Ok::<(), io::Error>(()) },
            |_, _, _| async { Ok::<(), io::Error>(()) },
        )
        .await;

        let error = result.expect_err("telemetry error expected");
        assert!(error.contains("telemetry initialization failed"));
    }

    #[tokio::test]
    async fn run_with_returns_http_server_error() {
        let cfg = AppConfig::default();
        let result = run_with(
            Some("http".to_string()),
            None,
            None,
            cfg,
            |_| Ok(()),
            |_, _, _| async { Err(io::Error::other("bind failed")) },
            |_, _, _| async { Ok::<(), io::Error>(()) },
        )
        .await;

        let error = result.expect_err("http error expected");
        assert!(error.contains("http server failed"));
    }
}
