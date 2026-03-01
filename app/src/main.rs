use std::env;

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

#[tokio::main]
async fn main() {
    let (mode, host, port) = resolve_runtime_settings(
        env::var(SERVE_MODE_ENV_KEY).ok(),
        env::var(HOST_ENV_KEY).ok(),
        env::var(PORT_ENV_KEY).ok(),
    );
    let cfg = AppConfig::default();
    if let Err(err) = init_telemetry(&cfg) {
        eprintln!("telemetry initialization failed: {err}");
        std::process::exit(1);
    }

    match mode.as_str() {
        "grpc" => {
            if let Err(err) = run_grpc_server(&host, port, cfg).await {
                eprintln!("grpc server failed: {err}");
                std::process::exit(1);
            }
        }
        _ => {
            if let Err(err) = run_http_server(&host, port, cfg).await {
                eprintln!("http server failed: {err}");
                std::process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_runtime_settings;

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
}
