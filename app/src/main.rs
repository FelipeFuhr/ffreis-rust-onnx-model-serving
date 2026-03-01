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

async fn run_from_env<InitFn, HttpFn, GrpcFn, HttpFut, GrpcFut, HttpErr, GrpcErr>(
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
    let (mode_raw, host_raw, port_raw) = runtime_env_values();
    run_with(
        mode_raw,
        host_raw,
        port_raw,
        cfg,
        init,
        http_runner,
        grpc_runner,
    )
    .await
}

fn runtime_env_values() -> (Option<String>, Option<String>, Option<String>) {
    (
        env::var(SERVE_MODE_ENV_KEY).ok(),
        env::var(HOST_ENV_KEY).ok(),
        env::var(PORT_ENV_KEY).ok(),
    )
}

async fn run() -> Result<(), String> {
    run_from_env(
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
    use super::run_from_env;
    use super::run_with;
    use super::runtime_env_values;
    use super::HOST_ENV_KEY;
    use super::PORT_ENV_KEY;
    use super::SERVE_MODE_ENV_KEY;
    use app::AppConfig;
    use std::env;
    use std::io;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

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

        fn remove(key: &'static str) -> Self {
            let previous = env::var(key).ok();
            env::remove_var(key);
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

    #[test]
    fn runtime_env_values_reads_present_variables() {
        let _lock = env_lock().lock().expect("env lock");
        let _mode = EnvVarGuard::set(SERVE_MODE_ENV_KEY, "grpc");
        let _host = EnvVarGuard::set(HOST_ENV_KEY, "10.0.0.1");
        let _port = EnvVarGuard::set(PORT_ENV_KEY, "9000");

        let (mode, host, port) = runtime_env_values();
        assert_eq!(mode.as_deref(), Some("grpc"));
        assert_eq!(host.as_deref(), Some("10.0.0.1"));
        assert_eq!(port.as_deref(), Some("9000"));
    }

    #[test]
    fn env_var_guard_set_restores_previous_value_on_drop() {
        let _lock = env_lock().lock().expect("env lock");
        env::set_var(SERVE_MODE_ENV_KEY, "original");
        {
            let _guard = EnvVarGuard::set(SERVE_MODE_ENV_KEY, "temporary");
            assert_eq!(
                env::var(SERVE_MODE_ENV_KEY).ok().as_deref(),
                Some("temporary")
            );
        }
        assert_eq!(
            env::var(SERVE_MODE_ENV_KEY).ok().as_deref(),
            Some("original")
        );
    }

    #[test]
    fn env_var_guard_remove_restores_missing_value_on_drop() {
        let _lock = env_lock().lock().expect("env lock");
        env::remove_var(PORT_ENV_KEY);
        {
            let _guard = EnvVarGuard::remove(PORT_ENV_KEY);
            assert!(env::var(PORT_ENV_KEY).is_err());
        }
        assert!(env::var(PORT_ENV_KEY).is_err());
    }

    #[tokio::test]
    async fn run_with_returns_grpc_server_error() {
        let cfg = AppConfig::default();
        let result = run_with(
            Some("grpc".to_string()),
            None,
            None,
            cfg,
            |_| Ok(()),
            |_, _, _| async { Ok::<(), io::Error>(()) },
            |_, _, _| async { Err(io::Error::other("grpc bind failed")) },
        )
        .await;

        let error = result.expect_err("grpc error expected");
        assert!(error.contains("grpc server failed"));
    }

    #[test]
    fn run_from_env_uses_grpc_settings_from_environment() {
        let _lock = env_lock().lock().expect("env lock");
        let _mode = EnvVarGuard::set(SERVE_MODE_ENV_KEY, "grpc");
        let _host = EnvVarGuard::set(HOST_ENV_KEY, "127.0.0.1");
        let _port = EnvVarGuard::set(PORT_ENV_KEY, "50099");
        let http_calls = Arc::new(AtomicUsize::new(0));
        let grpc_calls = Arc::new(AtomicUsize::new(0));
        let grpc_host = Arc::new(Mutex::new(String::default()));
        let grpc_port = Arc::new(AtomicUsize::new(0));
        let http_calls_ref = http_calls.clone();
        let grpc_calls_ref = grpc_calls.clone();
        let grpc_host_ref = grpc_host.clone();
        let grpc_port_ref = grpc_port.clone();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let result = rt.block_on(run_from_env(
            AppConfig::default(),
            |_| Ok(()),
            move |_, _, _| {
                http_calls_ref.fetch_add(1, Ordering::SeqCst);
                async { Ok::<(), io::Error>(()) }
            },
            move |host, port, _| {
                grpc_calls_ref.fetch_add(1, Ordering::SeqCst);
                *grpc_host_ref.lock().expect("grpc host lock") = host;
                grpc_port_ref.store(usize::from(port), Ordering::SeqCst);
                async { Ok::<(), io::Error>(()) }
            },
        ));

        assert!(result.is_ok());
        assert_eq!(http_calls.load(Ordering::SeqCst), 0);
        assert_eq!(grpc_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            grpc_host.lock().expect("grpc host read").as_str(),
            "127.0.0.1"
        );
        assert_eq!(grpc_port.load(Ordering::SeqCst), 50099);
    }

    #[test]
    fn run_from_env_defaults_to_http_when_env_is_absent() {
        let _lock = env_lock().lock().expect("env lock");
        let _mode = EnvVarGuard::remove(SERVE_MODE_ENV_KEY);
        let _host = EnvVarGuard::remove(HOST_ENV_KEY);
        let _port = EnvVarGuard::remove(PORT_ENV_KEY);
        let http_calls = Arc::new(AtomicUsize::new(0));
        let grpc_calls = Arc::new(AtomicUsize::new(0));
        let http_host = Arc::new(Mutex::new(String::default()));
        let http_port = Arc::new(AtomicUsize::new(0));
        let http_calls_ref = http_calls.clone();
        let grpc_calls_ref = grpc_calls.clone();
        let http_host_ref = http_host.clone();
        let http_port_ref = http_port.clone();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let result = rt.block_on(run_from_env(
            AppConfig::default(),
            |_| Ok(()),
            move |host, port, _| {
                http_calls_ref.fetch_add(1, Ordering::SeqCst);
                *http_host_ref.lock().expect("http host lock") = host;
                http_port_ref.store(usize::from(port), Ordering::SeqCst);
                async { Ok::<(), io::Error>(()) }
            },
            move |_, _, _| {
                grpc_calls_ref.fetch_add(1, Ordering::SeqCst);
                async { Ok::<(), io::Error>(()) }
            },
        ));

        assert!(result.is_ok());
        assert_eq!(http_calls.load(Ordering::SeqCst), 1);
        assert_eq!(grpc_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            http_host.lock().expect("http host read").as_str(),
            "0.0.0.0"
        );
        assert_eq!(http_port.load(Ordering::SeqCst), 8080);
    }
}
