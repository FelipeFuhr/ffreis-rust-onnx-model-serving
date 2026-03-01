use axum::http::HeaderName;
use axum::response::Response as AxumResponse;
use uuid::Uuid;

use crate::config::AppConfig;

pub fn init_telemetry(cfg: &AppConfig) -> Result<(), String> {
    if !cfg.otel_enabled {
        return Ok(());
    }
    if cfg.otel_endpoint.trim().is_empty() {
        eprintln!(
            "OTEL_ENABLED=true but OTEL_EXPORTER_OTLP_ENDPOINT is empty; tracing exporter is disabled"
        );
    }
    Ok(())
}

pub(crate) fn attach_trace_correlation_headers(response: &mut AxumResponse) {
    let trace_id = Uuid::new_v4().simple().to_string();
    let span_id = &trace_id[..16];
    if let Ok(value) = trace_id.parse() {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-trace-id"), value);
    }
    if let Ok(value) = span_id.parse() {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-span-id"), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;

    #[test]
    fn init_telemetry_returns_ok_when_disabled() {
        let cfg = AppConfig {
            otel_enabled: false,
            ..AppConfig::default()
        };
        assert!(init_telemetry(&cfg).is_ok());
    }

    #[test]
    fn init_telemetry_returns_ok_when_enabled_without_endpoint() {
        let cfg = AppConfig {
            otel_enabled: true,
            otel_endpoint: "   ".to_string(),
            ..AppConfig::default()
        };
        assert!(init_telemetry(&cfg).is_ok());
    }

    #[test]
    fn attach_trace_correlation_headers_sets_expected_headers() {
        let mut response = AxumResponse::new(Body::empty());
        attach_trace_correlation_headers(&mut response);

        let trace = response
            .headers()
            .get("x-trace-id")
            .and_then(|v| v.to_str().ok())
            .expect("x-trace-id");
        let span = response
            .headers()
            .get("x-span-id")
            .and_then(|v| v.to_str().ok())
            .expect("x-span-id");

        assert_eq!(trace.len(), 32);
        assert_eq!(span.len(), 16);
    }
}
