use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub service_name: String,
    pub service_version: String,
    pub deployment_env: String,
    pub model_dir: String,
    pub model_type: String,
    pub model_filename: String,
    pub input_mode: String,
    pub default_content_type: String,
    pub default_accept: String,
    pub tabular_dtype: String,
    pub csv_delimiter: String,
    pub csv_has_header: String,
    pub csv_skip_blank_lines: bool,
    pub json_key_instances: String,
    pub jsonl_features_key: String,
    pub tabular_id_columns: String,
    pub tabular_feature_columns: String,
    pub predictions_only: bool,
    pub json_output_key: String,
    pub max_body_bytes: usize,
    pub max_records: usize,
    pub max_inflight: usize,
    pub acquire_timeout_s: f64,
    pub prometheus_enabled: bool,
    pub prometheus_path: String,
    pub otel_enabled: bool,
    pub otel_endpoint: String,
    pub otel_headers: String,
    pub otel_timeout_s: f64,
    pub onnx_input_map_json: String,
    pub onnx_output_map_json: String,
    pub onnx_input_dtype_map_json: String,
    pub onnx_dynamic_batch: bool,
    pub tabular_num_features: usize,
    pub onnx_input_name: String,
    pub onnx_output_name: String,
    pub onnx_output_index: usize,
    pub swagger_enabled: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            service_name: env_str("SERVICE_NAME", "model-serving-universal"),
            service_version: env_str("SERVICE_VERSION", "dev"),
            deployment_env: env_str("DEPLOYMENT_ENV", "local"),
            model_dir: env_str("SM_MODEL_DIR", "/opt/ml/model"),
            model_type: env_str("MODEL_TYPE", "").trim().to_ascii_lowercase(),
            model_filename: env_str("MODEL_FILENAME", "").trim().to_string(),
            input_mode: env_str("INPUT_MODE", "tabular").trim().to_ascii_lowercase(),
            default_content_type: env_str("DEFAULT_CONTENT_TYPE", "application/json"),
            default_accept: env_str("DEFAULT_ACCEPT", "application/json"),
            tabular_dtype: env_str("TABULAR_DTYPE", "float32")
                .trim()
                .to_ascii_lowercase(),
            csv_delimiter: env_str("CSV_DELIMITER", ","),
            csv_has_header: env_str("CSV_HAS_HEADER", "auto")
                .trim()
                .to_ascii_lowercase(),
            csv_skip_blank_lines: env_bool("CSV_SKIP_BLANK_LINES", true),
            json_key_instances: env_str("JSON_KEY_INSTANCES", "instances"),
            jsonl_features_key: env_str("JSONL_FEATURES_KEY", "features"),
            tabular_id_columns: env_str("TABULAR_ID_COLUMNS", "").trim().to_string(),
            tabular_feature_columns: env_str("TABULAR_FEATURE_COLUMNS", "").trim().to_string(),
            predictions_only: env_bool("RETURN_PREDICTIONS_ONLY", true),
            json_output_key: env_str("JSON_OUTPUT_KEY", "predictions"),
            max_body_bytes: env_usize("MAX_BODY_BYTES", 6 * 1024 * 1024),
            max_records: env_usize("MAX_RECORDS", 5000),
            max_inflight: env_usize("MAX_INFLIGHT", 16),
            acquire_timeout_s: env_f64("ACQUIRE_TIMEOUT_S", 0.25),
            prometheus_enabled: env_bool("PROMETHEUS_ENABLED", true),
            prometheus_path: env_str("PROMETHEUS_PATH", "/metrics"),
            otel_enabled: env_bool("OTEL_ENABLED", true),
            otel_endpoint: env_str("OTEL_EXPORTER_OTLP_ENDPOINT", "")
                .trim()
                .to_string(),
            otel_headers: env_str("OTEL_EXPORTER_OTLP_HEADERS", ""),
            otel_timeout_s: env_f64("OTEL_EXPORTER_OTLP_TIMEOUT", 10.0),
            onnx_input_map_json: env_str("ONNX_INPUT_MAP_JSON", "").trim().to_string(),
            onnx_output_map_json: env_str("ONNX_OUTPUT_MAP_JSON", "").trim().to_string(),
            onnx_input_dtype_map_json: env_str("ONNX_INPUT_DTYPE_MAP_JSON", "").trim().to_string(),
            onnx_dynamic_batch: env_bool("ONNX_DYNAMIC_BATCH", true),
            tabular_num_features: env_usize("TABULAR_NUM_FEATURES", 0),
            onnx_input_name: env_str("ONNX_INPUT_NAME", "").trim().to_string(),
            onnx_output_name: env_str("ONNX_OUTPUT_NAME", "").trim().to_string(),
            onnx_output_index: env_usize("ONNX_OUTPUT_INDEX", 0),
            swagger_enabled: env_bool("SWAGGER_ENABLED", false),
        }
    }
}

fn env_str(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "y" | "on"
        ),
        Err(_) => default,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(default)
}

impl AppConfig {
    pub(crate) fn model_path(&self) -> PathBuf {
        let dir = Path::new(&self.model_dir);
        if !dir.exists() {
            panic!(
                "Configured model directory '{}' does not exist",
                self.model_dir
            );
        }
        if !dir.is_dir() {
            panic!(
                "Configured model path '{}' is not a directory",
                self.model_dir
            );
        }
        let filename = if self.model_filename.trim().is_empty() {
            "model.onnx".to_string()
        } else {
            self.model_filename.clone()
        };
        dir.join(filename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn env_helpers_parse_and_fallback() {
        let _guard = env_lock().lock().expect("env lock");

        let bool_key = "APP_TEST_BOOL";
        let usize_key = "APP_TEST_USIZE";
        let f64_key = "APP_TEST_F64";
        let swagger_key = "SWAGGER_ENABLED";

        env::remove_var(bool_key);
        env::remove_var(usize_key);
        env::remove_var(f64_key);
        env::remove_var(swagger_key);

        assert!(env_bool(bool_key, true));
        assert_eq!(env_usize(usize_key, 7), 7);
        assert_eq!(env_f64(f64_key, 1.5), 1.5);
        assert!(!AppConfig::default().swagger_enabled);

        env::set_var(bool_key, "yes");
        env::set_var(usize_key, "12");
        env::set_var(f64_key, "2.75");
        env::set_var(swagger_key, "true");
        assert!(env_bool(bool_key, false));
        assert_eq!(env_usize(usize_key, 0), 12);
        assert_eq!(env_f64(f64_key, 0.0), 2.75);
        assert!(AppConfig::default().swagger_enabled);

        env::set_var(bool_key, "no");
        env::set_var(usize_key, "bad");
        env::set_var(f64_key, "bad");
        assert!(!env_bool(bool_key, true));
        assert_eq!(env_usize(usize_key, 9), 9);
        assert_eq!(env_f64(f64_key, 3.5), 3.5);

        env::remove_var(bool_key);
        env::remove_var(usize_key);
        env::remove_var(f64_key);
        env::remove_var(swagger_key);
    }

    #[test]
    fn model_path_uses_default_and_custom_filename() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let default_cfg = AppConfig {
            model_dir: tmp.path().to_string_lossy().to_string(),
            model_filename: "".to_string(),
            ..AppConfig::default()
        };
        assert_eq!(default_cfg.model_path(), tmp.path().join("model.onnx"));

        let custom_cfg = AppConfig {
            model_dir: tmp.path().to_string_lossy().to_string(),
            model_filename: "custom.onnx".to_string(),
            ..AppConfig::default()
        };
        assert_eq!(custom_cfg.model_path(), tmp.path().join("custom.onnx"));
    }

    #[test]
    fn model_path_panics_when_path_is_not_a_directory() {
        let tmp = tempfile::tempdir().expect("temp dir");
        let file_path = tmp.path().join("model_file");
        fs::write(&file_path, b"x").expect("write test file");
        let cfg = AppConfig {
            model_dir: file_path.to_string_lossy().to_string(),
            ..AppConfig::default()
        };
        let panic = std::panic::catch_unwind(|| cfg.model_path());
        assert!(panic.is_err());
    }
}
