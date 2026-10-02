// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct AppInfo {
    name: String,
    version: String,
    engine_status: String,
    backend_port: u16,
}

#[derive(Debug, Serialize, Deserialize)]
struct SystemDiagnostics {
    os: String,
    arch: String,
    engine_ready: bool,
    storage_path: String,
    backend_url: String,
}

#[tauri::command]
fn get_app_info() -> AppInfo {
    AppInfo {
        name: "STRATA".into(),
        version: "1.0.0".into(),
        engine_status: "READY".into(),
        backend_port: 8000,
    }
}

#[tauri::command]
fn get_system_diagnostics() -> SystemDiagnostics {
    SystemDiagnostics {
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        engine_ready: true,
        storage_path: "data/storage".into(),
        backend_url: "http://127.0.0.1:8000".into(),
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            get_app_info,
            get_system_diagnostics
        ])
        .run(tauri::generate_context!())
        .expect("error while running STRATA Tauri desktop shell");
}
