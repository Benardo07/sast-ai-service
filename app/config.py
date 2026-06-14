from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8001
    backend_base_url: str = "http://127.0.0.1:8000"
    source_repo_root: str = "../tugas-akhir"
    active_release_file: str = "./var/active_release.json"
    default_device: str = "cpu"
    autoload_persisted_release: bool = False

    model_config = {
        "env_prefix": "SAST_AI_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    @property
    def service_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def workspace_root(self) -> Path:
        return self.service_root.parent.parent

    @property
    def source_repo_path(self) -> Path:
        path = Path(self.source_repo_root)
        if not path.is_absolute():
            path = (self.service_root / path).resolve()
        else:
            path = path.resolve()
        return path

    @property
    def active_release_path(self) -> Path:
        path = Path(self.active_release_file)
        if not path.is_absolute():
            path = (self.service_root / path).resolve()
        else:
            path = path.resolve()
        return path


settings = Settings()
