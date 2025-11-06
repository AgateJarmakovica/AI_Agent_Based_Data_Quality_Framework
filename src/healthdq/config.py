"""
Central configuration management for healthdq-ai framework
Author: Agate Jarmakoviča
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: str = Field(default="openai", description="LLM provider (openai, anthropic, huggingface)")
    model: str = Field(default="gpt-4", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, gt=0)
    timeout: int = Field(default=60, description="Request timeout in seconds")


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = Field(default="sqlite:///./healthdq.db")
    echo: bool = Field(default=False, description="Echo SQL queries")
    pool_size: int = Field(default=5)
    max_overflow: int = Field(default=10)


class VectorDBConfig(BaseModel):
    """Vector database configuration."""

    path: str = Field(default="./chroma_db")
    collection_name: str = Field(default="healthdq_memory")
    distance_metric: str = Field(default="cosine")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")


class AgentConfig(BaseModel):
    """Agent system configuration."""

    enable_memory: bool = Field(default=True)
    max_iterations: int = Field(default=10, gt=0)
    timeout_seconds: int = Field(default=300, gt=0)
    enable_collaboration: bool = Field(default=True)
    verbose: bool = Field(default=False)


class HITLConfig(BaseModel):
    """Human-in-the-Loop configuration."""

    enable: bool = Field(default=True)
    auto_approve_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    require_feedback: bool = Field(default=True)
    feedback_timeout: int = Field(default=300, description="Timeout in seconds")


class DataProcessingConfig(BaseModel):
    """Data processing configuration."""

    max_file_size_mb: float = Field(default=100.0, gt=0)
    supported_formats: list = Field(default=["csv", "json", "xlsx", "parquet"])
    chunk_size: int = Field(default=1000, gt=0)
    enable_async: bool = Field(default=True)
    encoding: str = Field(default="utf-8")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    file: Optional[str] = Field(default="./logs/healthdq.log")
    format: str = Field(default="text", description="text or json")
    rotation: str = Field(default="10 MB")
    retention: str = Field(default="1 week")


class MetadataConfig(BaseModel):
    """FAIR metadata configuration."""

    enable_versioning: bool = Field(default=True)
    enable_tracking: bool = Field(default=True)
    metadata_path: str = Field(default="./data/metadata")
    auto_save: bool = Field(default=True)


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, gt=0, lt=65536)
    reload: bool = Field(default=False)
    debug: bool = Field(default=False)
    workers: int = Field(default=1, gt=0)


class UIConfig(BaseModel):
    """Streamlit UI configuration."""

    server_port: int = Field(default=8501, gt=0, lt=65536)
    server_address: str = Field(default="0.0.0.0")
    theme: str = Field(default="light")


class Config(BaseModel):
    """Main configuration class."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    vectordb: VectorDBConfig = Field(default_factory=VectorDBConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    hitl: HITLConfig = Field(default_factory=HITLConfig)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("./data"))
    config_dir: Path = Field(default_factory=lambda: Path("./configs"))
    output_dir: Path = Field(default_factory=lambda: Path("./output"))

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()

        # LLM configuration
        config.llm.provider = os.getenv("DEFAULT_LLM_PROVIDER", config.llm.provider)
        config.llm.model = os.getenv("DEFAULT_MODEL", config.llm.model)
        config.llm.temperature = float(os.getenv("LLM_TEMPERATURE", config.llm.temperature))
        config.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS", config.llm.max_tokens))

        # API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        huggingface_key = os.getenv("HUGGINGFACE_API_KEY")

        if config.llm.provider == "openai" and openai_key:
            config.llm.api_key = openai_key
        elif config.llm.provider == "anthropic" and anthropic_key:
            config.llm.api_key = anthropic_key
        elif config.llm.provider == "huggingface" and huggingface_key:
            config.llm.api_key = huggingface_key

        # Database
        config.database.url = os.getenv("DATABASE_URL", config.database.url)

        # Vector DB
        config.vectordb.path = os.getenv("CHROMA_DB_PATH", config.vectordb.path)
        config.vectordb.collection_name = os.getenv("CHROMA_COLLECTION_NAME", config.vectordb.collection_name)

        # Agent
        config.agent.enable_memory = os.getenv("ENABLE_AGENT_MEMORY", "true").lower() == "true"
        config.agent.max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", config.agent.max_iterations))
        config.agent.timeout_seconds = int(os.getenv("AGENT_TIMEOUT_SECONDS", config.agent.timeout_seconds))

        # HITL
        config.hitl.enable = os.getenv("ENABLE_HITL", "true").lower() == "true"
        config.hitl.auto_approve_threshold = float(
            os.getenv("HITL_AUTO_APPROVE_THRESHOLD", config.hitl.auto_approve_threshold)
        )
        config.hitl.require_feedback = os.getenv("HITL_REQUIRE_FEEDBACK", "true").lower() == "true"

        # Data Processing
        config.data_processing.max_file_size_mb = float(
            os.getenv("MAX_FILE_SIZE_MB", config.data_processing.max_file_size_mb)
        )
        formats = os.getenv("SUPPORTED_FORMATS")
        if formats:
            config.data_processing.supported_formats = formats.split(",")

        # Logging
        config.logging.level = os.getenv("LOG_LEVEL", config.logging.level)
        config.logging.file = os.getenv("LOG_FILE", config.logging.file)
        config.logging.format = os.getenv("LOG_FORMAT", config.logging.format)

        # Metadata
        config.metadata.enable_versioning = os.getenv("ENABLE_VERSIONING", "true").lower() == "true"
        config.metadata.enable_tracking = os.getenv("ENABLE_METADATA_TRACKING", "true").lower() == "true"
        config.metadata.metadata_path = os.getenv("METADATA_PATH", config.metadata.metadata_path)

        # API
        config.api.host = os.getenv("API_HOST", config.api.host)
        config.api.port = int(os.getenv("API_PORT", config.api.port))
        config.api.reload = os.getenv("API_RELOAD", "false").lower() == "true"
        config.api.debug = os.getenv("API_DEBUG", "false").lower() == "true"

        # UI
        config.ui.server_port = int(os.getenv("STREAMLIT_SERVER_PORT", config.ui.server_port))
        config.ui.server_address = os.getenv("STREAMLIT_SERVER_ADDRESS", config.ui.server_address)

        return config

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.data_dir,
            self.config_dir,
            self.output_dir,
            Path(self.vectordb.path).parent,
            Path(self.logging.file).parent if self.logging.file else None,
            self.metadata.metadata_path,
        ]

        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
        _config.ensure_directories()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
    _config.ensure_directories()


__all__ = [
    "Config",
    "LLMConfig",
    "DatabaseConfig",
    "VectorDBConfig",
    "AgentConfig",
    "HITLConfig",
    "DataProcessingConfig",
    "LoggingConfig",
    "MetadataConfig",
    "APIConfig",
    "UIConfig",
    "get_config",
    "set_config",
]
