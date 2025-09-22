"""Config Management"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import os
import json
import yaml
from pathlib import Path


class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    database: str = Field(..., description="Database name")
    max_connections: int = Field(default=10, ge=1, le=100)

    @validator('host')
    def validate_host(cls, v):
        if not v or v.isspace():
            raise ValueError('Host cannot be empty')
        return v.strip()

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    password: Optional[str] = None
    db: int = Field(default=0, ge=0)
    max_connections: int = Field(default=20, ge=1)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", regex=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_file_size: str = "10MB"
    backup_count: int = Field(default=5, ge=1)


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = False
    secret_key: str = Field(..., min_length=32)
    allowed_hosts: List[str] = Field(default_factory=list)
    cors_origins: List[str] = Field(default_factory=list)

    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v


class AppConfig(BaseModel):
    """Main application configuration."""
    app_name: str = "MyApp"
    version: str = "1.0.0"
    environment: str = Field(default="development", regex=r"^(development|staging|production)$")
    debug: bool = True

    # Sub-configurations
    database: DatabaseConfig
    redis: Optional[RedisConfig] = None
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig

    # Feature flags
    features: Dict[str, bool] = Field(default_factory=dict)

    class Config:
        env_prefix = "APP_"
        case_sensitive = False

    @validator('debug')
    def debug_false_in_production(cls, v, values):
        if values.get('environment') == 'production' and v:
            raise ValueError('Debug must be False in production')
        return v


class ConfigLoader:
    """Configuration loader that supports multiple formats."""

    @staticmethod
    def load_from_file(file_path: str) -> AppConfig:
        """Load configuration from a file."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        content = path.read_text()

        if path.suffix.lower() == '.json':
            data = json.loads(content)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            data = yaml.safe_load(content)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        return AppConfig(**data)

    @staticmethod
    def load_from_env() -> AppConfig:
        """Load configuration from environment variables."""
        # Create minimal required config from env vars
        config_data = {
            "database": {
                "username": os.getenv("DB_USERNAME", "user"),
                "password": os.getenv("DB_PASSWORD", "password"),
                "database": os.getenv("DB_NAME", "mydb")
            },
            "api": {
                "secret_key": os.getenv("SECRET_KEY", "your-secret-key-here-must-be-32-chars-min")
            }
        }

        return AppConfig(**config_data)

    @staticmethod
    def create_example_config(file_path: str) -> None:
        """Create an example configuration file."""
        example_config = {
            "app_name": "MyApp",
            "version": "1.0.0",
            "environment": "development",
            "debug": True,
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "myuser",
                "password": "mypassword",
                "database": "mydatabase",
                "max_connections": 10
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "max_connections": 20
            },
            "logging": {
                "level": "INFO",
                "file": "app.log",
                "max_file_size": "10MB",
                "backup_count": 5
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
                "secret_key": "your-super-secret-key-here-32-chars-minimum",
                "allowed_hosts": ["localhost", "127.0.0.1"],
                "cors_origins": ["http://localhost:3000"]
            },
            "features": {
                "enable_caching": True,
                "enable_metrics": False,
                "enable_auth": True
            }
        }

        path = Path(file_path)

        if path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(example_config, f, indent=2)
        elif path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'w') as f:
                yaml.dump(example_config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        print(f"Example config created: {file_path}")


def demonstrate_config_management():
    """Demonstrate configuration management."""

    print("=== Config Management Demo ===")

    # Create example config
    ConfigLoader.create_example_config("example_config.json")

    # Load from file
    try:
        config = ConfigLoader.load_from_file("example_config.json")
        print(f"\nLoaded config for {config.app_name} v{config.version}")
        print(f"Environment: {config.environment}")
        print(f"Database connection: {config.database.connection_string}")
        print(f"API will run on {config.api.host}:{config.api.port}")
        print(f"Features enabled: {[k for k, v in config.features.items() if v]}")

        # Access nested configuration
        if config.redis:
            print(f"Redis configured on {config.redis.host}:{config.redis.port}")

        print(f"Logging level: {config.logging.level}")

    except Exception as e:
        print(f"Error loading config: {e}")

    # Load from environment (with defaults)
    print("\n--- Loading from Environment ---")
    try:
        env_config = ConfigLoader.load_from_env()
        print(f"Environment config loaded: {env_config.app_name}")
        print(f"Database: {env_config.database.connection_string}")
    except Exception as e:
        print(f"Error loading from env: {e}")

    # Demonstrate validation
    print("\n--- Validation Demo ---")
    try:
        invalid_config = AppConfig(
            database=DatabaseConfig(
                username="user",
                password="pass",
                database="db"
            ),
            api=APIConfig(
                secret_key="too-short"  # This will fail validation
            )
        )
    except Exception as e:
        print(f"Validation caught error: {e}")

    # Clean up
    try:
        os.remove("example_config.json")
    except:
        pass


if __name__ == "__main__":
    demonstrate_config_management()