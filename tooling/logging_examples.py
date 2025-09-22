"""Logging Examples"""

import logging
import logging.handlers
import json
import sys
from typing import Dict, Any
from datetime import datetime
import traceback
import contextvars


# Context variable for request tracking
request_id: contextvars.ContextVar[str] = contextvars.ContextVar('request_id', default='')


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add request ID if available
        if request_id.get():
            log_entry['request_id'] = request_id.get()

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add custom fields from extra
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class ContextualLogger:
    """Logger wrapper that adds contextual information."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}

    def set_context(self, **kwargs):
        """Set context that will be added to all log messages."""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear the current context."""
        self.context.clear()

    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log with context information."""
        extra_fields = {**self.context}
        if 'extra_fields' in kwargs:
            extra_fields.update(kwargs.pop('extra_fields'))

        extra = logging.LoggerAdapter(self.logger, {}).extra or {}
        extra['extra_fields'] = extra_fields

        self.logger.log(level, msg, *args, extra=extra, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)


class PerformanceLogger:
    """Logger for performance monitoring."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_duration(self, operation: str, duration: float, **kwargs):
        """Log the duration of an operation."""
        extra_fields = {
            'operation': operation,
            'duration_ms': round(duration * 1000, 2),
            'performance': True,
            **kwargs
        }

        self.logger.info(
            f"Operation '{operation}' completed in {duration:.3f}s",
            extra={'extra_fields': extra_fields}
        )

    def log_metric(self, metric_name: str, value: float, unit: str = '', **kwargs):
        """Log a metric value."""
        extra_fields = {
            'metric_name': metric_name,
            'metric_value': value,
            'metric_unit': unit,
            'metric': True,
            **kwargs
        }

        self.logger.info(
            f"Metric {metric_name}: {value} {unit}",
            extra={'extra_fields': extra_fields}
        )


class AuditLogger:
    """Logger for audit trails."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_action(self, action: str, user_id: str = None, resource: str = None,
                   success: bool = True, **kwargs):
        """Log an auditable action."""
        extra_fields = {
            'action': action,
            'user_id': user_id,
            'resource': resource,
            'success': success,
            'audit': True,
            **kwargs
        }

        level = logging.INFO if success else logging.WARNING
        status = "succeeded" if success else "failed"

        self.logger.log(
            level,
            f"Action '{action}' {status} for user {user_id}",
            extra={'extra_fields': extra_fields}
        )


def setup_logging(level: str = "INFO", structured: bool = False,
                  file_path: str = None) -> logging.Logger:
    """Set up logging configuration."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if structured:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    root_logger.addHandler(console_handler)

    # File handler if specified
    if file_path:
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        if structured:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        root_logger.addHandler(file_handler)

    return root_logger


def simulate_web_request(user_id: str, endpoint: str):
    """Simulate processing a web request with contextual logging."""
    import time
    import uuid
    import random

    # Set request context
    req_id = str(uuid.uuid4())[:8]
    request_id.set(req_id)

    # Create contextual logger
    logger = ContextualLogger('webapp')
    logger.set_context(
        user_id=user_id,
        endpoint=endpoint,
        request_id=req_id
    )

    # Create specialized loggers
    perf_logger = PerformanceLogger(logging.getLogger('performance'))
    audit_logger = AuditLogger(logging.getLogger('audit'))

    try:
        logger.info(f"Processing request to {endpoint}")

        # Simulate some processing time
        start_time = time.time()
        time.sleep(random.uniform(0.1, 0.3))

        # Log some metrics
        perf_logger.log_metric('memory_usage', random.uniform(50, 90), 'MB')
        perf_logger.log_metric('cpu_usage', random.uniform(10, 80), '%')

        # Simulate business logic
        if endpoint == '/users':
            logger.info("Fetching user list", extra_fields={'query_count': 3})
            audit_logger.log_action('list_users', user_id, 'users')

        elif endpoint == '/profile':
            logger.info("Loading user profile", extra_fields={'cache_hit': True})
            audit_logger.log_action('view_profile', user_id, f'user:{user_id}')

        elif endpoint == '/admin':
            if user_id != 'admin':
                logger.warning("Unauthorized access attempt")
                audit_logger.log_action('access_admin', user_id, 'admin', success=False)
                raise PermissionError("Access denied")
            else:
                logger.info("Admin access granted")
                audit_logger.log_action('access_admin', user_id, 'admin')

        # Log performance
        duration = time.time() - start_time
        perf_logger.log_duration('request_processing', duration,
                                endpoint=endpoint, status='success')

        logger.info("Request completed successfully")

    except Exception as e:
        logger.error("Request failed", exc_info=True,
                    extra_fields={'error_type': type(e).__name__})
        audit_logger.log_action('request_error', user_id, endpoint, success=False)
        raise
    finally:
        logger.clear_context()


def demonstrate_logging():
    """Demonstrate various logging patterns."""

    print("=== Logging Examples Demo ===")

    # Set up structured logging
    setup_logging(level="INFO", structured=True)

    print("\n--- Basic Structured Logging ---")
    logger = logging.getLogger('demo')
    logger.info("Application started", extra={'extra_fields': {'version': '1.0.0'}})
    logger.warning("Low disk space", extra={'extra_fields': {'disk_usage': 85}})

    print("\n--- Contextual Logging ---")
    ctx_logger = ContextualLogger('service')
    ctx_logger.set_context(service='user-service', version='2.1.0')
    ctx_logger.info("Service initialized")
    ctx_logger.debug("Database connection established")

    print("\n--- Performance Logging ---")
    perf_logger = PerformanceLogger(logging.getLogger('performance'))
    perf_logger.log_duration('database_query', 0.025, query_type='SELECT')
    perf_logger.log_metric('active_connections', 42, 'connections')

    print("\n--- Audit Logging ---")
    audit_logger = AuditLogger(logging.getLogger('audit'))
    audit_logger.log_action('user_login', 'user123', 'auth_system')
    audit_logger.log_action('failed_login', 'hacker', 'auth_system', success=False)

    print("\n--- Web Request Simulation ---")
    try:
        simulate_web_request('user123', '/users')
        simulate_web_request('user456', '/profile')
        simulate_web_request('guest', '/admin')  # This will fail
    except PermissionError:
        pass  # Expected

    print("\n--- Error Logging with Exception ---")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.error("Division by zero error", exc_info=True,
                    extra={'extra_fields': {'operation': 'divide'}})

    print("\n--- Different Log Levels ---")
    setup_logging(level="DEBUG", structured=False)  # Switch to readable format

    demo_logger = logging.getLogger('levels')
    demo_logger.debug("Debug message - detailed info")
    demo_logger.info("Info message - general info")
    demo_logger.warning("Warning message - something to watch")
    demo_logger.error("Error message - something went wrong")
    demo_logger.critical("Critical message - system failure")


if __name__ == "__main__":
    demonstrate_logging()