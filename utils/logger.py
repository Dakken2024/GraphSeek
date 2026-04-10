"""
Logging utilities for the application.
"""
import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'asctime'
            ):
                try:
                    json.dumps(value)  # Check if serializable
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    structured: bool = False,
    include_console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with optional file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file
        structured: Whether to use JSON formatting
        include_console: Whether to include console handler
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    
    # Console handler
    if include_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Use structured format for files by default
        file_formatter = StructuredFormatter() if not structured else formatter
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding contextual information to logs."""
    
    def __init__(self, logger: logging.Logger, **context: Any) -> None:
        self.logger = logger
        self.context = context
        self.old_fields: Dict[str, Any] = {}
    
    def __enter__(self) -> logging.Logger:
        """Add context to logger."""
        # Store old values
        for key in self.context:
            if hasattr(self.logger, key):
                self.old_fields[key] = getattr(self.logger, key)
        
        # Add new context (using extra dict pattern)
        # Note: This is a simplified approach
        # For production, consider using logging adapters
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore logger state."""
        # Cleanup if needed
        pass


def log_execution_time(logger: logging.Logger, operation: str):
    """Decorator factory for logging function execution time."""
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(
                    f"{operation} completed in {elapsed:.3f}s",
                    extra={"operation": operation, "duration_ms": elapsed * 1000},
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"{operation} failed after {elapsed:.3f}s: {str(e)}",
                    extra={"operation": operation, "duration_ms": elapsed * 1000},
                )
                raise
        return wrapper
    return decorator
