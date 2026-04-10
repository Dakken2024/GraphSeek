"""
Monitoring utilities for tracking performance and metrics.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import time
from functools import wraps
from contextlib import contextmanager
import threading


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_time: float = 0.0
    errors: int = 0
    
    @property
    def avg_time(self) -> float:
        """Calculate average execution time."""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    def record(self, execution_time: float, success: bool = True) -> None:
        """Record a new execution."""
        self.count += 1
        self.total_time += execution_time
        self.last_time = execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        if not success:
            self.errors += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "name": self.name,
            "count": self.count,
            "avg_time_ms": round(self.avg_time * 1000, 2),
            "min_time_ms": round(self.min_time * 1000, 2) if self.min_time != float('inf') else 0,
            "max_time_ms": round(self.max_time * 1000, 2),
            "last_time_ms": round(self.last_time * 1000, 2),
            "error_count": self.errors,
            "error_rate_percent": round((self.errors / self.count * 100) if self.count > 0 else 0, 2),
        }


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self) -> None:
        self._metrics: Dict[str, OperationMetrics] = {}
        self._raw_points: List[MetricPoint] = []
        self._lock = threading.Lock()
    
    def record(
        self, 
        operation_name: str, 
        execution_time: float,
        success: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric."""
        with self._lock:
            # Update aggregated metrics
            if operation_name not in self._metrics:
                self._metrics[operation_name] = OperationMetrics(name=operation_name)
            
            self._metrics[operation_name].record(execution_time, success)
            
            # Store raw data point
            if tags:
                point = MetricPoint(
                    name=operation_name,
                    value=execution_time,
                    tags=tags,
                )
                self._raw_points.append(point)
                
                # Limit raw points to prevent memory issues
                if len(self._raw_points) > 10000:
                    self._raw_points = self._raw_points[-5000:]
    
    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for one or all operations."""
        with self._lock:
            if operation_name:
                if operation_name in self._metrics:
                    return self._metrics[operation_name].to_dict()
                return {}
            
            return {
                name: metrics.to_dict()
                for name, metrics in self._metrics.items()
            }
    
    def reset(self, operation_name: Optional[str] = None) -> None:
        """Reset metrics."""
        with self._lock:
            if operation_name:
                if operation_name in self._metrics:
                    del self._metrics[operation_name]
            else:
                self._metrics.clear()
                self._raw_points.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        with self._lock:
            total_operations = sum(m.count for m in self._metrics.values())
            total_errors = sum(m.errors for m in self._metrics.values())
            
            return {
                "total_operations": total_operations,
                "total_errors": total_errors,
                "overall_error_rate": round(
                    (total_errors / total_operations * 100) if total_operations > 0 else 0, 2
                ),
                "operations": [m.to_dict() for m in self._metrics.values()],
            }


class PerformanceTracker:
    """Context manager and decorator for tracking performance."""
    
    def __init__(
        self, 
        metrics_collector: Optional[MetricsCollector] = None,
        operation_name: Optional[str] = None,
    ) -> None:
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.operation_name = operation_name
        self._start_time: Optional[float] = None
    
    def start(self, operation_name: Optional[str] = None) -> None:
        """Start timing."""
        self._start_time = time.time()
        if operation_name:
            self.operation_name = operation_name
    
    def stop(
        self, 
        success: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ) -> float:
        """Stop timing and record metric."""
        if self._start_time is None:
            return 0.0
        
        execution_time = time.time() - self._start_time
        self._start_time = None
        
        if self.operation_name:
            self.metrics_collector.record(
                self.operation_name,
                execution_time,
                success,
                tags,
            )
        
        return execution_time
    
    @contextmanager
    def track(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for tracking operations."""
        self.operation_name = operation_name
        self.start()
        try:
            yield
            self.stop(success=True, tags=tags)
        except Exception as e:
            self.stop(success=False, tags=tags)
            raise
    
    def __call__(self, func: Callable):
        """Decorator for tracking function performance."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = self.operation_name or func.__name__
            with self.track(operation_name):
                return func(*args, **kwargs)
        return wrapper


class Monitor:
    """Main monitoring class that combines all monitoring capabilities."""
    
    _instance: Optional['Monitor'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'Monitor':
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self.metrics_collector = MetricsCollector()
        self.performance_tracker = PerformanceTracker(self.metrics_collector)
        self._active_timers: Dict[str, float] = {}
        self._initialized = True
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._active_timers[name] = time.time()
    
    def stop_timer(self, name: str, success: bool = True) -> float:
        """Stop a named timer and record metric."""
        if name not in self._active_timers:
            return 0.0
        
        start_time = self._active_timers.pop(name)
        execution_time = time.time() - start_time
        
        self.metrics_collector.record(name, execution_time, success)
        return execution_time
    
    @contextmanager
    def measure(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Measure an operation using context manager."""
        self.start_timer(name)
        try:
            yield
            self.stop_timer(name, success=True)
        except Exception:
            self.stop_timer(name, success=False)
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self.metrics_collector.get_summary()
    
    def get_operation_metrics(self, operation_name: str) -> Dict[str, Any]:
        """Get metrics for a specific operation."""
        return self.metrics_collector.get_metrics(operation_name)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics_collector.reset()
        self._active_timers.clear()
    
    def track_function(self, name: Optional[str] = None):
        """Decorator factory for tracking functions."""
        def decorator(func: Callable):
            operation_name = name or f"function:{func.__name__}"
            tracker = PerformanceTracker(self.metrics_collector, operation_name)
            return tracker(func)
        return decorator
