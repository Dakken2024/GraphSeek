"""
Retrieval cache for improving performance and reducing redundant computations.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import hashlib
import time
from collections import OrderedDict


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    query: str
    results: Any
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self) -> None:
        """Update access time and count."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def is_expired(self, ttl: float) -> bool:
        """Check if entry has expired based on TTL."""
        return (time.time() - self.created_at) > ttl


class RetrievalCache:
    """LRU cache for retrieval results with TTL support."""
    
    def __init__(
        self, 
        max_size: int = 1000,
        default_ttl: float = 3600.0,  # 1 hour
    ) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, query: str, **kwargs) -> str:
        """Generate a cache key from query and parameters."""
        key_parts = [query]
        
        # Add sorted kwargs to key
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend(f"{k}={v}" for k, v in sorted_kwargs)
        
        combined = "|".join(key_parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """Get cached results for a query."""
        key = self._generate_key(query, **kwargs)
        
        if key in self._cache:
            entry = self._cache[key]
            
            # Check TTL
            if entry.is_expired(self.default_ttl):
                self.delete(key)
                self._misses += 1
                return None
            
            # Update access stats
            entry.touch()
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._hits += 1
            return entry.results
        
        self._misses += 1
        return None
    
    def set(
        self, 
        query: str, 
        results: Any, 
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Cache results for a query."""
        key = self._generate_key(query, **kwargs)
        
        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        entry = CacheEntry(
            key=key,
            query=query,
            results=results,
            metadata=metadata or {},
        )
        
        # Override TTL if specified
        if ttl is not None:
            entry.metadata["custom_ttl"] = ttl
        
        self._cache[key] = entry
        return key
    
    def delete(self, key: str) -> bool:
        """Delete a cache entry by key."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if self._cache:
            # Pop the first item (least recently used)
            self._cache.popitem(last=False)
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired(self.default_ttl)
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "default_ttl_seconds": self.default_ttl,
            "memory_estimate_kb": self._estimate_memory_usage(),
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in KB (rough approximation)."""
        # Very rough estimate based on number of entries
        # In production, use sys.getsizeof or memory profiler
        return len(self._cache) * 0.5  # Assume ~500 bytes per entry on average
    
    def get_entry(self, key: str) -> Optional[CacheEntry]:
        """Get a specific cache entry by key."""
        return self._cache.get(key)
    
    def list_keys(self, limit: int = 100) -> List[str]:
        """List cache keys (most recent first)."""
        keys = list(self._cache.keys())
        return list(reversed(keys))[:limit]
