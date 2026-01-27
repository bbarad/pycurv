"""
LRU cache for geodesic distance computations.

This module provides a caching mechanism for Dijkstra's shortest path computations
on graphs, reducing redundant calculations when adjacent vertices share overlapping
distance neighborhoods.

Author: Performance optimization module
"""

from collections import OrderedDict
import numpy as np
from graph_tool.topology import shortest_distance


class GeodesicDistanceCache:
    """
    LRU (Least Recently Used) cache for geodesic distance arrays.

    Caches the results of shortest_distance computations to avoid redundant
    Dijkstra calculations. Adjacent vertices often have overlapping neighborhoods,
    so caching distance arrays can significantly reduce computation time.

    Attributes:
        graph: The graph-tool Graph object
        weights: Edge weight property map for distance calculations
        max_cache_size: Maximum number of distance arrays to keep in cache
        g_max: Maximum geodesic distance (used as max_dist in shortest_distance)
        _cache: OrderedDict implementing the LRU cache
    """

    def __init__(self, graph, max_cache_size=10000, g_max=None):
        """
        Initialize the geodesic distance cache.

        Args:
            graph: graph_tool.Graph object
            max_cache_size: Maximum number of cached distance arrays (default 10000)
            g_max: Maximum geodesic distance for shortest_distance computations
        """
        self.graph = graph
        self.weights = graph.ep.distance
        self.max_cache_size = max_cache_size
        self.g_max = g_max
        self._cache = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get_distances(self, source_vertex):
        """
        Get the distance array from a source vertex to all other vertices.

        If the distances are cached, returns the cached result (and moves it
        to the end of the LRU cache). Otherwise, computes the distances using
        shortest_distance and caches the result.

        Args:
            source_vertex: The source vertex (graph_tool.Vertex or int index)

        Returns:
            numpy.ndarray of distances from source to all vertices
        """
        vertex_idx = int(source_vertex)

        if vertex_idx in self._cache:
            # Cache hit - move to end (most recently used)
            self._cache.move_to_end(vertex_idx)
            self._hits += 1
            return self._cache[vertex_idx]

        # Cache miss - compute and store
        self._misses += 1

        # Get the actual vertex object if we received an int
        if isinstance(source_vertex, int):
            source_vertex = self.graph.vertex(source_vertex)

        dist_v = shortest_distance(
            self.graph,
            source=source_vertex,
            target=None,
            weights=self.weights,
            max_dist=self.g_max
        )
        dist_array = dist_v.get_array().copy()  # Copy to avoid reference issues

        # Store in cache
        self._cache[vertex_idx] = dist_array

        # Evict oldest entries if cache is too large
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)

        return dist_array

    def get_neighbors(self, source_vertex, g_max):
        """
        Get geodesic neighbors within a maximum distance.

        Args:
            source_vertex: The source vertex (graph_tool.Vertex or int index)
            g_max: Maximum geodesic distance for neighbors

        Returns:
            dict mapping neighbor vertex index to geodesic distance
        """
        dist_array = self.get_distances(source_vertex)
        vertex_idx = int(source_vertex)

        # Find all vertices within g_max (excluding source itself)
        idxs = np.where((dist_array <= g_max) & (dist_array > 0))[0]
        return {int(idx): float(dist_array[idx]) for idx in idxs}

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self):
        """
        Get cache statistics.

        Returns:
            dict with 'size', 'hits', 'misses', and 'hit_rate'
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            'size': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate
        }
