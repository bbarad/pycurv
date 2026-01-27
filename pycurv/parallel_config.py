"""
Configuration for parallel computation and optimized Dijkstra algorithms.

This module provides:
1. OpenMP configuration for graph-tool's parallel operations
2. Conversion utilities between graph-tool and scipy sparse matrices
3. Batch Dijkstra computation using scipy for efficient multi-source queries

Note: All functions are designed to degrade gracefully if optional dependencies
(OpenMP, scipy) are not available.

Author: Performance optimization module
"""

import os
import multiprocessing


def configure_openmp(num_threads=None):
    """
    Configure OpenMP environment variables for graph-tool parallelization.

    This should be called before importing graph_tool for the settings to take
    effect. Graph-tool uses OpenMP for parallel algorithms like shortest_distance.

    If OpenMP is not available (e.g., graph-tool compiled without it), these
    environment variables are simply ignored - no error occurs.

    Args:
        num_threads: Number of threads to use. If None, uses all available CPUs.

    Returns:
        int: The number of threads configured
    """
    if num_threads is None:
        try:
            num_threads = multiprocessing.cpu_count()
        except NotImplementedError:
            num_threads = 1

    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['OMP_SCHEDULE'] = 'dynamic'

    return num_threads


def is_scipy_available():
    """Check if scipy is available for optimized batch operations."""
    try:
        import scipy.sparse
        import scipy.sparse.csgraph
        return True
    except ImportError:
        return False


def graph_to_scipy_sparse(graph, weights):
    """
    Convert a graph-tool graph to a scipy sparse CSR matrix.

    Creates a symmetric adjacency matrix suitable for scipy's graph algorithms.

    Args:
        graph: graph_tool.Graph object (undirected)
        weights: Edge weight PropertyMap (e.g., graph.ep.distance)

    Returns:
        scipy.sparse.csr_matrix: Symmetric sparse adjacency matrix

    Raises:
        ImportError: If scipy is not installed
    """
    from scipy.sparse import csr_matrix
    import numpy as np

    num_v = graph.num_vertices()
    edges = graph.get_edges()
    edge_weights = weights.a

    # Create symmetric adjacency (add both directions for undirected graph)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([edge_weights, edge_weights])

    return csr_matrix((data, (row, col)), shape=(num_v, num_v))


def batch_dijkstra_scipy(graph, source_indices, weights, max_dist=None):
    """
    Batch multi-source Dijkstra using scipy's optimized implementation.

    For large batches (100+ sources), scipy's dijkstra can be more efficient
    than repeated graph-tool calls due to better memory locality and reduced
    Python overhead.

    Args:
        graph: graph_tool.Graph object
        source_indices: List/array of source vertex indices
        weights: Edge weight PropertyMap
        max_dist: Maximum distance cutoff (None for unlimited)

    Returns:
        dict: Mapping from source index to numpy array of distances
    """
    from scipy.sparse.csgraph import dijkstra
    import numpy as np

    adj_matrix = graph_to_scipy_sparse(graph, weights)

    # scipy.dijkstra returns a 2D array: [num_sources, num_vertices]
    dist_matrix = dijkstra(
        adj_matrix,
        directed=False,
        indices=source_indices,
        limit=max_dist if max_dist is not None else np.inf
    )

    # Convert to dictionary format matching graph-tool interface
    return {src: dist_matrix[i] for i, src in enumerate(source_indices)}


def batch_dijkstra_graphtool(graph, source_indices, weights, max_dist=None):
    """
    Batch multi-source Dijkstra using graph-tool.

    For small batches, graph-tool's implementation may be faster due to
    avoiding the overhead of sparse matrix conversion.

    Args:
        graph: graph_tool.Graph object
        source_indices: List/array of source vertex indices
        weights: Edge weight PropertyMap
        max_dist: Maximum distance cutoff (None for unlimited)

    Returns:
        dict: Mapping from source index to numpy array of distances
    """
    from graph_tool.topology import shortest_distance

    results = {}
    for idx in source_indices:
        dist_v = shortest_distance(
            graph,
            source=graph.vertex(idx),
            target=None,
            weights=weights,
            max_dist=max_dist
        )
        results[idx] = dist_v.get_array().copy()

    return results


def batch_dijkstra(graph, source_indices, weights, max_dist=None, threshold=100):
    """
    Compute batch Dijkstra distances using the optimal backend.

    Automatically chooses between scipy (for large batches) and graph-tool
    (for small batches) based on the number of sources. Falls back to
    graph-tool if scipy is not available.

    Args:
        graph: graph_tool.Graph object
        source_indices: List/array of source vertex indices
        weights: Edge weight PropertyMap
        max_dist: Maximum distance cutoff (None for unlimited)
        threshold: Batch size threshold for switching to scipy (default 100)

    Returns:
        dict: Mapping from source index to numpy array of distances
    """
    if len(source_indices) >= threshold and is_scipy_available():
        try:
            return batch_dijkstra_scipy(graph, source_indices, weights, max_dist)
        except Exception:
            # Fall back to graph-tool if scipy fails for any reason
            pass
    return batch_dijkstra_graphtool(graph, source_indices, weights, max_dist)


# Configure OpenMP on module import
_openmp_threads = configure_openmp()
