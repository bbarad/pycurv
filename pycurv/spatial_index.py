"""
Spatial indexing for fast Euclidean neighbor queries.

This module provides a KD-tree based spatial index for quickly filtering
candidate geodesic neighbors. Since Euclidean distance <= geodesic distance,
vertices beyond Euclidean radius g_max cannot be within geodesic radius g_max.

Author: Performance optimization module
"""

import numpy as np
from scipy.spatial import cKDTree


class SpatialIndex:
    """
    KD-tree based spatial index for fast Euclidean neighbor queries.

    This index enables efficient pre-filtering of candidate geodesic neighbors.
    Since geodesic distance >= Euclidean distance on a surface, any vertex
    with Euclidean distance > g_max cannot have geodesic distance <= g_max.

    By first finding Euclidean neighbors (O(log N + k) with KD-tree), we can
    significantly reduce the number of vertices for which geodesic distances
    must be computed.

    Attributes:
        coordinates: (N, 3) numpy array of vertex coordinates
        kdtree: scipy.spatial.cKDTree for fast spatial queries
        vertex_indices: Array mapping KD-tree indices to graph vertex indices
    """

    def __init__(self, graph):
        """
        Build the spatial index from a graph's vertex coordinates.

        Args:
            graph: graph_tool.Graph with vp.xyz vertex property
        """
        num_vertices = graph.num_vertices()

        # Extract all vertex coordinates into a numpy array
        # graph.vp.xyz returns vector<float> for each vertex
        self.coordinates = np.zeros((num_vertices, 3), dtype=np.float64)
        self.vertex_indices = np.zeros(num_vertices, dtype=np.int64)

        for i, v in enumerate(graph.vertices()):
            self.coordinates[i] = graph.vp.xyz[v]
            self.vertex_indices[i] = int(v)

        # Build KD-tree for O(log N) spatial queries
        self.kdtree = cKDTree(self.coordinates)

        # Create reverse mapping: graph vertex index -> array index
        self._vertex_to_array_idx = {
            int(vidx): i for i, vidx in enumerate(self.vertex_indices)
        }

    def find_euclidean_neighbors(self, vertex_idx, radius):
        """
        Find all vertices within Euclidean distance 'radius' from a vertex.

        This is used to pre-filter candidates for geodesic neighbor search.
        Since geodesic_distance >= euclidean_distance, any vertex with
        euclidean_distance > radius cannot have geodesic_distance <= radius.

        Args:
            vertex_idx: Index of the source vertex in the graph
            radius: Maximum Euclidean distance

        Returns:
            numpy array of candidate vertex indices (graph indices)
        """
        # Get the array index for this vertex
        array_idx = self._vertex_to_array_idx.get(vertex_idx)
        if array_idx is None:
            return np.array([], dtype=np.int64)

        # Query KD-tree for neighbors within radius
        point = self.coordinates[array_idx]
        neighbor_array_indices = self.kdtree.query_ball_point(point, radius)

        # Convert array indices back to graph vertex indices
        return self.vertex_indices[neighbor_array_indices]

    def find_euclidean_neighbors_from_coords(self, coords, radius):
        """
        Find all vertices within Euclidean distance 'radius' from coordinates.

        Args:
            coords: (3,) array-like of source coordinates
            radius: Maximum Euclidean distance

        Returns:
            numpy array of candidate vertex indices (graph indices)
        """
        neighbor_array_indices = self.kdtree.query_ball_point(coords, radius)
        return self.vertex_indices[neighbor_array_indices]

    def get_vertex_coords(self, vertex_idx):
        """
        Get coordinates of a vertex by its graph index.

        Args:
            vertex_idx: Graph vertex index

        Returns:
            (3,) numpy array of coordinates, or None if not found
        """
        array_idx = self._vertex_to_array_idx.get(vertex_idx)
        if array_idx is None:
            return None
        return self.coordinates[array_idx]
