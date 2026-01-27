"""
PyCurv package can be used to analyze the membrane-bound ribosome density,
calculate intermembrane distances and estimate curvature of membranes in
cryo-electron tomograms or other volumetric data sources.
"""

# Configure OpenMP for parallel graph operations before importing graph_tool
# This is safe even if OpenMP is not available - the env vars are simply ignored
try:
    from .parallel_config import configure_openmp
    configure_openmp()
except Exception:
    pass  # Silently continue if configuration fails

from .pexceptions import *
from .pycurv_io import *
from .graphs import SegmentationGraph
from .ribosome_density import *
from .surface_graphs import *
from .vector_voting import *
from .tomogram_batch_processing import split_segmentation
from .distances_between_surfaces import *
from .surface import *
from .linalg import *
