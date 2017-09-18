import numpy as np
import math
import os

from pysurf_compact import pysurf_io as io
from pysurf_compact import run_gen_surface, pexceptions

"""A set of functions and classes for generating artificial segmentation volumes
(masks) of geometrical objects."""


class SphereMask(object):
    """
    A class for generating a sphere mask.
    """
    @staticmethod
    def generate_sphere_mask(r=10, box=23):
        # TODO docstring
        if 2 * r + 1 > box:
            error_msg = "Sphere diameter has to fit into the box."
            raise pexceptions.PySegInputError(
                expr='SphereMask.generate_sphere_mask', msg=error_msg)
        # Create a 3D grid with center (0, 0, 0) in the middle of the box
        low = - math.floor(box / 2.0)
        high = math.ceil(box / 2.0)
        xx, yy, zz = np.mgrid[low:high, low:high, low:high]

        # Calculate squared distances from the center
        sq_dist_from_center = xx ** 2 + yy ** 2 + zz ** 2

        # Threshold using squared radius to generate a filled sphere
        sphere = (sq_dist_from_center <= r ** 2).astype(int)
        return sphere


class CylinderMask(object):
    """
    A class for generating a cylinder mask.
    """
    @staticmethod
    def generate_cylinder_mask(r=10, h=21, box=27, t=0, opened=False):
        # TODO docstring
        if 2 * r + 1 > box:
            error_msg = "Cylinder diameter has to fit into the box."
            raise pexceptions.PySegInputError(
                expr='CylinderMask.generate_cylinder_mask', msg=error_msg)
        if h > box:
            error_msg = "Cylinder high has to be maximum the box size."
            raise pexceptions.PySegInputError(
                expr='CylinderMask.generate_cylinder_mask', msg=error_msg)

        # Create a 2D grid with center (0, 0) in the middle
        # (slice through the box in XY plane)
        low = - math.floor(box / 2.0)
        high = math.ceil(box / 2.0)
        xx, yy = np.mgrid[low:high, low:high]

        # Calculate squared distances from the center
        sq_dist_from_center = xx ** 2 + yy ** 2

        # Threshold using squared radius to generate a filled circle
        circle = (sq_dist_from_center <= r ** 2).astype(int)

        # Generate a cylinder consisting of N=box circles stacked in Z dimension
        cylinder = np.zeros(shape=(box, box, box))
        bottom = int(math.floor((box - h) / 2.0))
        top = bottom + h
        for zz in range(bottom, top):
            cylinder[:, :, zz] = circle

        if t > 0:  # Generate a hollow cylinder with thickness t
            # Generate an inner cylinder smaller by t ...
            if opened is True:  # ... in radius only ("open" cylinder)
                inner_cylinder = CylinderMask.generate_cylinder_mask(
                    r=r-t, h=h, box=box)
            else:  # ... in radius and by 2t in high ("closed" cylinder)
                inner_cylinder = CylinderMask.generate_cylinder_mask(
                    r=r-t, h=h-2*t, box=box)

            # Subtract it from the bigger cylinder
            cylinder = cylinder - inner_cylinder

        return cylinder


def main():
    """
    Main function generating some sphere and cylinder masks and from them signed
    surfaces.

    Returns:
        None
    """
    fold = "/fs/pool/pool-ruben/Maria/curvature/synthetic_volumes/"
    if not os.path.exists(fold):
        os.makedirs(fold)

    # # Generate a filled sphere mask
    # sm = SphereMask()
    # sphere_r10_box23 = sm.generate_sphere_mask()
    # io.save_numpy(sphere_r10_box23, fold + "sphere_r10_box23.mrc")
    #
    # # From the mask, generate a surface (is correct)
    # run_gen_surface(sphere_r10_box23, fold + "sphere_r10_box23")  # 2088 cells

    # Generate a hollow cylinder mask
    cm = CylinderMask()
    thickness = 1
    # cylinder_r10_h21_box27 = cm.generate_cylinder_mask(
    #     r=10, h=21, box=27, t=thickness)
    # io.save_numpy(cylinder_r10_h21_box27,
    #               "{}cylinder_r10_h21_box27_t{}.mrc".format(fold, thickness))
    cylinder_r10_h21_box27_open = cm.generate_cylinder_mask(
        r=10, h=21, box=27, t=thickness, opened=True)
    io.save_numpy(
        cylinder_r10_h21_box27_open,
        "{}cylinder_r10_h21_box27_t{}_open.mrc".format(fold, thickness))
    cylinder_r10_h27_box27 = cm.generate_cylinder_mask(
        r=10, h=27, box=27, t=thickness)
    io.save_numpy(cylinder_r10_h27_box27,
                  "{}cylinder_r10_h27_box27_t{}.mrc".format(fold, thickness))
    # cylinder_r10_h27_box27_open = cm.generate_cylinder_mask(
    #     r=10, h=27, box=27, t=thickness, open=True)
    # io.save_numpy(
    #     cylinder_r10_h27_box27_open,
    #     "{}cylinder_r10_h27_box27_t{}_open.mrc".format(fold, thickness))

    # From the mask, generate a surface
    # run_gen_surface(cylinder_r10_h21_box27,  # 3592 cells
    #                 "{}cylinder_r10_h21_box27_t{}".format(fold, thickness))
    # run_gen_surface(
    #     cylinder_r10_h21_box27_open,  # 2880 cells
    #     "{}cylinder_r10_h21_box27_t{}_open".format(fold, thickness))
    # run_gen_surface(cylinder_r10_h27_box27,  # 4310 cells
    #                 "{}cylinder_r10_h27_box27_t{}".format(fold, thickness))
    # run_gen_surface(
    #     cylinder_r10_h27_box27_open,  # 3672 cells, little unevenness
    #     "{}cylinder_r10_h27_box27_t{}_open".format(fold, thickness))
    run_gen_surface(
        cylinder_r10_h27_box27,
        "{}cylinder_r10_h27_box27_t{}_open2".format(fold, thickness),
        other_mask=cylinder_r10_h21_box27_open)  # 3107 cells
    # cylinder_r10_h25_box27_t1_open2.surface.vtp with MAX_DIST_SURF=1.5 and
    # delete cell if all points outside mask (count > 2), 2879 cells


if __name__ == "__main__":
    main()
