from pathlib2 import Path, PurePath
from curvature_calculation import (new_workflow,
                                   extract_curvatures_after_new_workflow)
from distances_calculation import distances_and_thicknesses_calculation
RADIUS_HIT = 10


# For smoothed cER
def task_calculate_curvatures():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/4Javier/new_curvature_ria/"
    pixel_size = 1.368
    radius_hit = RADIUS_HIT
    methods = ["VV"]
    lbl = 2  # cER
    holes = 3
    min_component = 100

    for condition in ["TCB", "WT", "IST2", "SCS"]:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold = str(subfold_p)
            seg_files = list(subfold_p.glob('**/*.mrc'))
            if len(seg_files) > 0:
                seg_file_p = seg_files[0]
                seg_file = str(seg_file_p)
                seg_filename = str(PurePath(seg_file_p).name)
                tomo = "{}{}{}".format(condition, subfold.split('_')[-2],
                                       subfold.split('_')[-1])
                base_filename = "{}_cER".format(tomo)
                subfold += '/'
                target_base = "{}{}.VV_area2_rh{}_epsilon0_eta0".format(
                    subfold, base_filename, radius_hit)
                yield {'name': tomo,
                       # 'verbosity': 2,
                       'actions': [
                           (new_workflow,
                            [subfold, base_filename, pixel_size, radius_hit], {
                                'methods': methods,
                                'seg_file': seg_filename,
                                'label': lbl,
                                'holes': holes,
                                'min_component': min_component,
                                'cores': 4
                            })
                        ],
                       'file_dep': [seg_file],
                       'targets': [
                           "{}.gt".format(target_base),
                           "{}.vtp".format(target_base)
                       ],
                       # force doit to always mark the task as up-to-date
                       # (unless target removed)
                       'uptodate': [True]
                       }
            else:
                print("No segmentation file was found.")


def task_extract_curvatures():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/4Javier/new_curvature_ria/"
    radius_hit = RADIUS_HIT
    methods = ["VV"]

    for condition in ["TCB", "WT", "IST2", "SCS"]:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold = str(subfold_p)
            tomo = "{}{}{}".format(condition, subfold.split('_')[-2],
                                   subfold.split('_')[-1])
            base_filename = "{}_cER".format(tomo)
            subfold += '/'
            target_base = "{}{}.VV_area2_rh{}_epsilon0_eta0".format(
                subfold, base_filename, radius_hit)
            yield {'name': tomo,
                   # 'verbosity': 2,
                   'actions': [
                       (extract_curvatures_after_new_workflow,
                        [subfold, base_filename, radius_hit], {
                            'methods': methods,
                            'exclude_borders': 0
                        }),
                       (extract_curvatures_after_new_workflow,
                        [subfold, base_filename, radius_hit], {
                            'methods': methods,
                            'exclude_borders': 1
                        })
                    ],
                   'file_dep': [
                       "{}.gt".format(target_base),
                       "{}.vtp".format(target_base)
                   ],
                   'targets': [
                       "{}.csv".format(target_base),
                       "{}_excluding1borders.csv".format(target_base)
                   ],
                   # force doit to always mark the task as up-to-date (unless
                   # target removed)
                   'uptodate': [True]
                   }


def task_calculate_distances():
    # constant parameters for all conditions and segmentations:
    base_fold = "/fs/pool/pool-ruben/Maria/4Javier/smooth_distances/"

    for condition in ["TCB", "WT", "IST2", "SCS"]:
        fold = "{}{}/".format(base_fold, condition)
        fold_p = Path(fold)
        # iterate over all subfolders
        for subfold_p in [x for x in fold_p.iterdir() if x.is_dir()]:
            subfold = str(subfold_p)
            subfold_name = subfold_p.name  # subfold.split('/')[-1]
            date, _, lamella, tomo = subfold_name.split('_')
            base_filename = "{}_{}_{}_{}".format(
                condition, date, lamella, tomo)
            subfold += '/'
            segmentation_file_p = list(subfold_p.glob('*.mrc'))[0].name
            segmentation_file = str(segmentation_file_p)
            target_base = "{}{}".format(subfold, base_filename)
            yield {'name': base_filename,
                   # 'verbosity': 2,
                   'actions': [
                       (distances_and_thicknesses_calculation,
                        [subfold, segmentation_file, base_filename], {
                            'radius_hit': RADIUS_HIT
                        })
                    ],
                   'targets': [
                       "{}.cER.distancesFromPM.csv".format(target_base),
                       "{}.innercER.thicknesses.csv".format(target_base)
                   ],
                   # force doit to always mark the task as up-to-date (unless
                   # target removed)
                   'uptodate': [True]
                   }


# Note: to run one condition only, e.g. TCB: doit *:TCB*
