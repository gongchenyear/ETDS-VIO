# ETDS-VIO

This directory is the release scope referenced by the manuscript for
`ETDS-VIO: a sensor-uncertainty-aware dual-screening visual--inertial odometry for robust pose measurement in dynamic environments`.

The public code release is limited to `src/VINS-Fusion`. Other local workspace directories used during experimentation are not part of the repository release.

## What Is in `src/VINS-Fusion`

`src/VINS-Fusion` is a research fork of VINS-Fusion for RGB-D + IMU dynamic-scene odometry. Relative to upstream VINS-Fusion, it adds:

1. An event-triggered dual-screening front end for dynamic environments.
   This includes pre-RANSAC depth-guided static-candidate screening and post-RANSAC suspicious-inlier verification.
2. An integrated DRT initialization path.
   The DRT code used by this fork is vendored under `vins_estimator/src/drt_init`, and `DRTAdapter` bridges its output into the VINS-Fusion estimator state.

The active runtime package name is `vins_drt_dyn`.

## Main Code Locations

- Dynamic front end:
  `VINS-Fusion/vins_estimator/src/featureTracker/feature_tracker.cpp`
  `VINS-Fusion/vins_estimator/src/featureTracker/feature_tracker.h`
- Estimator and DRT handoff:
  `VINS-Fusion/vins_estimator/src/estimator/estimator.cpp`
  `VINS-Fusion/vins_estimator/src/estimator/estimator.h`
  `VINS-Fusion/vins_estimator/src/estimator/drt_adapter.cpp`
  `VINS-Fusion/vins_estimator/src/estimator/drt_adapter.h`
- Parameter declaration and loading:
  `VINS-Fusion/vins_estimator/src/estimator/parameters.h`
  `VINS-Fusion/vins_estimator/src/estimator/parameters.cpp`
- Vendored DRT implementation:
  `VINS-Fusion/vins_estimator/src/drt_init/`
- Paper configurations:
  `VINS-Fusion/config/openloris/`
  `VINS-Fusion/config/vcu_rvi/`

## Scope of the Paper Release

The released repository provides:

- The modified VINS-Fusion source code used in the paper
- The integrated DRT code used by this fork
- The main OpenLORIS and VCU-RVI configuration files referenced by the manuscript

It does not, by itself, include:

- Public dataset files
- Local experiment output directories
- Local plotting scripts
- Manuscript workspace files outside `src/VINS-Fusion`

## Tested Configurations Used in the Paper

Representative configuration files are:

- `VINS-Fusion/config/openloris/market_mono_imu_ours_full.yaml`
- `VINS-Fusion/config/openloris/market_mono_imu_ours_nodrt.yaml`
- `VINS-Fusion/config/openloris/market_mono_imu_ours_full_nopre.yaml`
- `VINS-Fusion/config/openloris/market_mono_imu_ours_full_nopost.yaml`
- `VINS-Fusion/config/vcu_rvi/dynamic_mono_imu_ours_full.yaml`
- `VINS-Fusion/config/vcu_rvi/dynamic_mono_imu_ours_nodrt.yaml`

The most complete OpenLORIS configuration is `VINS-Fusion/config/openloris/market_mono_imu_ours_full.yaml`.

## Build Notes

This source tree is intended to live inside a catkin workspace.

The build files still contain machine-local dependency paths inherited from the development environment. Before building on another machine, update those paths to match the local installation:

- `VINS-Fusion/camera_models/CMakeLists.txt`
- `VINS-Fusion/vins_estimator/CMakeLists.txt`
- `VINS-Fusion/loop_fusion/CMakeLists.txt`
- `VINS-Fusion/global_fusion/CMakeLists.txt`

In particular, check the local paths used for:

- Ceres include and CMake package lookup
- Sophus headers if Sophus is not found from the active environment

The YAML files also contain local output directories such as `output_path` and `pose_graph_save_path`. Change them to valid writable paths on the target machine before running experiments.

## Dependencies

The development environment used for the paper was:

- Ubuntu 20.04
- ROS Noetic
- Eigen 3.3.7
- Ceres 1.14
- OpenCV 4.2

The experiments reported in the manuscript were run on CPU only.

## Build

From the catkin workspace root:

```bash
catkin_make
source devel/setup.bash
```

If the build fails on a new machine, first verify that the CMake dependency paths described above have been changed to the local environment.

## Run

From the catkin workspace root:

```bash
rosrun vins_drt_dyn vins_drt_dyn_node /path/to/src/VINS-Fusion/config/openloris/market_mono_imu_ours_full.yaml
```

Optional RViz:

```bash
roslaunch vins_drt_dyn vins_rviz.launch
```

## Datasets

The paper evaluates on public datasets:

- OpenLORIS-Scene
- VCU-RVI Benchmark

This release does not redistribute those datasets. They must be obtained from their original public sources and replayed separately.

## Notes for Reuse

This is a research code release rather than a cleaned general-purpose SDK. When reusing it, review:

- CMake dependency paths
- YAML output paths
- ROS topic names in the selected configuration file
- Sensor calibration files referenced by the selected configuration

## License

`src/VINS-Fusion` inherits the original VINS-Fusion licensing context. See `VINS-Fusion/LICENCE` and the package metadata files before redistribution or derivative use.
