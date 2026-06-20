# ETDS-VIO on VINS-Fusion

This repository is the code release for the paper
`ETDS-VIO: a sensor-uncertainty-aware dual-screening visual--inertial odometry for robust pose measurement in dynamic environments`.

It is a research fork of VINS-Fusion for RGB-D + IMU dynamic-scene odometry. The public release scope is limited to this `src/VINS-Fusion` tree. Other local workspace directories used during experimentation are not part of the release.

## What This Repository Contains

Relative to upstream VINS-Fusion, this fork adds two components in `vins_estimator`:

1. An event-triggered dual-screening front end for dynamic environments.
   It adds pre-RANSAC depth-guided static-candidate screening and post-RANSAC suspicious-inlier verification.
2. An integrated DRT initialization path.
   The DRT code is vendored under `vins_estimator/src/drt_init`, and `DRTAdapter` bridges its output into the VINS-Fusion estimator state.

The active runtime package name is `vins_drt_dyn`.

## Main Code Locations

- Dynamic front end:
  `vins_estimator/src/featureTracker/feature_tracker.cpp`
  `vins_estimator/src/featureTracker/feature_tracker.h`
- Estimator and DRT handoff:
  `vins_estimator/src/estimator/estimator.cpp`
  `vins_estimator/src/estimator/estimator.h`
  `vins_estimator/src/estimator/drt_adapter.cpp`
  `vins_estimator/src/estimator/drt_adapter.h`
- Parameter declaration and loading:
  `vins_estimator/src/estimator/parameters.h`
  `vins_estimator/src/estimator/parameters.cpp`
- Vendored DRT implementation:
  `vins_estimator/src/drt_init/`
- Paper configurations:
  `config/openloris/`
  `config/vcu_rvi/`

## Scope of the Paper Release

This repository provides:

- The modified VINS-Fusion source code used in the paper
- The integrated DRT code used by this fork
- The main OpenLORIS and VCU-RVI configuration files referenced by the manuscript

This repository does not, by itself, include:

- Public dataset files
- Local experiment output directories
- Local plotting scripts or manuscript workspace files outside `src/VINS-Fusion`

## Tested Configurations Used in the Paper

Representative configuration files are:

- `config/openloris/market_mono_imu_ours_full.yaml`
- `config/openloris/market_mono_imu_ours_nodrt.yaml`
- `config/openloris/market_mono_imu_ours_full_nopre.yaml`
- `config/openloris/market_mono_imu_ours_full_nopost.yaml`
- `config/vcu_rvi/dynamic_mono_imu_ours_full.yaml`
- `config/vcu_rvi/dynamic_mono_imu_ours_nodrt.yaml`

The most complete OpenLORIS configuration is `config/openloris/market_mono_imu_ours_full.yaml`.

## Build Notes

This codebase is a catkin workspace subtree and follows the original VINS-Fusion package layout.

The build files in this release still contain machine-local dependency paths inherited from the working environment used during development. Before building on another machine, update those paths to match the local installation:

- `camera_models/CMakeLists.txt`
- `vins_estimator/CMakeLists.txt`
- `loop_fusion/CMakeLists.txt`
- `global_fusion/CMakeLists.txt`

In particular, check the local paths used for:

- Ceres include and CMake package lookup
- Sophus headers if Sophus is not found from the active environment

The YAML files also contain local output directories such as `output_path` and `pose_graph_save_path`. These should be changed to valid writable paths on the target machine before running experiments.

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

This repository does not redistribute those datasets. They must be obtained from their original public sources and replayed separately.

## Notes for Reuse

This is a research code release rather than a cleaned general-purpose SDK. The package names, local path assumptions, and configuration defaults follow the development workspace used for the reported experiments. When reusing the code, review:

- CMake dependency paths
- YAML output paths
- ROS topic names in the selected configuration file
- Sensor calibration files referenced by the selected configuration

## License

This repository inherits the original VINS-Fusion licensing context. See `LICENCE` and the individual package metadata files before redistribution or derivative use.
