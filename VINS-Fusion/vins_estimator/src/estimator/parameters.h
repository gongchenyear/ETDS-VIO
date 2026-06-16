/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>
#include <mutex>

using namespace std;

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_F = 1000;
// 移除：MAX_REASONABLE_VELOCITY - 动态检测不再使用速度门控
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern double INIT_REL_POSE_MIN_PARALLAX_PX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;
extern double IMU_ACC_SCALE;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern double TD;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern int ROW, COL;
extern int NUM_OF_CAM;
extern int STEREO;
extern int USE_IMU;
extern int MULTIPLE_THREAD;
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int FLOW_BACK;
extern double FLOW_BACK_THRESHOLD;

// Dynamic detection parameters (from previous step, ensure these are present)
extern int ENABLE_DYNAMIC_DETECTION;
// 移除：DYNAMIC_V_THRESHOLD_LOW_SPEED - 不再使用速度门控
extern double DYNAMIC_DSS_THRESHOLD_FRONTEND;
// 已删除：DYNAMIC_DSS_THRESHOLD_FRONTEND_RESET - 冗余参数，已被POST_TRIGGER_RESET_RATIO_THRESHOLD替代
// 已删除：DYNAMIC_N_RESET_FRAMES - 冗余参数，已被DYNAMIC_MIN_RESET_FRAMES_INTERNAL替代
extern double DYNAMIC_W_FLOW;
extern double DYNAMIC_W_EPOPOLAR;
// 🗑️ DYNAMIC_W_MULTIFRAME已移除

// Metric-specific thresholds (sigmoid parameters removed - no longer used)
extern double DYNAMIC_T_FLOW_DIFF_SQ;
extern double DYNAMIC_T_EPIPOLAR_RES_SQ;
// 🗑️ Multiframe相关参数已移除
extern double DYNAMIC_MIN_PERCENT_INCONSISTENT_OBS_FOR_TRACK;

// ACE修复：恢复前端动态检测仍需要的参数
extern int ENABLE_DSS_TEMPORAL_ANOMALY;
extern double DSS_JUMP_THRESHOLD;
extern double ANOMALY_BOOST_FACTOR;
extern int DYNAMIC_MIN_TRIGGER_FRAMES;
extern int DYNAMIC_MIN_RESET_FRAMES_INTERNAL;
extern double DYNAMIC_DSS_HYSTERESIS_RATIO;
extern double DYNAMIC_TEMPORAL_CONSISTENCY_THRESHOLD;

// 保留的必要参数
extern double DYNAMIC_DEFAULT_DEPTH;                   // Flow计算备用深度（仍在使用）
extern int DSS_INTELLIGENT_SCORING_DEBUG;              // 调试级别控制（仍在使用）

// Event-Triggered Dynamic Filtering - Phase 1: Simplified Trigger Mechanism
extern bool TRIGGER_ENABLED;                           // 触发机制总开关
extern double TRACKING_RATE_THRESHOLD;                 // 光流追踪成功率阈值
extern int TRIGGER_K_ON_FRAMES;                        // 触发平滑帧数（快速进入）
extern int TRIGGER_K_OFF_FRAMES;                       // 取消触发平滑帧数（保守退出）

// DG-RANSAC Parameters - Module 2
extern bool ENABLE_DG_RANSAC;                          // DG-RANSAC总开关
extern double DG_RANSAC_DEPTH_NOISE_SIGMA;             // 深度噪声标准差
extern double DG_RANSAC_CONF_MIN_THRESHOLD;            // 最低置信度阈值
extern double DG_RANSAC_CONF_HIGH_THRESHOLD;           // 高置信度阈值
extern double DG_RANSAC_CONF_SIGMA_RATE;               // 深度变化率标准差 (m/s)
extern double DG_RANSAC_CONF_SIGMA_RELATIVE;           // 相对变化标准差
extern int DG_RANSAC_LOCAL_WINDOW_SIZE;                // 局部滤波窗口大小
extern int DG_RANSAC_FILTER_KERNEL;                    // 中值滤波核大小
extern double DG_RANSAC_HIGH_CONF_FILTER_THRESHOLD;    // 高置信度筛选阈值
extern int DG_RANSAC_MIN_POINTS;                       // RANSAC最少点数
extern bool DG_RANSAC_FALLBACK_ENABLED;                // 启用回退到传统RANSAC
extern double DG_RANSAC_EPIPOLAR_THRESHOLD;            // 对极误差阈值（像素）
extern double DG_RANSAC_DEPTH_ERROR_THRESHOLD;         // 深度误差阈值（归一化）
extern int DG_RANSAC_MIN_ITERATIONS;                   // 最小迭代次数
extern int DG_RANSAC_MAX_ITERATIONS;                   // 最大迭代次数
extern bool DG_RANSAC_ADAPTIVE_ITERATIONS;             // 启用自适应迭代
extern double DG_RANSAC_EARLY_TERMINATION_RATIO;       // 提前终止内点比例
extern int K_ON_FRAMES;                                // 触发平滑帧数（快速进入）
extern int K_OFF_FRAMES;                               // 取消触发平滑帧数（保守退出）

// Logging Configuration
extern bool ENABLE_DETAILED_LOG;                       // 启用详细调试日志
extern int LOG_INTERVAL;                               // 统计日志输出间隔（帧）
extern bool LOG_TRIGGER_EVENTS;                        // 记录触发状态切换
extern bool LOG_STATISTICS;                            // 记录统计信息
extern bool LOG_PERFORMANCE;                           // 记录性能计时
extern bool LOG_DEPTH_PREPROCESS;                      // 记录深度预处理详情
extern bool LOG_CONFIDENCE_COMPUTE;                    // 记录置信度计算详情
extern bool LOG_RANSAC_DETAILS;                        // 记录RANSAC详细信息
extern bool LOG_DEPTH_MOTION;                          // 记录深度运动验证详情

// Phase 3: 深度变化率验证参数
extern int MIN_DEPTH_SAMPLES;
extern double DEPTH_RATE_TOLERANCE_SIGMA;
extern double DEPTH_NOISE_STD;
extern double DEPTH_MIN_RANGE;
extern double DEPTH_MAX_RANGE;
extern std::string DEPTH_TOPIC;

// Module Control Switches
extern bool ENABLE_PRE_FILTERING;            // 前置流程：RANSAC前的深度引导（Module 2）
extern bool ENABLE_DRT_FALLBACK;             // DRT初始化与回退总开关

// Depth Confidence Computation Parameters
extern double DEPTH_CONFIDENCE_SIGMA_RELATIVE;  // 相对变化标准差
extern double DEPTH_CONFIDENCE_SIGMA_RATE;      // 深度变化率标准差 (m/s)
extern double DEPTH_CONFIDENCE_LAMBDA_DEPTH;    // 深度自适应尺度系数
extern double DEPTH_CONFIDENCE_LAMBDA_FOV;      // 视场位置自适应尺度系数
extern double DEPTH_CONFIDENCE_MIN_THRESHOLD;   // 最低置信度阈值
extern double DEPTH_CONFIDENCE_HIGH_THRESHOLD;  // 高置信度阈值
extern int DEPTH_FILTER_SIZE;                   // 深度滤波窗口大小

// Module 3: Post-Verification Parameters
extern bool ENABLE_POST_VERIFICATION;           // Module 3开关
extern double POST_VERIFY_SUSPICIOUS_THRESHOLD; // 可疑内点置信度阈值
extern int POST_VERIFY_MAX_COUNT;               // 最大验证数量

// Phase 3: IMU速度全局变量（线程安全）
extern Eigen::Vector3d LATEST_CAMERA_VELOCITY;
extern Eigen::Vector3d LATEST_CAMERA_ANGULAR_VELOCITY;
extern Eigen::Matrix3d LATEST_CAMERA_ROTATION;
extern bool HAS_VALID_CAMERA_VELOCITY;
extern std::mutex VELOCITY_MUTEX;

// Phase 2: Suspicious Point Expansion Parameters
extern double BASE_EXPAND_RADIUS;
extern double DENSE_EXPAND_RADIUS;
extern int OUTLIER_DENSITY_THRESHOLD;
extern double MAX_SUSPICIOUS_RATIO;

// Temporary compatibility parameters (to avoid compilation errors)
extern double FRONTEND_DYNAMIC_RATIO_THRESHOLD;
extern double FRONTEND_RESET_RATIO_THRESHOLD;
extern int FRONTEND_DYNAMIC_DEBUG_LEVEL;

// DRT Prior Covariance Parameters
extern double DRT_PRIOR_POS_COV;
extern double DRT_PRIOR_ROT_COV;
extern double DRT_PRIOR_VEL_COV;
extern double DRT_PRIOR_ACC_BIAS_COV;
extern double DRT_PRIOR_GYR_BIAS_COV;

extern int MAX_VINS_INIT_FAILURES;


void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
