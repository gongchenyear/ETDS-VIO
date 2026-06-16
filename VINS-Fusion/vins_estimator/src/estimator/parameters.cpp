/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

#include <algorithm>
#include <cmath>

double INIT_DEPTH;
double MIN_PARALLAX;
double INIT_REL_POSE_MIN_PARALLAX_PX = 30.0;
double ACC_N, ACC_W;
double GYR_N, GYR_W;
double IMU_ACC_SCALE = 1.0;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;
double FLOW_BACK_THRESHOLD;

// Dynamic detection parameters
int ENABLE_DYNAMIC_DETECTION;
double DYNAMIC_DSS_THRESHOLD_FRONTEND;
double DYNAMIC_W_FLOW;
double DYNAMIC_W_EPOPOLAR;

// DRT Prior Covariance Parameters (Default values)
double DRT_PRIOR_POS_COV = 1.0;
double DRT_PRIOR_ROT_COV = 0.2;
double DRT_PRIOR_VEL_COV = 1.0;
double DRT_PRIOR_ACC_BIAS_COV = 0.05;
double DRT_PRIOR_GYR_BIAS_COV = 0.01;
int MAX_VINS_INIT_FAILURES = 8;


// Metric-specific thresholds (sigmoid parameters removed - no longer used)
double DYNAMIC_T_FLOW_DIFF_SQ;
double DYNAMIC_T_EPIPOLAR_RES_SQ;

double DYNAMIC_MIN_PERCENT_INCONSISTENT_OBS_FOR_TRACK;

// DSS时序异常检测参数
int ENABLE_DSS_TEMPORAL_ANOMALY;
double DSS_JUMP_THRESHOLD;
double ANOMALY_BOOST_FACTOR;

// ACE v14小动态物体优化：新增单个特征点异常阈值参数
double DYNAMIC_SUSPICIOUS_FEATURE_THRESHOLD;


// ACE v26关键硬编码参数暴露：将影响动态检测性能的关键硬编码参数暴露到配置文件
// ACE 多帧验证完善：重新引入DYNAMIC_MIN_TRIGGER_FRAMES参数，实现触发多帧验证
int DYNAMIC_MIN_TRIGGER_FRAMES;           // 触发所需的最小连续帧数
int DYNAMIC_MIN_RESET_FRAMES_INTERNAL;    // 内部重置逻辑的最小帧数
double DYNAMIC_DSS_HYSTERESIS_RATIO;      // DSS滞后比例
// ACE 中间帧逻辑清理：移除DYNAMIC_INTERMEDIATE_FRAME_INTERVAL参数（中间帧不进入滑动窗口）


// ACE v30全面参数暴露：将剩余硬编码参数暴露到配置文件
double DYNAMIC_TEMPORAL_CONSISTENCY_THRESHOLD;  // 时序一致性阈值

double DYNAMIC_DEPTH_THRESHOLD;                 // DSS深度有效性阈值，与VINS对齐
// 已移除未使用的特征点生命周期管理参数定义

// ACE 硬编码数组参数暴露：敏感性分析和调试参数
std::vector<double> DYNAMIC_TRIGGER_SENSITIVITY_THRESHOLDS;  // 触发敏感性分析阈值数组
std::vector<double> DYNAMIC_RESET_SENSITIVITY_THRESHOLDS;    // 重置敏感性分析阈值数组
std::vector<double> DYNAMIC_FLOW_SENSITIVITY_THRESHOLDS;     // Flow阈值敏感性分析数组

int DYNAMIC_CLEANUP_INTERVAL;                                // DSS时序过滤器清理间隔（帧数）
int DYNAMIC_WEIGHT_DEBUG_COUNT;                              // 权重配置调试输出的特征点数量

// ACE 硬编码参数完全暴露：统计分析相关参数
double DYNAMIC_HIGH_SCORE_THRESHOLD;            // 高分指标统计阈值（原硬编码0.5）


// 统一评分范围参数：支持多种归一化方法
double DYNAMIC_DEFAULT_DEPTH;                   // Flow计算默认深度
double UNIFIED_DSS_FLOW_THRESHOLD;              // Flow指标统一阈值（线性映射）
double UNIFIED_DSS_EPIPOLAR_THRESHOLD;          // Epipolar指标统一阈值（线性映射）

double UNIFIED_DSS_ANOMALY_SCORE;               // 统一异常评分
double POST_TRIGGER_FLOW_SUSPICIOUS_THRESHOLD;  // POST_TRIGGER状态Flow指标可疑判定阈值
double POST_TRIGGER_RESET_RATIO_THRESHOLD;      // POST_TRIGGER状态可疑比例重置判定阈值



// ACE调试参数
int DSS_INTELLIGENT_SCORING_DEBUG;              // 调试级别控制（0=关闭，1=基础，2=详细）

// Event-Triggered Dynamic Filtering - Phase 1: Simplified Trigger Mechanism
bool TRIGGER_ENABLED = true;                     // 触发机制总开关
double TRACKING_RATE_THRESHOLD = 0.7;            // 光流追踪成功率阈值（默认值，会被配置文件覆盖）
int TRIGGER_K_ON_FRAMES = 1;                     // 触发平滑帧数（快速进入）
int TRIGGER_K_OFF_FRAMES = 10;                   // 取消触发平滑帧数（保守退出）

// DG-RANSAC Parameters - Module 2
bool ENABLE_DG_RANSAC = true;                    // DG-RANSAC总开关
double DG_RANSAC_DEPTH_NOISE_SIGMA = 0.15;       // 深度噪声标准差
double DG_RANSAC_CONF_MIN_THRESHOLD = 0.3;       // 最低置信度阈值
double DG_RANSAC_CONF_HIGH_THRESHOLD = 0.7;      // 高置信度阈值
double DG_RANSAC_CONF_SIGMA_RATE = 0.5;          // 深度变化率标准差 (m/s)
double DG_RANSAC_CONF_SIGMA_RELATIVE = 0.05;     // 相对变化标准差 (5%)
int DG_RANSAC_LOCAL_WINDOW_SIZE = 9;             // 局部滤波窗口大小 (9x9) - 减小以提高性能
int DG_RANSAC_FILTER_KERNEL = 5;                 // 中值滤波核大小 (5x5) - 减小滤波核
double DG_RANSAC_HIGH_CONF_FILTER_THRESHOLD = 0.5; // 高置信度筛选阈值
int DG_RANSAC_MIN_POINTS = 8;                    // RANSAC最少点数
bool DG_RANSAC_FALLBACK_ENABLED = true;          // 启用回退到传统RANSAC
double DG_RANSAC_EPIPOLAR_THRESHOLD = 1.0;       // 对极误差阈值（像素）
double DG_RANSAC_DEPTH_ERROR_THRESHOLD = 0.2;    // 深度误差阈值（归一化）
int DG_RANSAC_MIN_ITERATIONS = 50;               // 最小迭代次数
int DG_RANSAC_MAX_ITERATIONS = 300;              // 最大迭代次数
bool DG_RANSAC_ADAPTIVE_ITERATIONS = true;       // 启用自适应迭代
double DG_RANSAC_EARLY_TERMINATION_RATIO = 0.8;  // 提前终止内点比例

// Logging Configuration
bool ENABLE_DETAILED_LOG = false;                // 启用详细调试日志
int LOG_INTERVAL = 100;                          // 统计日志输出间隔（帧）
bool LOG_TRIGGER_EVENTS = true;                  // 记录触发状态切换
bool LOG_STATISTICS = true;                      // 记录统计信息
bool LOG_PERFORMANCE = true;                     // 记录性能计时
bool LOG_DEPTH_PREPROCESS = false;               // 记录深度预处理详情
bool LOG_CONFIDENCE_COMPUTE = false;             // 记录置信度计算详情
bool LOG_RANSAC_DETAILS = false;                 // 记录RANSAC详细信息
bool LOG_DEPTH_MOTION = false;                   // 记录深度运动验证详情

// Phase 2: Suspicious Point Expansion Parameters
double BASE_EXPAND_RADIUS = 50.0;
double DENSE_EXPAND_RADIUS = 100.0;
int OUTLIER_DENSITY_THRESHOLD = 3;
double MAX_SUSPICIOUS_RATIO = 0.5;

// Phase 3: 深度图处理参数
std::string DEPTH_TOPIC;
double DEPTH_MIN_RANGE = 0.1;  // 放宽最小范围，D435i在0.1-0.3m也有部分有效数据
double DEPTH_MAX_RANGE = 8.0;  // 增大最大范围，与配置文件保持一致

// Phase 3: 深度变化率验证参数
int MIN_DEPTH_SAMPLES = 2;
double DEPTH_RATE_TOLERANCE_SIGMA = 3.0;
double DEPTH_NOISE_STD = 0.02;

// Phase 3: IMU速度全局变量（线程安全）
Eigen::Vector3d LATEST_CAMERA_VELOCITY = Eigen::Vector3d::Zero();
Eigen::Vector3d LATEST_CAMERA_ANGULAR_VELOCITY = Eigen::Vector3d::Zero();
Eigen::Matrix3d LATEST_CAMERA_ROTATION = Eigen::Matrix3d::Identity();
bool HAS_VALID_CAMERA_VELOCITY = false;
std::mutex VELOCITY_MUTEX;

// Module Control Switches
bool ENABLE_PRE_FILTERING = false;           // 前置流程：默认关闭（暂未完全实现）
bool ENABLE_DRT_FALLBACK = true;             // DRT初始化与回退默认开启

// Depth Confidence Computation Parameters
double DEPTH_CONFIDENCE_SIGMA_RELATIVE = 0.08;  // 相对变化标准差（8%）- 放宽以适应D435i噪声
double DEPTH_CONFIDENCE_SIGMA_RATE = 0.5;       // 深度变化率标准差 (m/s) - 放宽以适应相机运动
double DEPTH_CONFIDENCE_LAMBDA_DEPTH = 0.0;     // 深度自适应残差尺度系数，0表示退化为固定sigma
double DEPTH_CONFIDENCE_LAMBDA_FOV = 0.0;       // 视场位置自适应残差尺度系数
double DEPTH_CONFIDENCE_MIN_THRESHOLD = 0.3;    // 最低置信度阈值
double DEPTH_CONFIDENCE_HIGH_THRESHOLD = 0.7;   // 高置信度阈值
int DEPTH_FILTER_SIZE = 5;                      // 深度滤波窗口大小

// Module 3: Post-Verification Parameters
bool ENABLE_POST_VERIFICATION = false;          // Module 3开关（默认关闭）
double POST_VERIFY_SUSPICIOUS_THRESHOLD = 0.5;  // 可疑内点置信度阈值
int POST_VERIFY_MAX_COUNT = 20;                 // 最大验证数量

// Temporary compatibility parameters (to avoid compilation errors)
double FRONTEND_DYNAMIC_RATIO_THRESHOLD = 0.15;
double FRONTEND_RESET_RATIO_THRESHOLD = 0.05;
int FRONTEND_DYNAMIC_DEBUG_LEVEL = 1;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

namespace {

int sanitizeMinInt(int value, int min_value, const char* name)
{
    if (value < min_value) {
        ROS_WARN("[ConfigCheck] %s=%d is invalid, clamping to %d", name, value, min_value);
        return min_value;
    }
    return value;
}

int sanitizeOddInt(int value, int min_value, const char* name)
{
    int sanitized = sanitizeMinInt(value, min_value, name);
    if (sanitized % 2 == 0) {
        ROS_WARN("[ConfigCheck] %s=%d should be odd, adjusting to %d", name, sanitized, sanitized + 1);
        sanitized += 1;
    }
    return sanitized;
}

double sanitizePositiveDouble(double value, double fallback, const char* name)
{
    if (!std::isfinite(value) || value <= 0.0) {
        ROS_WARN("[ConfigCheck] %s=%.6f is invalid, using %.6f", name, value, fallback);
        return fallback;
    }
    return value;
}

double sanitizeNonNegativeDouble(double value, double fallback, const char* name)
{
    if (!std::isfinite(value) || value < 0.0) {
        ROS_WARN("[ConfigCheck] %s=%.6f is invalid, using %.6f", name, value, fallback);
        return fallback;
    }
    return value;
}

double sanitizeRangeDouble(double value, double min_value, double max_value,
                           double fallback, const char* name)
{
    if (!std::isfinite(value) || value < min_value || value > max_value) {
        ROS_WARN("[ConfigCheck] %s=%.6f is out of range [%.6f, %.6f], using %.6f",
                 name, value, min_value, max_value, fallback);
        return fallback;
    }
    return value;
}

void warnBoolConflict(const char* lhs_name, bool lhs_value,
                      const char* rhs_name, bool rhs_value,
                      const char* effective_name)
{
    if (lhs_value != rhs_value) {
        ROS_WARN("[ConfigCheck] Conflict: %s=%s, %s=%s. Effective value follows %s.",
                 lhs_name, lhs_value ? "ON" : "OFF",
                 rhs_name, rhs_value ? "ON" : "OFF",
                 effective_name);
    }
}

void validateExperimentParameters()
{
    TRACKING_RATE_THRESHOLD = sanitizeRangeDouble(
        TRACKING_RATE_THRESHOLD, 0.0, 1.0, 0.7, "dynamic_trigger.tracking_rate_threshold");
    TRIGGER_K_ON_FRAMES = sanitizeMinInt(
        TRIGGER_K_ON_FRAMES, 1, "dynamic_trigger.k_on_frames");
    TRIGGER_K_OFF_FRAMES = sanitizeMinInt(
        TRIGGER_K_OFF_FRAMES, 1, "dynamic_trigger.k_off_frames");

    DG_RANSAC_LOCAL_WINDOW_SIZE = sanitizeOddInt(
        DG_RANSAC_LOCAL_WINDOW_SIZE, 3, "dg_ransac.local_window_size");
    DG_RANSAC_FILTER_KERNEL = sanitizeOddInt(
        DG_RANSAC_FILTER_KERNEL, 1, "dg_ransac.filter_kernel");
    if (DG_RANSAC_FILTER_KERNEL > DG_RANSAC_LOCAL_WINDOW_SIZE) {
        ROS_WARN("[ConfigCheck] dg_ransac.filter_kernel=%d is larger than local_window_size=%d, clamping.",
                 DG_RANSAC_FILTER_KERNEL, DG_RANSAC_LOCAL_WINDOW_SIZE);
        DG_RANSAC_FILTER_KERNEL = DG_RANSAC_LOCAL_WINDOW_SIZE;
        if (DG_RANSAC_FILTER_KERNEL % 2 == 0) {
            DG_RANSAC_FILTER_KERNEL = std::max(1, DG_RANSAC_FILTER_KERNEL - 1);
        }
    }
    DG_RANSAC_HIGH_CONF_FILTER_THRESHOLD = sanitizeRangeDouble(
        DG_RANSAC_HIGH_CONF_FILTER_THRESHOLD, 0.0, 1.0, 0.5,
        "dg_ransac.high_conf_filter_threshold");
    DG_RANSAC_MIN_POINTS = sanitizeMinInt(
        DG_RANSAC_MIN_POINTS, 8, "dg_ransac.min_points_for_ransac");
    DG_RANSAC_EPIPOLAR_THRESHOLD = sanitizePositiveDouble(
        DG_RANSAC_EPIPOLAR_THRESHOLD, 1.0, "dg_ransac.epipolar_threshold");
    DG_RANSAC_DEPTH_ERROR_THRESHOLD = sanitizePositiveDouble(
        DG_RANSAC_DEPTH_ERROR_THRESHOLD, 0.2, "dg_ransac.depth_error_threshold");
    DG_RANSAC_MIN_ITERATIONS = sanitizeMinInt(
        DG_RANSAC_MIN_ITERATIONS, 1, "dg_ransac.min_iterations");
    DG_RANSAC_MAX_ITERATIONS = sanitizeMinInt(
        DG_RANSAC_MAX_ITERATIONS, DG_RANSAC_MIN_ITERATIONS, "dg_ransac.max_iterations");
    DG_RANSAC_EARLY_TERMINATION_RATIO = sanitizeRangeDouble(
        DG_RANSAC_EARLY_TERMINATION_RATIO, 0.0, 1.0, 0.8,
        "dg_ransac.early_termination_ratio");

    LOG_INTERVAL = sanitizeMinInt(LOG_INTERVAL, 1, "logging.log_interval");

    DEPTH_CONFIDENCE_SIGMA_RELATIVE = sanitizePositiveDouble(
        DEPTH_CONFIDENCE_SIGMA_RELATIVE, 0.08, "depth_confidence.sigma_relative");
    DEPTH_CONFIDENCE_SIGMA_RATE = sanitizePositiveDouble(
        DEPTH_CONFIDENCE_SIGMA_RATE, 0.5, "depth_confidence.sigma_rate");
    DEPTH_CONFIDENCE_LAMBDA_DEPTH = sanitizeNonNegativeDouble(
        DEPTH_CONFIDENCE_LAMBDA_DEPTH, 0.0, "depth_confidence.lambda_depth");
    DEPTH_CONFIDENCE_LAMBDA_FOV = sanitizeNonNegativeDouble(
        DEPTH_CONFIDENCE_LAMBDA_FOV, 0.0, "depth_confidence.lambda_fov");
    DEPTH_CONFIDENCE_MIN_THRESHOLD = sanitizeRangeDouble(
        DEPTH_CONFIDENCE_MIN_THRESHOLD, 0.0, 1.0, 0.3, "depth_confidence.min_threshold");
    DEPTH_CONFIDENCE_HIGH_THRESHOLD = sanitizeRangeDouble(
        DEPTH_CONFIDENCE_HIGH_THRESHOLD, 0.0, 1.0, 0.7, "depth_confidence.high_threshold");
    if (DEPTH_CONFIDENCE_MIN_THRESHOLD > DEPTH_CONFIDENCE_HIGH_THRESHOLD) {
        ROS_WARN("[ConfigCheck] depth_confidence.min_threshold > high_threshold, swapping values.");
        std::swap(DEPTH_CONFIDENCE_MIN_THRESHOLD, DEPTH_CONFIDENCE_HIGH_THRESHOLD);
    }
    DEPTH_FILTER_SIZE = sanitizeOddInt(
        DEPTH_FILTER_SIZE, 1, "depth_confidence.filter_size");

    if (DG_RANSAC_HIGH_CONF_FILTER_THRESHOLD < DEPTH_CONFIDENCE_MIN_THRESHOLD) {
        ROS_WARN("[ConfigCheck] dg_ransac.high_conf_filter_threshold=%.3f is below depth_confidence.min_threshold=%.3f, raising it.",
                 DG_RANSAC_HIGH_CONF_FILTER_THRESHOLD, DEPTH_CONFIDENCE_MIN_THRESHOLD);
        DG_RANSAC_HIGH_CONF_FILTER_THRESHOLD = DEPTH_CONFIDENCE_MIN_THRESHOLD;
    }

    POST_VERIFY_SUSPICIOUS_THRESHOLD = sanitizeRangeDouble(
        POST_VERIFY_SUSPICIOUS_THRESHOLD, 0.0, 1.0, 0.5,
        "post_verification.suspicious_threshold");
    POST_VERIFY_MAX_COUNT = sanitizeMinInt(
        POST_VERIFY_MAX_COUNT, 1, "post_verification.max_verify_count");

    DEPTH_MIN_RANGE = sanitizePositiveDouble(
        DEPTH_MIN_RANGE, 0.3, "depth_verification.depth_min_range");
    DEPTH_MAX_RANGE = sanitizePositiveDouble(
        DEPTH_MAX_RANGE, 8.0, "depth_verification.depth_max_range");
    if (DEPTH_MAX_RANGE <= DEPTH_MIN_RANGE) {
        ROS_WARN("[ConfigCheck] depth_verification.depth_max_range=%.3f must be greater than depth_min_range=%.3f, resetting to defaults [0.3, 8.0].",
                 DEPTH_MAX_RANGE, DEPTH_MIN_RANGE);
        DEPTH_MIN_RANGE = 0.3;
        DEPTH_MAX_RANGE = 8.0;
    }
    MIN_DEPTH_SAMPLES = sanitizeMinInt(
        MIN_DEPTH_SAMPLES, 1, "depth_verification.min_depth_samples");
    DEPTH_RATE_TOLERANCE_SIGMA = sanitizePositiveDouble(
        DEPTH_RATE_TOLERANCE_SIGMA, 3.0, "depth_verification.depth_rate_tolerance_sigma");
    DEPTH_NOISE_STD = sanitizePositiveDouble(
        DEPTH_NOISE_STD, 0.02, "depth_verification.depth_noise_std");

    DRT_PRIOR_POS_COV = sanitizePositiveDouble(
        DRT_PRIOR_POS_COV, 1.0, "drt_prior.pos_cov");
    DRT_PRIOR_ROT_COV = sanitizePositiveDouble(
        DRT_PRIOR_ROT_COV, 0.2, "drt_prior.rot_cov");
    DRT_PRIOR_VEL_COV = sanitizePositiveDouble(
        DRT_PRIOR_VEL_COV, 1.0, "drt_prior.vel_cov");
    DRT_PRIOR_ACC_BIAS_COV = sanitizePositiveDouble(
        DRT_PRIOR_ACC_BIAS_COV, 0.05, "drt_prior.acc_bias_cov");
    DRT_PRIOR_GYR_BIAS_COV = sanitizePositiveDouble(
        DRT_PRIOR_GYR_BIAS_COV, 0.01, "drt_prior.gyr_bias_cov");
    INIT_REL_POSE_MIN_PARALLAX_PX = sanitizePositiveDouble(
        INIT_REL_POSE_MIN_PARALLAX_PX,
        std::max(1.0, MIN_PARALLAX * FOCAL_LENGTH),
        "initialization.relative_pose_min_parallax_px");
    MAX_VINS_INIT_FAILURES = sanitizeMinInt(
        MAX_VINS_INIT_FAILURES, 1, "initialization.max_vins_init_failures");

    if (!ENABLE_DG_RANSAC && (ENABLE_PRE_FILTERING || ENABLE_POST_VERIFICATION)) {
        ROS_WARN("[ConfigCheck] DG-RANSAC is disabled, so pre_filtering/post_verification will not execute in trackImage().");
    }
    if (ENABLE_DG_RANSAC && !ENABLE_PRE_FILTERING && !ENABLE_POST_VERIFICATION) {
        ROS_WARN("[ConfigCheck] DG-RANSAC is enabled, but both pre_filtering and post_verification are disabled. Frontend will behave like legacy RANSAC.");
    }
    if (!TRIGGER_ENABLED && (ENABLE_DG_RANSAC || ENABLE_PRE_FILTERING || ENABLE_POST_VERIFICATION)) {
        ROS_WARN("[ConfigCheck] dynamic_trigger is disabled. Dynamic frontend modules will not enter ACTIVE state automatically.");
    }
    if (!LOG_STATISTICS) {
        ROS_WARN("[ConfigCheck] logging.log_statistics=0. E2/E3/E5 process metrics will be incomplete.");
    }
    if (!LOG_PERFORMANCE) {
        ROS_WARN("[ConfigCheck] logging.log_performance=0. E5 runtime metrics will be incomplete.");
    }
}

void logEffectiveExperimentConfiguration(const std::string& config_file)
{
    ROS_INFO("[ConfigCheck] Closed-loop verification config: %s", config_file.c_str());
    ROS_INFO("[ConfigCheck] effective_modules: trigger=%s, dg_ransac=%s, pre_filtering=%s, post_verification=%s, drt_fallback=%s",
             TRIGGER_ENABLED ? "ON" : "OFF",
             ENABLE_DG_RANSAC ? "ON" : "OFF",
             ENABLE_PRE_FILTERING ? "ON" : "OFF",
             ENABLE_POST_VERIFICATION ? "ON" : "OFF",
             ENABLE_DRT_FALLBACK ? "ON" : "OFF");
    ROS_INFO("[ConfigCheck] trigger_params: tracking_rate_threshold=%.3f, k_on=%d, k_off=%d",
             TRACKING_RATE_THRESHOLD, TRIGGER_K_ON_FRAMES, TRIGGER_K_OFF_FRAMES);
    ROS_INFO("[ConfigCheck] init_params: keyframe_parallax=%.3fpx, relative_pose_min_parallax=%.3fpx",
             MIN_PARALLAX * FOCAL_LENGTH, INIT_REL_POSE_MIN_PARALLAX_PX);
    ROS_INFO("[ConfigCheck] init_params: max_vins_init_failures=%d", MAX_VINS_INIT_FAILURES);
    ROS_INFO("[ConfigCheck] depth_confidence: sigma_rate=%.3f, sigma_relative=%.3f, lambda_depth=%.4f, lambda_fov=%.4f, min=%.3f, high=%.3f",
             DEPTH_CONFIDENCE_SIGMA_RATE, DEPTH_CONFIDENCE_SIGMA_RELATIVE,
             DEPTH_CONFIDENCE_LAMBDA_DEPTH, DEPTH_CONFIDENCE_LAMBDA_FOV,
             DEPTH_CONFIDENCE_MIN_THRESHOLD, DEPTH_CONFIDENCE_HIGH_THRESHOLD);
    ROS_INFO("[ConfigCheck] dg_ransac_params: high_conf_filter_threshold=%.3f, min_points=%d, epipolar_threshold=%.3f, depth_error_threshold=%.3f, iterations=[%d,%d], early_termination=%.3f",
             DG_RANSAC_HIGH_CONF_FILTER_THRESHOLD, DG_RANSAC_MIN_POINTS,
             DG_RANSAC_EPIPOLAR_THRESHOLD, DG_RANSAC_DEPTH_ERROR_THRESHOLD,
             DG_RANSAC_MIN_ITERATIONS, DG_RANSAC_MAX_ITERATIONS,
             DG_RANSAC_EARLY_TERMINATION_RATIO);
    ROS_INFO("[ConfigCheck] post_verification: suspicious_threshold=%.3f, max_verify_count=%d",
             POST_VERIFY_SUSPICIOUS_THRESHOLD, POST_VERIFY_MAX_COUNT);
    ROS_INFO("[ConfigCheck] depth_range: [%.3f, %.3f], min_depth_samples=%d, rate_sigma=%.3f, noise_std=%.4f",
             DEPTH_MIN_RANGE, DEPTH_MAX_RANGE, MIN_DEPTH_SAMPLES,
             DEPTH_RATE_TOLERANCE_SIGMA, DEPTH_NOISE_STD);
    ROS_INFO("[ConfigCheck] logging: interval=%d, trigger_events=%s, statistics=%s, performance=%s, depth_preprocess=%s, confidence=%s, ransac_details=%s, depth_motion=%s",
             LOG_INTERVAL,
             LOG_TRIGGER_EVENTS ? "ON" : "OFF",
             LOG_STATISTICS ? "ON" : "OFF",
             LOG_PERFORMANCE ? "ON" : "OFF",
             LOG_DEPTH_PREPROCESS ? "ON" : "OFF",
             LOG_CONFIDENCE_COMPUTE ? "ON" : "OFF",
             LOG_RANSAC_DETAILS ? "ON" : "OFF",
             LOG_DEPTH_MOTION ? "ON" : "OFF");
    ROS_INFO("[ConfigCheck] expected_outputs: init_logs=ON, first_pose_log=ON, tracking_rate_log=ON, trigger_event_logs=%s, dg_stats_logs=%s, module3_stats_logs=%s, dg_perf_logs=%s",
             LOG_TRIGGER_EVENTS ? "ON" : "OFF",
             LOG_STATISTICS ? "ON" : "OFF",
             LOG_STATISTICS ? "ON" : "OFF",
             LOG_PERFORMANCE ? "ON" : "OFF");
    ROS_INFO("[ConfigCheck] note: dynamic frontend logs appear only after ACTIVE is entered.");
}

} // namespace

void readParameters(std::string config_file)
{
    ROS_INFO("[BuildInfo] vins_drt_dyn built at %s %s", __DATE__, __TIME__);

    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];
    
    // Load flow_back_threshold with default value
    cv::FileNode fb_thresh_node = fsSettings["flow_back_threshold"];
    if (fb_thresh_node.empty()) {
        FLOW_BACK_THRESHOLD = 1.5;  // Default to 1.5 pixels
        ROS_WARN("flow_back_threshold not found, using default: 1.5");
    } else {
        fb_thresh_node >> FLOW_BACK_THRESHOLD;
    }
    ROS_INFO_STREAM("Loaded flow_back_threshold: " << FLOW_BACK_THRESHOLD);

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
        // Optional IMU acc unit scale; default 1.0 (SI). VCU-RVI sets 9.80665
        // to convert g-units to m/s^2. OpenLORIS leaves this field absent.
        if (!fsSettings["imu_acc_scale"].empty())
            IMU_ACC_SCALE = (double)fsSettings["imu_acc_scale"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    double keyframe_parallax_px = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = keyframe_parallax_px / FOCAL_LENGTH;
    INIT_REL_POSE_MIN_PARALLAX_PX = keyframe_parallax_px;
    cv::FileNode initialization_node = fsSettings["initialization"];
    if (!initialization_node.empty() && initialization_node.isMap())
    {
        cv::FileNode rel_pose_parallax_node = initialization_node["relative_pose_min_parallax_px"];
        if (!rel_pose_parallax_node.empty())
        {
            rel_pose_parallax_node >> INIT_REL_POSE_MIN_PARALLAX_PX;
        }

        cv::FileNode max_init_failures_node = initialization_node["max_vins_init_failures"];
        if (!max_init_failures_node.empty())
        {
            max_init_failures_node >> MAX_VINS_INIT_FAILURES;
        }
    }

    // Load DRT Prior Covariance Parameters
    cv::FileNode drt_prior_node = fsSettings["drt_prior"];
    if (!drt_prior_node.empty()) {
        if (!drt_prior_node["pos_cov"].empty()) drt_prior_node["pos_cov"] >> DRT_PRIOR_POS_COV;
        if (!drt_prior_node["rot_cov"].empty()) drt_prior_node["rot_cov"] >> DRT_PRIOR_ROT_COV;
        if (!drt_prior_node["vel_cov"].empty()) drt_prior_node["vel_cov"] >> DRT_PRIOR_VEL_COV;
        if (!drt_prior_node["acc_bias_cov"].empty()) drt_prior_node["acc_bias_cov"] >> DRT_PRIOR_ACC_BIAS_COV;
        if (!drt_prior_node["gyr_bias_cov"].empty()) drt_prior_node["gyr_bias_cov"] >> DRT_PRIOR_GYR_BIAS_COV;
    } else {
        ROS_WARN("[ConfigCheck] drt_prior block not found; default DRT prior covariance will be used only if DRT fallback is enabled.");
    }

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    if (!fout.is_open()) {
        ROS_ERROR("[ConfigCheck] Failed to open trajectory output file: %s", VINS_RESULT_PATH.c_str());
    } else {
        ROS_INFO("[ConfigCheck] Trajectory output path is writable: %s", VINS_RESULT_PATH.c_str());
    }
    fout.close();

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }


    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);

    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    if(NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib;
        //printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);

        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    // Load dynamic detection parameters
    fsSettings["enable_dynamic_detection"] >> ENABLE_DYNAMIC_DETECTION;
    ROS_INFO_STREAM("Loaded enable_dynamic_detection: " << ENABLE_DYNAMIC_DETECTION);

    // ACE修复：恢复前端动态检测仍需要的参数加载
    fsSettings["enable_dss_temporal_anomaly"] >> ENABLE_DSS_TEMPORAL_ANOMALY;
    ROS_INFO_STREAM("Loaded enable_dss_temporal_anomaly: " << ENABLE_DSS_TEMPORAL_ANOMALY);
    fsSettings["dss_jump_threshold"] >> DSS_JUMP_THRESHOLD;
    ROS_INFO_STREAM("Loaded dss_jump_threshold: " << DSS_JUMP_THRESHOLD);
    fsSettings["anomaly_boost_factor"] >> ANOMALY_BOOST_FACTOR;
    ROS_INFO_STREAM("Loaded anomaly_boost_factor: " << ANOMALY_BOOST_FACTOR);
    fsSettings["dynamic_min_trigger_frames"] >> DYNAMIC_MIN_TRIGGER_FRAMES;
    ROS_INFO_STREAM("Loaded dynamic_min_trigger_frames: " << DYNAMIC_MIN_TRIGGER_FRAMES);
    fsSettings["dynamic_min_reset_frames_internal"] >> DYNAMIC_MIN_RESET_FRAMES_INTERNAL;
    ROS_INFO_STREAM("Loaded dynamic_min_reset_frames_internal: " << DYNAMIC_MIN_RESET_FRAMES_INTERNAL);
    fsSettings["dynamic_dss_hysteresis_ratio"] >> DYNAMIC_DSS_HYSTERESIS_RATIO;
    ROS_INFO_STREAM("Loaded dynamic_dss_hysteresis_ratio: " << DYNAMIC_DSS_HYSTERESIS_RATIO);
    fsSettings["dynamic_temporal_consistency_threshold"] >> DYNAMIC_TEMPORAL_CONSISTENCY_THRESHOLD;
    ROS_INFO_STREAM("Loaded dynamic_temporal_consistency_threshold: " << DYNAMIC_TEMPORAL_CONSISTENCY_THRESHOLD);



    // 读取保留的必要参数
    cv::FileNode default_depth_node = fsSettings["dynamic_default_depth"];
    if (default_depth_node.empty()) {
        DYNAMIC_DEFAULT_DEPTH = 5.0;  // 默认值5.0米
        ROS_WARN("dynamic_default_depth not found, using default: 5.0");
    } else {
        default_depth_node >> DYNAMIC_DEFAULT_DEPTH;
    }
    ROS_INFO_STREAM("Loaded dynamic_default_depth: " << DYNAMIC_DEFAULT_DEPTH);

    fsSettings["dss_intelligent_scoring_debug"] >> DSS_INTELLIGENT_SCORING_DEBUG;
    ROS_INFO_STREAM("Loaded dss_intelligent_scoring_debug: " << DSS_INTELLIGENT_SCORING_DEBUG);

    // Event-Triggered Dynamic Filtering - Phase 1: Simplified Trigger Mechanism
    cv::FileNode trigger_node = fsSettings["dynamic_trigger"];
    if (!trigger_node.empty()) {
        int enabled = 1;
        trigger_node["enabled"] >> enabled;
        TRIGGER_ENABLED = (enabled != 0);
        trigger_node["tracking_rate_threshold"] >> TRACKING_RATE_THRESHOLD;
        trigger_node["k_on_frames"] >> TRIGGER_K_ON_FRAMES;
        trigger_node["k_off_frames"] >> TRIGGER_K_OFF_FRAMES;
        
        ROS_INFO("[DynamicTrigger] Enabled: %s", TRIGGER_ENABLED ? "YES" : "NO");
        if (TRIGGER_ENABLED) {
            ROS_INFO("[DynamicTrigger] tracking_rate_threshold=%.2f, k_on=%d, k_off=%d",
                     TRACKING_RATE_THRESHOLD, TRIGGER_K_ON_FRAMES, TRIGGER_K_OFF_FRAMES);
        }
    } else {
        ROS_WARN("[DynamicTrigger] 'dynamic_trigger' section not found, using defaults");
        ROS_INFO("[DynamicTrigger] Defaults: enabled=%s, threshold=%.2f, k_on=%d, k_off=%d",
                 TRIGGER_ENABLED ? "YES" : "NO", TRACKING_RATE_THRESHOLD, TRIGGER_K_ON_FRAMES, TRIGGER_K_OFF_FRAMES);
    }
    
    // DG-RANSAC Parameters - Module 2
    cv::FileNode dg_ransac_node = fsSettings["dg_ransac"];
    if (!dg_ransac_node.empty()) {
        cv::FileNode dg_enabled_node = dg_ransac_node["enabled"];
        if (!dg_enabled_node.empty()) {
            int enabled = 1;
            dg_enabled_node >> enabled;
            ENABLE_DG_RANSAC = (enabled != 0);
        }

        // 基础参数
        dg_ransac_node["local_window_size"] >> DG_RANSAC_LOCAL_WINDOW_SIZE;
        dg_ransac_node["filter_kernel"] >> DG_RANSAC_FILTER_KERNEL;
        dg_ransac_node["high_conf_filter_threshold"] >> DG_RANSAC_HIGH_CONF_FILTER_THRESHOLD;
        dg_ransac_node["min_points_for_ransac"] >> DG_RANSAC_MIN_POINTS;
        
        // 读取fallback_enabled
        cv::FileNode fallback_node = dg_ransac_node["fallback_enabled"];
        if (!fallback_node.empty()) {
            int fallback_val = 1;
            fallback_node >> fallback_val;
            DG_RANSAC_FALLBACK_ENABLED = (fallback_val != 0);
        }
        
        // RANSAC执行参数
        dg_ransac_node["epipolar_threshold"] >> DG_RANSAC_EPIPOLAR_THRESHOLD;
        dg_ransac_node["depth_error_threshold"] >> DG_RANSAC_DEPTH_ERROR_THRESHOLD;
        dg_ransac_node["min_iterations"] >> DG_RANSAC_MIN_ITERATIONS;
        dg_ransac_node["max_iterations"] >> DG_RANSAC_MAX_ITERATIONS;
        
        cv::FileNode adaptive_node = dg_ransac_node["adaptive_iterations"];
        if (!adaptive_node.empty()) {
            int adaptive_val = 1;
            adaptive_node >> adaptive_val;
            DG_RANSAC_ADAPTIVE_ITERATIONS = (adaptive_val != 0);
        }
        
        dg_ransac_node["early_termination_ratio"] >> DG_RANSAC_EARLY_TERMINATION_RATIO;
        
        ROS_INFO("[DG-RANSAC] Loaded parameters:");
        ROS_INFO("[DG-RANSAC]   enabled=%s", ENABLE_DG_RANSAC ? "YES" : "NO");
        ROS_INFO("[DG-RANSAC]   Preprocessing: window=%dx%d, kernel=%dx%d",
                 DG_RANSAC_LOCAL_WINDOW_SIZE, DG_RANSAC_LOCAL_WINDOW_SIZE,
                 DG_RANSAC_FILTER_KERNEL, DG_RANSAC_FILTER_KERNEL);
        ROS_INFO("[DG-RANSAC]   Pre-filtering: high_conf_threshold=%.2f, min_points=%d, fallback=%s",
                 DG_RANSAC_HIGH_CONF_FILTER_THRESHOLD, DG_RANSAC_MIN_POINTS,
                 DG_RANSAC_FALLBACK_ENABLED ? "YES" : "NO");
        ROS_INFO("[DG-RANSAC]   RANSAC: epipolar_thresh=%.1f, depth_error_thresh=%.2f",
                 DG_RANSAC_EPIPOLAR_THRESHOLD, DG_RANSAC_DEPTH_ERROR_THRESHOLD);
        ROS_INFO("[DG-RANSAC]   Iterations: min=%d, max=%d, adaptive=%s, early_term=%.2f",
                 DG_RANSAC_MIN_ITERATIONS, DG_RANSAC_MAX_ITERATIONS,
                 DG_RANSAC_ADAPTIVE_ITERATIONS ? "YES" : "NO",
                 DG_RANSAC_EARLY_TERMINATION_RATIO);
    } else {
        ROS_WARN("[DG-RANSAC] 'dg_ransac' section not found, using defaults");
    }
    
    // Logging Configuration
    cv::FileNode logging_node = fsSettings["logging"];
    if (!logging_node.empty()) {
        cv::FileNode detailed_log_node = logging_node["enable_detailed_log"];
        if (!detailed_log_node.empty()) {
            int detailed_val = 0;
            detailed_log_node >> detailed_val;
            ENABLE_DETAILED_LOG = (detailed_val != 0);
        }
        
        logging_node["log_interval"] >> LOG_INTERVAL;
        
        cv::FileNode trigger_log_node = logging_node["log_trigger_events"];
        if (!trigger_log_node.empty()) {
            int trigger_val = 1;
            trigger_log_node >> trigger_val;
            LOG_TRIGGER_EVENTS = (trigger_val != 0);
        }
        
        cv::FileNode stats_log_node = logging_node["log_statistics"];
        if (!stats_log_node.empty()) {
            int stats_val = 1;
            stats_log_node >> stats_val;
            LOG_STATISTICS = (stats_val != 0);
        }
        
        cv::FileNode perf_log_node = logging_node["log_performance"];
        if (!perf_log_node.empty()) {
            int perf_val = 1;
            perf_log_node >> perf_val;
            LOG_PERFORMANCE = (perf_val != 0);
        }
        
        cv::FileNode depth_prep_log_node = logging_node["log_depth_preprocess"];
        if (!depth_prep_log_node.empty()) {
            int depth_prep_val = 0;
            depth_prep_log_node >> depth_prep_val;
            LOG_DEPTH_PREPROCESS = (depth_prep_val != 0);
        }
        
        cv::FileNode conf_log_node = logging_node["log_confidence_compute"];
        if (!conf_log_node.empty()) {
            int conf_val = 0;
            conf_log_node >> conf_val;
            LOG_CONFIDENCE_COMPUTE = (conf_val != 0);
        }
        
        cv::FileNode ransac_log_node = logging_node["log_ransac_details"];
        if (!ransac_log_node.empty()) {
            int ransac_val = 0;
            ransac_log_node >> ransac_val;
            LOG_RANSAC_DETAILS = (ransac_val != 0);
        }
        
        cv::FileNode motion_log_node = logging_node["log_depth_motion"];
        if (!motion_log_node.empty()) {
            int motion_val = 0;
            motion_log_node >> motion_val;
            LOG_DEPTH_MOTION = (motion_val != 0);
        }
        
        ROS_INFO("[Logging] Configuration loaded:");
        ROS_INFO("[Logging]   detailed_log=%s, interval=%d",
                 ENABLE_DETAILED_LOG ? "ON" : "OFF", LOG_INTERVAL);
        ROS_INFO("[Logging]   trigger=%s, stats=%s, perf=%s",
                 LOG_TRIGGER_EVENTS ? "ON" : "OFF",
                 LOG_STATISTICS ? "ON" : "OFF",
                 LOG_PERFORMANCE ? "ON" : "OFF");
        ROS_INFO("[Logging]   depth_prep=%s, confidence=%s, ransac=%s, motion=%s",
                 LOG_DEPTH_PREPROCESS ? "ON" : "OFF",
                 LOG_CONFIDENCE_COMPUTE ? "ON" : "OFF",
                 LOG_RANSAC_DETAILS ? "ON" : "OFF",
                 LOG_DEPTH_MOTION ? "ON" : "OFF");
    } else {
        ROS_WARN("[Logging] 'logging' section not found, using defaults");
        ROS_INFO("[Logging] Defaults: trigger=%s, stats=%s, perf=%s",
                 LOG_TRIGGER_EVENTS ? "ON" : "OFF",
                 LOG_STATISTICS ? "ON" : "OFF",
                 LOG_PERFORMANCE ? "ON" : "OFF");
    }
    
    // Depth Confidence Parameters
    cv::FileNode depth_conf_node = fsSettings["depth_confidence"];
    if (!depth_conf_node.empty()) {
        depth_conf_node["sigma_relative"] >> DEPTH_CONFIDENCE_SIGMA_RELATIVE;
        depth_conf_node["sigma_rate"] >> DEPTH_CONFIDENCE_SIGMA_RATE;
        cv::FileNode lambda_depth_node = depth_conf_node["lambda_depth"];
        if (!lambda_depth_node.empty()) {
            lambda_depth_node >> DEPTH_CONFIDENCE_LAMBDA_DEPTH;
        }
        cv::FileNode lambda_fov_node = depth_conf_node["lambda_fov"];
        if (!lambda_fov_node.empty()) {
            lambda_fov_node >> DEPTH_CONFIDENCE_LAMBDA_FOV;
        }
        depth_conf_node["min_threshold"] >> DEPTH_CONFIDENCE_MIN_THRESHOLD;
        depth_conf_node["high_threshold"] >> DEPTH_CONFIDENCE_HIGH_THRESHOLD;
        depth_conf_node["filter_size"] >> DEPTH_FILTER_SIZE;
        
        ROS_INFO("[DepthConfidence] Loaded parameters:");
        ROS_INFO("[DepthConfidence]   sigma_rate=%.2f m/s, sigma_relative=%.3f, lambda_depth=%.4f, lambda_fov=%.4f",
                 DEPTH_CONFIDENCE_SIGMA_RATE, DEPTH_CONFIDENCE_SIGMA_RELATIVE,
                 DEPTH_CONFIDENCE_LAMBDA_DEPTH, DEPTH_CONFIDENCE_LAMBDA_FOV);
        ROS_INFO("[DepthConfidence]   thresholds: min=%.2f, high=%.2f, filter_size=%d",
                 DEPTH_CONFIDENCE_MIN_THRESHOLD, DEPTH_CONFIDENCE_HIGH_THRESHOLD, DEPTH_FILTER_SIZE);
    } else {
        ROS_WARN("[DepthConfidence] 'depth_confidence' section not found, using defaults");
        ROS_INFO("[DepthConfidence] Defaults: sigma_rate=%.2f, sigma_relative=%.3f, lambda_depth=%.4f, lambda_fov=%.4f, thresholds=[%.2f, %.2f]",
                 DEPTH_CONFIDENCE_SIGMA_RATE, DEPTH_CONFIDENCE_SIGMA_RELATIVE,
                 DEPTH_CONFIDENCE_LAMBDA_DEPTH, DEPTH_CONFIDENCE_LAMBDA_FOV,
                 DEPTH_CONFIDENCE_MIN_THRESHOLD, DEPTH_CONFIDENCE_HIGH_THRESHOLD);
    }
    
    // Module 3: Post-Verification Parameters
    cv::FileNode post_verify_node = fsSettings["post_verification"];
    if (!post_verify_node.empty()) {
        int enabled;
        post_verify_node["enabled"] >> enabled;
        ENABLE_POST_VERIFICATION = (enabled != 0);
        post_verify_node["suspicious_threshold"] >> POST_VERIFY_SUSPICIOUS_THRESHOLD;
        post_verify_node["max_verify_count"] >> POST_VERIFY_MAX_COUNT;
        
        ROS_INFO("[PostVerification] Loaded parameters:");
        ROS_INFO("[PostVerification]   enabled=%s, suspicious_threshold=%.2f, max_count=%d",
                 ENABLE_POST_VERIFICATION ? "YES" : "NO",
                 POST_VERIFY_SUSPICIOUS_THRESHOLD, POST_VERIFY_MAX_COUNT);
    } else {
        ROS_INFO("[PostVerification] 'post_verification' section not found, using defaults (disabled)");
    }
    
    // Phase 2: Suspicious Point Expansion Parameter Loading
    cv::FileNode phase2_node = fsSettings["suspicious_point_expansion"];
    if (!phase2_node.empty()) {
        phase2_node["base_expand_radius"] >> BASE_EXPAND_RADIUS;
        phase2_node["dense_expand_radius"] >> DENSE_EXPAND_RADIUS;
        phase2_node["outlier_density_threshold"] >> OUTLIER_DENSITY_THRESHOLD;
        phase2_node["max_suspicious_ratio"] >> MAX_SUSPICIOUS_RATIO;
        
        // 参数范围检查
        if (BASE_EXPAND_RADIUS <= 0 || DENSE_EXPAND_RADIUS <= 0 || 
            DENSE_EXPAND_RADIUS < BASE_EXPAND_RADIUS ||
            OUTLIER_DENSITY_THRESHOLD < 1 || MAX_SUSPICIOUS_RATIO <= 0 || MAX_SUSPICIOUS_RATIO > 1) {
            ROS_ERROR("Phase 2 parameters out of valid range, using defaults");
            BASE_EXPAND_RADIUS = 50.0;
            DENSE_EXPAND_RADIUS = 100.0;
            OUTLIER_DENSITY_THRESHOLD = 3;
            MAX_SUSPICIOUS_RATIO = 0.5;
        }
        
        ROS_INFO_STREAM("Phase 2 Expansion Loaded: base_radius=" << BASE_EXPAND_RADIUS 
                        << " dense_radius=" << DENSE_EXPAND_RADIUS
                        << " density_thresh=" << OUTLIER_DENSITY_THRESHOLD);
    } else {
        ROS_WARN("suspicious_point_expansion section not found, using defaults");
    }

    // Phase 3: 读取深度图参数
    cv::FileNode depth_topic_node = fsSettings["depth0_topic"];
    if(!depth_topic_node.empty()) {
        depth_topic_node >> DEPTH_TOPIC;
        ROS_INFO("Depth topic: %s", DEPTH_TOPIC.c_str());
    } else {
        DEPTH_TOPIC = "";
    }

    // 深度范围参数在depth_verification节点下
    cv::FileNode depth_verification_node = fsSettings["depth_verification"];
    if(!depth_verification_node.empty()) {
        cv::FileNode depth_min_node = depth_verification_node["depth_min_range"];
        if(!depth_min_node.empty()) {
            depth_min_node >> DEPTH_MIN_RANGE;
            ROS_INFO("Read depth_min_range from config: %.1f", DEPTH_MIN_RANGE);
        }
        
        cv::FileNode depth_max_node = depth_verification_node["depth_max_range"];
        if(!depth_max_node.empty()) {
            depth_max_node >> DEPTH_MAX_RANGE;
            ROS_INFO("Read depth_max_range from config: %.1f", DEPTH_MAX_RANGE);
        }
    } else {
        ROS_WARN("depth_verification node not found in config, using defaults");
        DEPTH_MIN_RANGE = 0.3;
        DEPTH_MAX_RANGE = 5.0;
    }

    if(!DEPTH_TOPIC.empty()) {
        ROS_INFO_STREAM("Depth range: [" << DEPTH_MIN_RANGE << "m, " << DEPTH_MAX_RANGE << "m]");
    }

    // Phase 3: 深度变化率验证参数
    cv::FileNode depth_verif_node = fsSettings["depth_verification"];
    if (!depth_verif_node.empty() && depth_verif_node.isMap()) {
        MIN_DEPTH_SAMPLES = depth_verif_node["min_depth_samples"];
        DEPTH_RATE_TOLERANCE_SIGMA = depth_verif_node["depth_rate_tolerance_sigma"];
        DEPTH_NOISE_STD = depth_verif_node["depth_noise_std"];

        ROS_INFO("[ParamCheck] depth_rate_tolerance_sigma loaded: %.2f", DEPTH_RATE_TOLERANCE_SIGMA);

        ROS_INFO("[Phase3Config] min_samples=%d, sigma=%.1f, noise_std=%.3f",
                 MIN_DEPTH_SAMPLES, DEPTH_RATE_TOLERANCE_SIGMA, DEPTH_NOISE_STD);
    } else {
        ROS_WARN("[Phase3Config] depth_verification not found in config, using defaults");
    }

    // Module Control Switches
    cv::FileNode module_control_node = fsSettings["module_control"];
    if (!module_control_node.empty() && module_control_node.isMap()) {
        const bool post_from_post_verification = ENABLE_POST_VERIFICATION;
        const bool dg_from_dg_ransac = ENABLE_DG_RANSAC;

        int enable_pre = ENABLE_PRE_FILTERING ? 1 : 0;
        int enable_post = ENABLE_POST_VERIFICATION ? 1 : 0;
        int enable_dg = ENABLE_DG_RANSAC ? 1 : 0;
        int enable_drt = ENABLE_DRT_FALLBACK ? 1 : 0;

        cv::FileNode pre_node = module_control_node["enable_pre_filtering"];
        if (!pre_node.empty()) {
            pre_node >> enable_pre;
        }

        cv::FileNode post_node = module_control_node["enable_post_verification"];
        if (!post_node.empty()) {
            post_node >> enable_post;
        }

        cv::FileNode dg_node = module_control_node["enable_dg_ransac"];
        if (!dg_node.empty()) {
            dg_node >> enable_dg;
        }

        cv::FileNode drt_node = module_control_node["enable_drt_fallback"];
        if (!drt_node.empty()) {
            drt_node >> enable_drt;
        }

        warnBoolConflict("dg_ransac.enabled", dg_from_dg_ransac,
                         "module_control.enable_dg_ransac", enable_dg != 0,
                         "module_control.enable_dg_ransac");
        warnBoolConflict("post_verification.enabled", post_from_post_verification,
                         "module_control.enable_post_verification", enable_post != 0,
                         "module_control.enable_post_verification");

        ENABLE_PRE_FILTERING = (enable_pre != 0);
        ENABLE_POST_VERIFICATION = (enable_post != 0);
        ENABLE_DG_RANSAC = (enable_dg != 0);
        ENABLE_DRT_FALLBACK = (enable_drt != 0);
        
        ROS_INFO("[ModuleControl] DG-RANSAC: %s, Pre-filtering: %s, Post-verification: %s, DRT-fallback: %s",
                 ENABLE_DG_RANSAC ? "ENABLED" : "DISABLED",
                 ENABLE_PRE_FILTERING ? "ENABLED" : "DISABLED",
                 ENABLE_POST_VERIFICATION ? "ENABLED" : "DISABLED",
                 ENABLE_DRT_FALLBACK ? "ENABLED" : "DISABLED");
        ROS_INFO("[ConfigCheck] drt_prior_params: active=%s, pos=%.3f, rot=%.3f, vel=%.3f, acc_bias=%.3f, gyr_bias=%.3f",
                 ENABLE_DRT_FALLBACK ? "ON" : "OFF",
                 DRT_PRIOR_POS_COV, DRT_PRIOR_ROT_COV, DRT_PRIOR_VEL_COV,
                 DRT_PRIOR_ACC_BIAS_COV, DRT_PRIOR_GYR_BIAS_COV);
    } else {
        ROS_WARN("[ModuleControl] module_control not found, using current switches (dg=%s, pre=%s, post=%s, drt=%s)",
                 ENABLE_DG_RANSAC ? "ON" : "OFF",
                 ENABLE_PRE_FILTERING ? "ON" : "OFF",
                 ENABLE_POST_VERIFICATION ? "ON" : "OFF",
                 ENABLE_DRT_FALLBACK ? "ON" : "OFF");
    }

    // End of Event-Triggered Dynamic Filtering Parameter Loading




    // ACE 硬编码数组参数暴露：敏感性分析和调试参数
    cv::FileNode trigger_thresholds_node = fsSettings["dynamic_trigger_sensitivity_thresholds"];
    DYNAMIC_TRIGGER_SENSITIVITY_THRESHOLDS.clear();
    for (auto it = trigger_thresholds_node.begin(); it != trigger_thresholds_node.end(); ++it) {
        DYNAMIC_TRIGGER_SENSITIVITY_THRESHOLDS.push_back((double)*it);
    }
    ROS_INFO_STREAM("Loaded dynamic_trigger_sensitivity_thresholds: " << DYNAMIC_TRIGGER_SENSITIVITY_THRESHOLDS.size() << " values");

    cv::FileNode reset_thresholds_node = fsSettings["dynamic_reset_sensitivity_thresholds"];
    DYNAMIC_RESET_SENSITIVITY_THRESHOLDS.clear();
    for (auto it = reset_thresholds_node.begin(); it != reset_thresholds_node.end(); ++it) {
        DYNAMIC_RESET_SENSITIVITY_THRESHOLDS.push_back((double)*it);
    }
    ROS_INFO_STREAM("Loaded dynamic_reset_sensitivity_thresholds: " << DYNAMIC_RESET_SENSITIVITY_THRESHOLDS.size() << " values");

    cv::FileNode flow_thresholds_node = fsSettings["dynamic_flow_sensitivity_thresholds"];
    DYNAMIC_FLOW_SENSITIVITY_THRESHOLDS.clear();
    for (auto it = flow_thresholds_node.begin(); it != flow_thresholds_node.end(); ++it) {
        DYNAMIC_FLOW_SENSITIVITY_THRESHOLDS.push_back((double)*it);
    }
    ROS_INFO_STREAM("Loaded dynamic_flow_sensitivity_thresholds: " << DYNAMIC_FLOW_SENSITIVITY_THRESHOLDS.size() << " values");

    // fsSettings["dynamic_debug_feature_count"] >> DYNAMIC_DEBUG_FEATURE_COUNT; // 🗑️ 未使用
    // ROS_INFO_STREAM("Loaded dynamic_debug_feature_count: " << DYNAMIC_DEBUG_FEATURE_COUNT); // 🗑️ 未使用
    fsSettings["dynamic_cleanup_interval"] >> DYNAMIC_CLEANUP_INTERVAL;
    ROS_INFO_STREAM("Loaded dynamic_cleanup_interval: " << DYNAMIC_CLEANUP_INTERVAL);
    fsSettings["dynamic_weight_debug_count"] >> DYNAMIC_WEIGHT_DEBUG_COUNT;
    ROS_INFO_STREAM("Loaded dynamic_weight_debug_count: " << DYNAMIC_WEIGHT_DEBUG_COUNT);

    // 添加参数验证输出 (sigmoid参数已移除)
    ROS_INFO("=== Dynamic Detection Parameters Summary ===");
    ROS_INFO_STREAM("DSS Temporal Anomaly: enabled=" << ENABLE_DSS_TEMPORAL_ANOMALY
                   << ", jump_threshold=" << DSS_JUMP_THRESHOLD
                   << ", boost_factor=" << ANOMALY_BOOST_FACTOR);
    ROS_INFO_STREAM("Threshold parameters: flow_sq=" << DYNAMIC_T_FLOW_DIFF_SQ
                   << ", epipolar_sq=" << DYNAMIC_T_EPIPOLAR_RES_SQ
                   << " (multiframe已移除)");

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    validateExperimentParameters();
    logEffectiveExperimentConfiguration(config_file);

    fsSettings.release();
}
