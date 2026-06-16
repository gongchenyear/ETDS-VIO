/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <deque>
#include <algorithm>
#include <numeric>
#include <execinfo.h>
#include <csignal>
#include <functional>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);
void reduceVector(vector<float> &v, vector<uchar> status);

class FeatureTracker
{
public:
    FeatureTracker();
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat(), const cv::Mat &_depth_img = cv::Mat());
    void setMask();
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    
    // Event trigger based on optical flow tracking rate
    bool checkDynamicSceneTrigger(int original_prev_count, int successful_tracked_count);  // 返回是否处于ACTIVE状态
    void resetDynamicState(); // 重置动态检测状态（用于初始化后强制回退到STATIC）
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts,
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2,
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts,
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);



    int row, col;
    cv::Mat imTrack;
    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    double cur_time;
    double prev_time;
    bool stereo_cam;
    int n_id;
    bool hasPrediction;

    // =================================================================
    // START: Event-Triggered Dynamic Filtering (Simplified)
    // =================================================================
    
    // 1. Configuration parameters
    bool b_trigger_enabled;                  // 触发机制总开关
    double d_tracking_rate_threshold;        // 光流追踪成功率阈值
    int i_k_on_frames;                       // 触发平滑帧数（快速进入）
    int i_k_off_frames;                      // 取消触发平滑帧数（保守退出）
    
    // Module control switches
    bool b_enable_pre_filtering;             // 前置流程：RANSAC前的深度引导（Module 2）

    // 2. State machine
    enum DynamicState { STATIC, ACTIVE };
    DynamicState m_dynamic_state;            // 当前状态
    int m_trigger_on_count;                  // 连续满足触发条件的帧数
    int m_trigger_off_count;                 // 连续不满足触发条件的帧数
    bool is_first_triggered_frame_;          // 是否是触发后的第一帧
    
    // 3. Logging and statistics
    double m_last_state_change_time;         // 上次状态切换时间
    double m_active_window_start_time;       // ACTIVE窗口起始时间
    int m_frame_count;                       // 总帧数计数器
    int m_active_frame_count;                // ACTIVE状态帧数计数器
    int m_total_trigger_count;               // 总触发次数
    
    // 追踪率统计
    double m_tracking_rate_sum;              // 追踪率累计和
    double m_tracking_rate_min;              // 最小追踪率
    double m_tracking_rate_max;              // 最大追踪率
    int m_tracking_rate_below_threshold_count; // 低于触发阈值的帧数

    // DG-RANSAC / Module 3 experiment-facing statistics
    int m_dg_attempt_count;                  // DG-RANSAC尝试次数
    int m_dg_success_count;                  // DG-RANSAC成功执行次数
    int m_pre_filter_fallback_count;         // 预筛选不足导致回退次数
    int m_ransac_failure_fallback_count;     // RANSAC失败导致回退次数
    double m_pre_filter_ratio_sum;           // 预筛选保留率累计
    double m_ransac_inlier_ratio_sum;        // 最终内点率累计
    double m_dg_ransac_time_sum;             // DG-RANSAC耗时累计
    int m_module3_call_count;                // Module 3调用次数
    int m_module3_removed_total;             // Module 3累计剔除点数
    double m_module3_time_sum;               // Module 3耗时累计
    
    // Frame processing time statistics (E4 experiment)
    double m_frame_time_sum;                 // 总处理时间累计 (ms)
    double m_frame_time_normal_sum;          // NORMAL 状态处理时间累计 (ms)
    double m_frame_time_enhanced_sum;        // ENHANCED 状态处理时间累计 (ms)
    int m_normal_frame_count;                // NORMAL 状态帧数
    int m_enhanced_frame_count;              // ENHANCED 状态帧数
    
    // =================================================================
    // END: Event-Triggered Dynamic Filtering
    // =================================================================
    






    // =================================================================
    // START: DG-RANSAC Implementation - Depth-Guided RANSAC
    // =================================================================
    
    // Depth preprocessing and maintenance
    cv::Mat m_prev_depth_img;                // 前一帧深度图（原始）
    cv::Mat m_cur_depth_img;                 // 当前帧深度图（原始）
    
    // Depth pair structure for preprocessing result
    struct DepthPair {
        float depth_prev;  // t-1帧深度（已降噪）
        float depth_cur;   // t帧深度（已降噪）
        float depth_rate;  // 深度变化率 (m/s) - 缓存供Module 3复用
        bool valid;        // 是否有效
        
        DepthPair() : depth_prev(0.0f), depth_cur(0.0f), depth_rate(0.0f), valid(false) {}
        DepthPair(float d_prev, float d_cur, bool v) 
            : depth_prev(d_prev), depth_cur(d_cur), depth_rate(0.0f), valid(v) {}
    };
    
    // Depth cache for all features (computed once, used by Module 2 pre-filtering)
    map<int, DepthPair> m_depth_cache;       // feature_id -> DepthPair
    
    // Depth confidence for each feature
    map<int, double> m_depth_confidence;     // feature_id -> confidence [0,1]
    
    struct MotionConsistencyStats {
        double observed_rate;
        double predicted_rate;
        double residual;
        double sigma_rate;
        double rot_compensation;
        double trans_compensation;
        bool valid;

        MotionConsistencyStats()
            : observed_rate(0.0), predicted_rate(0.0), residual(0.0), sigma_rate(0.0),
              rot_compensation(0.0), trans_compensation(0.0), valid(false) {}
    };
    map<int, MotionConsistencyStats> m_motion_consistency_cache; // feature_id -> motion stats
    
    // Camera motion for depth confidence computation
    Vector3d m_latest_camera_velocity;            // 相机速度（世界坐标系）
    Matrix3d m_latest_camera_rotation;            // 相机旋转（世界到相机）
    bool m_has_valid_velocity;                    // 是否有有效速度
    bool m_has_valid_rotation;                    // 是否有有效旋转
    
    // 回调函数指针：用于获取相机运动（避免循环依赖）
    typedef std::function<bool(Vector3d&, Matrix3d&)> CameraMotionCallback;
    CameraMotionCallback m_camera_motion_callback;
    
    // Configuration parameters
    double d_depth_noise_sigma;              // 深度噪声标准差 (default: 0.15)
    double d_depth_conf_min_threshold;       // 最低置信度阈值 (default: 0.3)
    double d_depth_conf_high_threshold;      // 高置信度阈值 (default: 0.7)
    int i_depth_local_window_size;           // 局部滤波窗口大小 (default: 11)
    int i_depth_filter_kernel;               // 中值滤波核大小 (default: 5)
    double d_depth_confidence_sigma_relative; // 深度置信度：相对变化标准差 (default: 0.05)
    double d_depth_confidence_sigma_rate;    // 深度置信度：深度变化率标准差 (default: 0.5 m/s)
    double d_depth_confidence_lambda_depth;  // 深度自适应尺度系数
    double d_depth_confidence_lambda_fov;    // 视场位置自适应尺度系数
    
    // Depth validation ranges (loaded from config file)
    float DEPTH_MIN_RANGE;   // Minimum valid depth (m) - loaded from config
    float DEPTH_MAX_RANGE;   // Maximum valid depth (m) - loaded from config
    static constexpr double MIN_TIME_INTERVAL = 0.001; // Minimum valid time interval (s)
    static constexpr double MAX_TIME_INTERVAL = 1.0;   // Maximum valid time interval (s)
    
    // Core functions
    cv::Mat getSyncedDepthImage(double rgb_timestamp); // 同步获取深度图
    void preprocessAllDepths();              // 局部深度预处理（局部滤波，缓存结果）
    DepthPair getCachedDepth(int feature_id); // 获取缓存的深度对
    void fetchCameraMotion();                // 获取相机运动（从后端优化结果）
    void setCameraMotion(const Vector3d& velocity, const Matrix3d& rotation); // 设置相机运动（从后端传入）
    void setCameraMotionCallback(const CameraMotionCallback& callback); // 设置相机运动回调函数
    void computeDepthConfidence();           // 计算深度置信度（用于Module 2，带运动补偿）
    double computeAdaptiveRateSigma(size_t feature_index, double depth_value) const; // 基于深度和视场位置的自适应sigma
    bool buildMotionConsistencyStats(size_t feature_index, const DepthPair& depth, double dt,
                                     const Vector3d& V_camera, bool has_valid_omega,
                                     const Vector3d& omega_camera, MotionConsistencyStats& stats) const;
    vector<int> filterHighConfidencePoints(double threshold); // 筛选高置信度点
    void rejectWithF_Filtered();            // 在高置信度点上执行RANSAC（预筛选）
    void cleanupDepthCache();               // 清理深度缓存（防止内存泄漏）
    
    // =================================================================
    // Module 3: Post-Verification (Depth Motion Consistency)
    // =================================================================
    bool b_enable_post_verification;        // Module 3开关
    double d_suspicious_conf_threshold;     // 可疑内点置信度阈值 (default: 0.5)
    int i_max_suspicious_verify_count;      // 最大验证数量 (default: 20)
    
    vector<int> filterSuspiciousInliers();  // 筛选可疑内点（复用Module 2的置信度）
    void verifyDepthRate(const vector<int>& suspicious_indices); // 深度率验证（复用Module 2的深度变化率）
    void postVerification();                // Module 3主入口函数
    
    // Helper functions - Local Filtering
    cv::Mat convertDepthToFloat(const cv::Mat& depth_img); // 转换深度图为32FC1格式
    float extractDepthWithLocalFilter(const cv::Mat& depth_float, cv::Point2f pt, int window_size); // 局部窗口滤波并提取深度
    float extractDepthFromRawImage(const cv::Mat& raw_depth_img, cv::Point2f pt); // 直接从原始深度图提取并滤波特征点深度
    float getSimpleDepth(const cv::Mat& raw_depth_img, cv::Point2f pt); // 最简单的深度提取方法
    
    // Debug and analysis functions
    void analyzeDepthImageQuality();           // 分析深度图质量，诊断深度值丢失原因
    void analyzeFeatureDepthCorrelation();     // 分析特征点分布与深度有效性的关系
    void analyzeDepthValueDistribution();      // 分析深度值分布，帮助确定合适的深度范围配置
    void analyzeFeatureDepthDistribution();    // 分析特征点的深度值分布
    void validateDepthExtraction();            // 验证深度值提取的正确性
    void analyzeDepthMatching();               // 分析前后帧深度匹配问题
    
    // 针对根源问题的专门调试函数
    void diagnoseFeatureIDDiscontinuity();     // 诊断特征点ID不连续问题
    void diagnoseDepthTimeSynchronization();   // 诊断深度图时序同步问题
    void diagnoseFeaturePositionDrift();       // 诊断特征点位置偏移问题
    
    // =================================================================
    // END: DG-RANSAC Implementation
    // =================================================================

private:
};
