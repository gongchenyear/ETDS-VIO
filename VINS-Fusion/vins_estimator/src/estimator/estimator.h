/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
 
#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/Image.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <vector>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"

using namespace std;
using namespace Eigen;

// 前向声明 DRT 适配器
class DRTAdapter;

// FeatureQualityScore is now defined in feature_tracker.h

class Estimator
{
  public:
    // ACE动态场景触发：可疑特征点信息结构
    struct SuspiciousFeatureInfo {
        int feature_id;                    // 特征点ID
        double quality_score;              // 前端质量评分
        double timestamp;                  // 时间戳
        int frame_id;                      // 帧ID
        Vector3d normalized_pos;           // 归一化坐标 (x, y, z)
        cv::Point2f pixel_pos;             // 像素坐标 (u, v)
        cv::Point2f velocity;              // 光流速度 (vx, vy)

        // ACE坐标筛选：额外信息
        double tracking_quality;           // 跟踪质量评分
        double optical_flow_score;         // 光流一致性评分
        double lifecycle_score;            // 生命周期评分
        int track_count;                   // 跟踪次数

        SuspiciousFeatureInfo() : feature_id(-1), quality_score(0.0), timestamp(0.0),
                                 frame_id(-1), normalized_pos(Vector3d::Zero()),
                                 pixel_pos(cv::Point2f(0,0)), velocity(cv::Point2f(0,0)),
                                 tracking_quality(0.0), optical_flow_score(0.0),
                                 lifecycle_score(0.0), track_count(0) {}
    };

    // [NEW] Initial Prior from DRT
    Eigen::Matrix<double, 15, 15> initial_P;
    bool use_initial_prior = false;
    
    // [NEW] Initial State Mean for PriorFactor
    Eigen::Vector3d initial_P_mean;
    Eigen::Vector3d initial_V_mean;
    Eigen::Vector3d initial_Ba_mean;
    Eigen::Vector3d initial_Bg_mean;
    Eigen::Quaterniond initial_R_mean;

    Estimator();
    ~Estimator();
    void setParameter();

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    void inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat(), 
                    std::queue<sensor_msgs::ImageConstPtr>* depth_buf = nullptr);
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1, 
                    std::queue<sensor_msgs::ImageConstPtr>* depth_buf, const cv::Mat &_depth_img);
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header);
    void processMeasurements();
    void changeSensorType(int use_imu, int use_stereo);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                              vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex, double threshold = -1.0);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates();
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector);
    
    // 新增：获取最新的相机速度和旋转（用于深度置信度计算）
    bool getLatestCameraMotion(Vector3d &velocity, Matrix3d &rotation);
    Vector3d getLatestCameraVelocity();
    Matrix3d getLatestCameraRotation();
    double getExperimentStartTime() const { return experiment_start_time_; }
    bool hasExperimentStartTime() const { return experiment_start_time_initialized_; }
    bool shouldLogFirstPose() const { return !first_pose_logged_; }
    void markFirstPoseLogged() { first_pose_logged_ = true; }

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    std::mutex mProcess;
    std::mutex mBuf;
    std::mutex mPropagate;
    std::queue<std::pair<double, Eigen::Vector3d>> accBuf;
    std::queue<std::pair<double, Eigen::Vector3d>> gyrBuf;
    std::queue<std::pair<double, std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;
    double prevTime, curTime;
    bool openExEstimation;

    std::thread trackThread;
    std::thread processThread;

    FeatureTracker featureTracker;

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;

    Matrix3d ric[2];
    Vector3d tic[2];

    Vector3d        Ps[(WINDOW_SIZE + 1)];
    Vector3d        Vs[(WINDOW_SIZE + 1)];
    Matrix3d        Rs[(WINDOW_SIZE + 1)];
    Vector3d        Bas[(WINDOW_SIZE + 1)];
    Vector3d        Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    int inputImageCnt;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[2][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;

    double latest_time;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;

    bool initFirstPoseFlag;
    bool initThreadFlag;

private:
    // ACE DSS容错机制：DSS指标类型枚举
    enum DSS_INDICATOR_TYPE {
        DSS_INDICATOR_FLOW,         // Flow指标（最敏感）
        DSS_INDICATOR_EPIPOLAR,     // Epipolar指标（最保守）
        DSS_INDICATOR_MULTIFRAME    // Multiframe指标（中等敏感性）
    };
    // ACE代码清理：移除旧的FeatureDSSInfo结构



    // ACE 中间帧逻辑清理：简化为纯关键帧时序异常分析结构
    struct TemporalAnomalyInfo {
        int feature_id;

        // 关键帧异常信息（双指标综合）
        double keyframe_combined_dss;     // 双指标综合评分
        double keyframe_flow_score;       // 关键帧Flow评分
        double keyframe_epipolar_score;   // 关键帧Epipolar评分
        bool keyframe_suspicious;         // 关键帧是否可疑
        int keyframe_id;                  // 关键帧ID
        double keyframe_timestamp;        // 关键帧时间戳

        // 关键帧时序一致性分析结果
        double temporal_consistency_score;  // 时序一致性评分 [0,1]
        bool is_temporal_anomaly;          // 是否为时序异常

        // 鲁棒性评估
        double robustness_confidence;      // 鲁棒性置信度 [0,1]

        TemporalAnomalyInfo() : feature_id(-1), keyframe_combined_dss(0.0), keyframe_flow_score(0.0),
                               keyframe_epipolar_score(0.0),
                               keyframe_suspicious(false), keyframe_id(-1), keyframe_timestamp(0.0),
                               temporal_consistency_score(0.0), is_temporal_anomaly(false),
                               robustness_confidence(0.0) {}
    };

    // DSS时序突变检测过滤器
    struct DSSTemporalFilter {
        std::map<int, double> last_dss_values_;
        const double DSS_JUMP_THRESHOLD = 2.0;  // DSS突变阈值（200%变化率）

        bool isAnomalousChange(int feature_id, double current_dss) {
            auto it = last_dss_values_.find(feature_id);
            if (it == last_dss_values_.end()) {
                // 首次计算，记录当前值
                last_dss_values_[feature_id] = current_dss;
                return false;
            }

            double previous_dss = it->second;
            double change_ratio = std::abs(current_dss - previous_dss) / (previous_dss + 1e-6);

            // 更新历史值
            last_dss_values_[feature_id] = current_dss;

            // DSS值突然增加超过200%认为是异常突变
            return change_ratio > DSS_JUMP_THRESHOLD;
        }

        double getLastValue(int feature_id) const {
            auto it = last_dss_values_.find(feature_id);
            return (it != last_dss_values_.end()) ? it->second : 0.0;
        }

        void clearStaleEntries() {
            // 简单的大小限制清理：如果记录过多，清理最旧的一半
            const size_t MAX_ENTRIES = 500;  // 最大记录数
            if (last_dss_values_.size() > MAX_ENTRIES) {
                auto it = last_dss_values_.begin();
                std::advance(it, last_dss_values_.size() / 2);
                last_dss_values_.erase(last_dss_values_.begin(), it);
            }
        }
    };

    // 注意：旧版动态检测状态变量已删除，现在使用FeatureTracker中的STATIC/ACTIVE状态机

    // ACE修复：恢复前端动态检测仍需要的变量
    int dynamic_dss_consistent_low_frames_;
    int stable_frames_count_;
    int last_keyframe_id_;
    std::vector<SuspiciousFeatureInfo> suspicious_features_;
    std::map<int, double> keyframe_suspicious_scores_;
    std::vector<int> robust_suspicious_feature_ids_;
    std::vector<int> suspicious_feature_ids_;

    // 新增：轻量级配置结构体
    struct DynamicDetectionConfig {
        double dss_trigger_threshold;
        double dss_hysteresis_ratio;
        // 移除：camera_speed_threshold - 不再使用速度门控
        int min_stable_frames;
        // ACE 多帧验证完善：重新引入min_trigger_frames参数
        int min_trigger_frames;              // 触发所需的最小连续帧数
        int min_reset_frames;                // 重置所需的最小连续帧数


        // DSS时序异常检测配置
        bool enable_dss_temporal_anomaly;    // 是否启用DSS时序异常检测
        double dss_jump_threshold;           // DSS突变阈值（变化率）
        double anomaly_boost_factor;         // 时序异常增强因子

        // 构造函数，从全局参数初始化
        DynamicDetectionConfig() {
            dss_trigger_threshold = 0.3;  // 默认值，运行时会更新
            dss_hysteresis_ratio = 0.8;   // 默认值，运行时会从配置文件更新
            // 移除：camera_speed_threshold - 不再使用速度门控
            min_stable_frames = 5;         // 默认值，运行时会更新
            // ACE 多帧验证完善：重新引入min_trigger_frames参数
            min_trigger_frames = 3;        // 默认值，运行时会从配置文件更新
            min_reset_frames = 15;         // 默认值，运行时会从配置文件更新

            // DSS时序异常检测默认配置
            enable_dss_temporal_anomaly = true;
            dss_jump_threshold = 2.0;      // 默认值，运行时会从配置文件更新
            anomaly_boost_factor = 0.2;    // 默认值，运行时会从配置文件更新
        }
    } dynamic_config_;
    
    // 注意：已移除全局DSS相关的历史缓冲区、时序衰减和趋势分析机制
    // 现在使用基于可疑特征点比例的简化动态检测逻辑
    
    // 注意：旧版动态检测函数已删除，现在使用DG-RANSAC前端处理

    // ==================== 深度处理模块接口预留 ====================
    // 获取当前帧的可疑特征点列表（供深度处理模块使用）
    const vector<SuspiciousFeatureInfo>& getCurrentSuspiciousFeatures() const {
        return suspicious_features_;
    }

    // 注意：旧版动态检测模式查询函数已删除，使用FeatureTracker::isInActiveState()

    // 深度处理结果接口（后续实现）
    // void processSuspiciousFeatureDepth(const vector<SuspiciousFeatureInfo>& features);
    // void updateFeatureWeights(const map<int, double>& feature_weights);
    // void markDynamicFeatures(const vector<int>& dynamic_feature_ids);

    // ACE修复：保留的辅助函数
    double calculateTemporalConsistencyForFeature(int feature_id);
    
    // 特征点生命周期管理函数已移除（未使用）









    // ACE代码清理：移除所有旧的DSS计算相关函数
    void analyzeTemporalDataAssociation();
    void updateRobustSuspiciousFeatures();

private:
    // ACE单一指标动态检测：已有成员变量复用，无需重复定义
    
    // DRT 初始化适配器
    std::unique_ptr<DRTAdapter> drt_adapter_;

    // Experiment-friendly timing logs (console only)
    bool experiment_start_time_initialized_ = false;
    bool first_pose_logged_ = false;
    double experiment_start_time_ = 0.0;
    
    // DRT 初始化后的宽限期计数器（用于逐步收紧外点阈值）
    int drt_init_grace_period_;

    // DRT 初始化后短窗健康守护（更严格的发散判定窗口）
    int drt_health_guard_period_;

    // DRT 初始先验在边缘化阶段的保留轮数，避免过早失去锚定
    int drt_prior_keep_marg_count_;

    // [NEW] VINS init failure counter for robust fallback
    int vins_init_fail_count = 0;
};
