/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <cmath> // For std::exp
#include <algorithm> // For std::sort
#include <numeric> // For std::accumulate
#include <iomanip> // For std::setprecision
#include "estimator.h"
#include "drt_adapter.h"
#include "../utility/visualization.h"

Estimator::Estimator(): f_manager{Rs}, drt_init_grace_period_(0), drt_health_guard_period_(0), drt_prior_keep_marg_count_(0)
{
    ROS_INFO("Estimator init");
    initThreadFlag = false;

    // 设置FeatureTracker的相机运动回调函数
    featureTracker.setCameraMotionCallback([this](Vector3d& velocity, Matrix3d& rotation) -> bool {
        return this->getLatestCameraMotion(velocity, rotation);
    });

    clearState();
    
    // 初始化 DRT 适配器
    // 为什么在这里初始化？确保 Estimator 创建时 DRT 就准备好
    drt_adapter_ = std::make_unique<DRTAdapter>(this);

}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState()
{
    mProcess.lock();
    while(!accBuf.empty())
        accBuf.pop();
    while(!gyrBuf.empty())
        gyrBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    experiment_start_time_initialized_ = false;
    experiment_start_time_ = 0.0;
    first_pose_logged_ = false;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    drt_init_grace_period_ = 0;
    drt_health_guard_period_ = 0;
    drt_prior_keep_marg_count_ = 0;
    vins_init_fail_count = 0;
    
    // 清理 DRT 先验状态，防止残留旧先验污染下次初始化
    use_initial_prior = false;
    initial_P.setZero();
    initial_P_mean.setZero();
    initial_V_mean.setZero();
    initial_Ba_mean.setZero();
    initial_Bg_mean.setZero();
    initial_R_mean = Eigen::Quaterniond::Identity();

    mProcess.unlock();
    
    // ==================== 动态检测相关初始化 ====================
    // 注意：旧版动态检测初始化已删除，现在使用FeatureTracker中的事件触发机制
    if (ENABLE_DYNAMIC_DETECTION) {
        dynamic_dss_consistent_low_frames_ = 0;
        stable_frames_count_ = 0;
        last_keyframe_id_ = -1;

        // 可疑特征点完整信息初始化
        suspicious_features_.clear();
        keyframe_suspicious_scores_.clear();
        robust_suspicious_feature_ids_.clear();

        // 注意：旧版trigger_candidate_frames_count_等变量已删除

        // 初始化DSS时序异常检测配置
        dynamic_config_.enable_dss_temporal_anomaly = (ENABLE_DSS_TEMPORAL_ANOMALY == 1);
        dynamic_config_.dss_jump_threshold = DSS_JUMP_THRESHOLD;
        dynamic_config_.anomaly_boost_factor = ANOMALY_BOOST_FACTOR;

        // ACE v58时序修复：参数赋值移动到setParameter()中，确保在参数加载后进行
        // 这里只进行基本初始化，参数赋值在setParameter()中完成

        ROS_INFO("Dynamic detection with temporal data association initialized");
        ROS_INFO_STREAM("DSS Temporal Anomaly Detection: "
                       << (dynamic_config_.enable_dss_temporal_anomaly ? "ENABLED" : "DISABLED")
                       << ", Jump Threshold: " << dynamic_config_.dss_jump_threshold
                       << ", Boost Factor: " << dynamic_config_.anomaly_boost_factor);
    }
}

void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }

    // ACE v58时序修复：在参数加载后正确设置动态检测参数
    if (ENABLE_DYNAMIC_DETECTION) {
        // ACE 多帧验证完善：重新引入min_trigger_frames参数设置
        dynamic_config_.min_trigger_frames = DYNAMIC_MIN_TRIGGER_FRAMES;
        dynamic_config_.min_reset_frames = DYNAMIC_MIN_RESET_FRAMES_INTERNAL;  // 修复：使用正确的参数名
        dynamic_config_.dss_hysteresis_ratio = DYNAMIC_DSS_HYSTERESIS_RATIO;

        ROS_INFO_STREAM("ACE 多帧验证完善: Dynamic detection parameters updated after config loading:");
        ROS_INFO_STREAM("  min_trigger_frames: " << dynamic_config_.min_trigger_frames);
        ROS_INFO_STREAM("  min_reset_frames: " << dynamic_config_.min_reset_frames);
        ROS_INFO_STREAM("  dss_hysteresis_ratio: " << dynamic_config_.dss_hysteresis_ratio);
    }

    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if(!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if(USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if(USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }
        
        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if(restart)
    {
        clearState();
        setParameter();
    }
}

void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1, 
                          std::queue<sensor_msgs::ImageConstPtr>* depth_buf)
{
    // 调用带深度图的重载版本，传入空深度图
    inputImage(t, _img, _img1, depth_buf, cv::Mat());
}

void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1, 
                          std::queue<sensor_msgs::ImageConstPtr>* depth_buf, const cv::Mat &_depth_img)
{
    inputImageCnt++;
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;

    // 深度图通过ROS topic直接传递给feature tracker，无需在此处理

    // 深度图通过参数传递
    if(_img1.empty())
        featureFrame = featureTracker.trackImage(t, _img, cv::Mat(), _depth_img);
    else
        featureFrame = featureTracker.trackImage(t, _img, _img1, _depth_img);
    //printf("featureTracker time: %f\n", featureTrackerTime.toc());

    // ACE清理：移除基于光流误差的前端动态检测调用

    if (SHOW_TRACK)
    {
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }
    
    // 并行喂给 DRT（在初始化阶段，可由配置关闭）
    if (ENABLE_DRT_FALLBACK && drt_adapter_ && solver_flag == INITIAL) {
        drt_adapter_->feedImage(t, featureFrame);
    }
    
    if(MULTIPLE_THREAD)  
    {     
        if(inputImageCnt % 2 == 0)
        {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame));
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
    
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    // 并行喂给 DRT（在初始化阶段，可由配置关闭）
    if (ENABLE_DRT_FALLBACK && drt_adapter_ && solver_flag == INITIAL) {
        drt_adapter_->feedIMU(t, linearAcceleration, angularVelocity);
    }
    
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if(!MULTIPLE_THREAD)
        processMeasurements();
}


bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if(t1 <= accBuf.back().first)
    {
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

bool Estimator::IMUAvailable(double t)
{
    if(!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

void Estimator::processMeasurements()
{
    while (1)
    {
        //printf("process measurments\n");
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        if(!featureBuf.empty())
        {
            feature = featureBuf.front();
            curTime = feature.first + td;
            while(1)
            {
                if ((!USE_IMU  || IMUAvailable(feature.first + td)))
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            mBuf.lock();
            if(USE_IMU)
                getIMUInterval(prevTime, curTime, accVector, gyrVector);

            featureBuf.pop();
            mBuf.unlock();

            if(USE_IMU)
            {
                if(!initFirstPoseFlag)
                    initFirstIMUPose(accVector);
                for(size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }
            mProcess.lock();
            
            // ========== 运动补偿：传递相机运动信息给前端 ==========
            // 在processImage之前设置，确保前端可以使用最新的相机运动信息
            if (frame_count > 0 && solver_flag == NON_LINEAR) {
                // Vs: world frame velocity
                // Rs: R_w_i (IMU -> world), so R_i_w = Rs.transpose()
                // Front-end depth-rate model uses camera normalized coordinates, so we
                // pass R_c_w (world -> camera) to avoid frame mismatch.
                Vector3d velocity = Vs[frame_count];
                Matrix3d R_i_w = Rs[frame_count].transpose();
                Matrix3d R_c_w = R_i_w;
                if (!RIC.empty()) {
                    // RIC[0] is R_i_c (camera -> IMU), thus R_c_i = R_i_c^T.
                    R_c_w = RIC[0].transpose() * R_i_w;
                }
                
                // ========== 坐标系验证（首次运行时检查）==========
                static bool coord_verified = false;
                if (!coord_verified && velocity.norm() > 0.1) {
                    Vector3d V_imu = R_i_w * velocity;
                    Vector3d V_camera = R_c_w * velocity;
                    
                    ROS_INFO("========== Coordinate System Verification ==========");
                    ROS_INFO("[CoordVerify] V_world = [%.3f, %.3f, %.3f] m/s, |v|=%.3f", 
                             velocity.x(), velocity.y(), velocity.z(), velocity.norm());
                    ROS_INFO("[CoordVerify] V_imu = [%.3f, %.3f, %.3f] m/s", 
                             V_imu.x(), V_imu.y(), V_imu.z());
                    ROS_INFO("[CoordVerify] V_camera = [%.3f, %.3f, %.3f] m/s", 
                             V_camera.x(), V_camera.y(), V_camera.z());
                    ROS_INFO("[CoordVerify] Using R_c_w for front-end motion compensation");
                    ROS_INFO("====================================================");
                    
                    coord_verified = true;
                }
                
                // Pass world velocity + world->camera rotation to front-end.
                featureTracker.setCameraMotion(velocity, R_c_w);
                
                ROS_DEBUG("[Estimator] Set camera motion: v=[%.3f,%.3f,%.3f] m/s, |v|=%.3f",
                         velocity.x(), velocity.y(), velocity.z(), velocity.norm());
            }
            // ========== 结束运动补偿 ==========
            
            processImage(feature.second, feature.first);
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world_dyn";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            mProcess.unlock();
        }

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}


void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());

    if (!experiment_start_time_initialized_) {
        experiment_start_time_initialized_ = true;
        experiment_start_time_ = header;
        ROS_INFO("[Experiment] Start time initialized: t0=%.6f", experiment_start_time_);
    }

    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    {
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                bool drt_used_for_init = false;
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    ROS_INFO("[Init] Attempt: t=%.6f, dt_from_start=%.3fs, frame_count=%d, feature_count=%d, drt_enabled=%s",
                             header,
                             experiment_start_time_initialized_ ? (header - experiment_start_time_) : 0.0,
                             frame_count,
                             f_manager.getFeatureCount(),
                             (ENABLE_DRT_FALLBACK && drt_adapter_) ? "YES" : "NO");

                    // VINS-first (conservative): only allow DRT after repeated VINS failures.
                    result = initialStructure();
                    if (result)
                    {
                        vins_init_fail_count = 0;
                        ROS_INFO("[Init] VINS initialization SUCCESS!");
                    }
                    else
                    {
                        vins_init_fail_count++;
                        const int remaining_for_drt = std::max(0, MAX_VINS_INIT_FAILURES - vins_init_fail_count);
                        ROS_WARN("[Init] VINS init failed (%d/%d before DRT allowed)",
                                 vins_init_fail_count, MAX_VINS_INIT_FAILURES);

                        if (ENABLE_DRT_FALLBACK && drt_adapter_ && vins_init_fail_count >= MAX_VINS_INIT_FAILURES)
                        {
                            ROS_WARN("[Init] VINS repeatedly failed, trying DRT fallback...");
                            result = drt_adapter_->tryInitialize();
                            if (result)
                            {
                                drt_used_for_init = true;
                                vins_init_fail_count = 0;
                                ROS_INFO("[Init] DRT fallback initialization SUCCESS!");
                            }
                        }
                        else if (ENABLE_DRT_FALLBACK && drt_adapter_)
                        {
                            ROS_INFO("[Init] DRT fallback deferred, need %d more VINS init failures",
                                     remaining_for_drt);
                        }
                    }

                    initial_timestamp = header;
                }
                if(result)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    ROS_INFO("[Init] Success: mode=%s, dt_from_start=%.3fs, frame_count=%d",
                             drt_used_for_init ? "DRT-fallback" : "VINS",
                             experiment_start_time_initialized_ ? (header - experiment_start_time_) : 0.0,
                             frame_count);

                    // 初始化完成后立即检查bias估计值
                    Vector3d Ba_init = Bas[0];
                    Vector3d Bg_init = Bgs[0];
                    double Ba_norm = Ba_init.norm();
                    double Bg_norm = Bg_init.norm();
                    ROS_WARN("[INIT-CHECK] Initial bias estimates:");
                    ROS_WARN("[INIT-CHECK]   Ba: [%.6f, %.6f, %.6f] (norm=%.6f)",
                             Ba_init.x(), Ba_init.y(), Ba_init.z(), Ba_norm);
                    ROS_WARN("[INIT-CHECK]   Bg: [%.6f, %.6f, %.6f] (norm=%.6f)",
                             Bg_init.x(), Bg_init.y(), Bg_init.z(), Bg_norm);

                    // 检查bias是否超出合理范围 - 如果超出，拒绝VINS初值，让DRT继续尝试
                    bool bias_out_of_range = false;
                    if (Ba_norm > 0.7) {
                        ROS_ERROR("[INIT-CHECK] REJECT: Ba_norm=%.3f EXCEEDS THRESHOLD 0.7!", Ba_norm);
                        bias_out_of_range = true;
                    }
                    if (Bg_norm > 0.02) {
                        ROS_ERROR("[INIT-CHECK] REJECT: Bg_norm=%.6f EXCEEDS THRESHOLD 0.02!", Bg_norm);
                        bias_out_of_range = true;
                    }

                    // 输出初始位姿和速度
                    Vector3d P_init = Ps[frame_count];
                    Vector3d V_init = Vs[frame_count];
                    ROS_WARN("[INIT-CHECK]   P: [%.3f, %.3f, %.3f] (norm=%.3f)",
                             P_init.x(), P_init.y(), P_init.z(), P_init.norm());
                    ROS_WARN("[INIT-CHECK]   V: [%.3f, %.3f, %.3f] (norm=%.3f)",
                             V_init.x(), V_init.y(), V_init.z(), V_init.norm());

                    // 如果VINS初值质量太差，拒绝此初化，触发DRT降级
                    if (bias_out_of_range && !drt_used_for_init)
                    {
                        ROS_ERROR("[INIT-CHECK] VINS init rejected due to poor bias estimates. Defer to DRT fallback.");
                        result = false;  // 强制初化失败，继续等待DRT
                    }

                    // 关键修复：初始化后重置FeatureTracker的动态检测状态
                    // 防止因初始化期间的追踪波动误触发DG-RANSAC，导致因深度缓存未就绪而跟踪失败
                    featureTracker.resetDynamicState();
                    
                    // Only enable the relaxed outlier grace period when DRT actually performed initialization.
                    if (drt_used_for_init)
                    {
                        drt_init_grace_period_ = 10;
                        drt_health_guard_period_ = 30;
                        drt_prior_keep_marg_count_ = 6;
                        ROS_INFO("[Estimator] DRT init success, grace period set to %d frames", drt_init_grace_period_);
                        ROS_INFO("[Estimator] DRT health guard set to %d frames, prior keep marg count=%d",
                                 drt_health_guard_period_, drt_prior_keep_marg_count_);
                    }
                    else
                    {
                        drt_init_grace_period_ = 0;
                        drt_health_guard_period_ = 0;
                        drt_prior_keep_marg_count_ = 0;
                        ROS_INFO("[Estimator] VINS init success, no DRT grace period applied");
                    }
                    
                    // 关键修复：只有当窗口满时才进行滑动窗口边缘化
                    // 如果DRT初始化返回的帧数少于WINDOW_SIZE，不应调用slideWindow，而是让后续帧填充窗口
                    if (frame_count == WINDOW_SIZE) {
                        slideWindow();
                    }
                    
                    ROS_INFO("Initialization finish!");
                }
                else {
                    ROS_WARN("[Init] Failed: dt_from_start=%.3fs, frame_count=%d, feature_count=%d",
                             experiment_start_time_initialized_ ? (header - experiment_start_time_) : 0.0,
                             frame_count,
                             f_manager.getFeatureCount());
                    slideWindow();
                }
            }
        }

        // stereo + IMU initilization
        if(STEREO && USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();

            if(frame_count == WINDOW_SIZE)
            {
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {
        TicToc t_solve;
        if(!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        
        // Phase 3已经在"间隙"中执行（processImage开始处）
        // 动态点已经从filtered_image中过滤，不会进入feature_manager
        // 因此这里不需要再次删除
        
        // 后端优化 - 使用的是已经清理过的特征点集合
        
        // 调试：记录优化前的状态
        ROS_WARN("[OPT-DEBUG] ===== Before Optimization (frame_count=%d) =====", frame_count);
        for (int i = 0; i <= frame_count; i++) {
            ROS_WARN("[OPT-DEBUG] Frame[%d] P:[%.6f, %.6f, %.6f] V:[%.6f, %.6f, %.6f] Ba:[%.6f, %.6f, %.6f] Bg:[%.6f, %.6f, %.6f]", 
                     i, Ps[i].x(), Ps[i].y(), Ps[i].z(), 
                     Vs[i].x(), Vs[i].y(), Vs[i].z(),
                     Bas[i].x(), Bas[i].y(), Bas[i].z(),
                     Bgs[i].x(), Bgs[i].y(), Bgs[i].z());
        }
        
        optimization();
        
        // [NEW] 调试：计算scale变化（基于位置）
        double scale_change = 0.0;
        if (frame_count > 0) {
            double before_dist = (Ps[frame_count] - Ps[0]).norm();
            // 这里before_dist实际上是优化后的，我们需要在optimization()内部记录
            ROS_WARN("[SCALE-DEBUG] Distance between frame 0 and %d: %.6f meters", frame_count, before_dist);
        }
        
        // 调试：记录优化后的状态变化
        ROS_WARN("[OPT-DEBUG] ===== After Optimization (frame_count=%d) =====", frame_count);
        for (int i = 0; i <= frame_count; i++) {
            double vel_norm = Vs[i].norm();
            ROS_WARN("[OPT-DEBUG] Frame[%d] P:[%.6f, %.6f, %.6f] (|p|=%.3f) V:[%.6f, %.6f, %.6f] (|v|=%.3f) Ba:[%.6f, %.6f, %.6f] Bg:[%.6f, %.6f, %.6f]", 
                     i, Ps[i].x(), Ps[i].y(), Ps[i].z(), Ps[i].norm(),
                     Vs[i].x(), Vs[i].y(), Vs[i].z(), vel_norm,
                     Bas[i].x(), Bas[i].y(), Bas[i].z(),
                     Bgs[i].x(), Bgs[i].y(), Bgs[i].z());
        }
        
        // 后端外点剔除（基于重投影误差）
        set<int> removeIndex;
        
        double outlier_threshold = -1.0; // 使用默认值
        if (drt_init_grace_period_ > 0) {
            drt_init_grace_period_--;
            // 分级收紧阈值，降低坏约束持续注入后端的风险。
            if (drt_init_grace_period_ >= 8)
                outlier_threshold = 10.0;
            else if (drt_init_grace_period_ >= 6)
                outlier_threshold = 8.0;
            else if (drt_init_grace_period_ >= 4)
                outlier_threshold = 6.0;
            else if (drt_init_grace_period_ >= 2)
                outlier_threshold = 4.5;
            else
                outlier_threshold = 3.5;

            ROS_INFO("[Estimator] Relaxed outliersRejection (grace period: %d remaining, threshold: %.1f)",
                     drt_init_grace_period_, outlier_threshold);
        }

        if (drt_health_guard_period_ > 0)
            drt_health_guard_period_--;
        
        outliersRejection(removeIndex, outlier_threshold);

        // 旧版VINS动态检测已完全禁用，使用新的前端事件触发机制
        // 新机制：FeatureTracker中的STATIC/ACTIVE状态机 + 阶段二/三处理流水线
        // 旧版检测通过初始化时的 ENABLE_DYNAMIC_DETECTION && false 条件禁用
        /*
        if (ENABLE_DYNAMIC_DETECTION && solver_flag == NON_LINEAR) {
            performVINSBasedDynamicDetection(removeIndex);
        }
        */

        f_manager.removeOutlier(removeIndex);
        if (! MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
        }
            
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        // [CLEANUP] Disabled failure detection - was introduced during DRT migration
        // but not needed by original VINS. Returning early prevented normal optimization
        // convergence when bias was being legitimately adjusted (cafe1-2 failure case).
        // Original VINS-Fusion had this hardcoded to `return false;` - we follow that.
        /*
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }
        */

        // 关键修复：在NON_LINEAR模式下，如果窗口未满，需要增加frame_count
        // 这是为了支持DRT初始化后的正常运行（DRT初始化时frame_count < WINDOW_SIZE）
        if (frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }
        else
        {
            slideWindow();
        }
        
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();

        // ACE清理：移除基于光流误差评分的动态检测逻辑
    }
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    // 修复VLA错误：使用std::vector代替变长数组
    vector<Quaterniond> Q(frame_count + 1);
    vector<Vector3d> T(frame_count + 1);
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    int skipped_recent_features = 0;
    int skipped_late_start_features = 0;
    for (auto &it_per_id : f_manager.feature)
    {
        int start_frame = it_per_id.start_frame;
        if (start_frame >= frame_count - 2)
        {
            skipped_recent_features++;
            continue;
        }
        if (it_per_id.start_frame > frame_count * 3.0 / 4.0)
        {
            skipped_late_start_features++;
            continue;
        }
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_sfm_f;
        tmp_sfm_f.state = false;
        tmp_sfm_f.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_sfm_f.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_sfm_f);
    }
    ROS_INFO("[Init][SFM] candidate_features=%zu from_feature_manager=%zu, skipped_recent=%d, skipped_late_start=%d",
             sfm_f.size(), f_manager.feature.size(), skipped_recent_features, skipped_late_start_features);
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q.data(), T.data(), l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_WARN("[Init][SFM] global SFM failed after relativePose success: ref_frame=%d, sfm_features=%zu",
                 l, sfm_f.size());
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_WARN("[Init][PnP] Not enough points for solvePnP at frame index %d: points=%zu", i, pts_3_vector.size());
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_WARN("[Init][PnP] solvePnP failed at frame index %d with points=%zu", i, pts_3_vector.size());
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_WARN("[Init][Align] VisualIMUAlignment failed");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    double best_parallax_px = 0.0;
    int best_correspondence_count = 0;
    int best_frame_index = -1;
    bool had_enough_correspondence = false;
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            had_enough_correspondence = true;
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            const double average_parallax_px = average_parallax * FOCAL_LENGTH;
            if (average_parallax_px > best_parallax_px)
            {
                best_parallax_px = average_parallax_px;
                best_correspondence_count = static_cast<int>(corres.size());
                best_frame_index = i;
            }
            if (average_parallax_px > INIT_REL_POSE_MIN_PARALLAX_PX &&
                m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_INFO("[Init][RelativePose] success: ref_frame=%d, correspondences=%zu, average_parallax=%.3fpx, threshold=%.3fpx",
                         l, corres.size(), average_parallax_px, INIT_REL_POSE_MIN_PARALLAX_PX);
                return true;
            }
        }
    }
    ROS_INFO("[Init][RelativePose] failed: reason=%s, best_ref_frame=%d, best_correspondences=%d, best_average_parallax=%.3fpx, threshold=%.3fpx",
             had_enough_correspondence ? "parallax_below_threshold_or_pose_solve_failed" : "insufficient_correspondences",
             best_frame_index, best_correspondence_count, best_parallax_px, INIT_REL_POSE_MIN_PARALLAX_PX);
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }


    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

                Bas[i] = Vector3d(para_SpeedBias[i][3],
                                  para_SpeedBias[i][4],
                                  para_SpeedBias[i][5]);

                Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                  para_SpeedBias[i][7],
                                  para_SpeedBias[i][8]);
            
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if(USE_IMU)
        td = para_Td[0][0];

}

bool Estimator::failureDetection()
{
    // [CLEANUP] Disabled failure detection - was introduced during DRT migration
    // but contradicts original VINS-Fusion design (which had `return false;` at start).
    // Keeping old logic as dead code for reference, but disabled to match original behavior.
    return false;
    
    /*
    // === DISABLED CODE BELOW ===
    int idx = std::min(frame_count, WINDOW_SIZE);

    if (idx < 0)
        return false;

    if (!std::isfinite(Ps[idx].x()) || !std::isfinite(Ps[idx].y()) || !std::isfinite(Ps[idx].z()) ||
        !std::isfinite(Vs[idx].x()) || !std::isfinite(Vs[idx].y()) || !std::isfinite(Vs[idx].z()) ||
        !std::isfinite(Bas[idx].x()) || !std::isfinite(Bas[idx].y()) || !std::isfinite(Bas[idx].z()) ||
        !std::isfinite(Bgs[idx].x()) || !std::isfinite(Bgs[idx].y()) || !std::isfinite(Bgs[idx].z()))
    {
        ROS_ERROR("[FAILURE] Non-finite state detected at frame %d", idx);
        return true;
    }

    double pos_div_thresh = 200.0;
    double vel_div_thresh = 20.0;
    // Relax hard Ba guardrail to avoid reset on borderline values (e.g. 0.50x).
    double ba_div_thresh = 0.7;
    double bg_div_thresh = 0.02;

    // 初始化后短窗采用更严格阈值，尽早阻断发散链条。
    if (drt_health_guard_period_ > 0)
    {
        pos_div_thresh = 60.0;
        vel_div_thresh = 8.0;
        ba_div_thresh = 0.30;
        bg_div_thresh = 0.01;
    }

    // Hard divergence guardrail: once exceeded, current estimate is unusable.
    if (Ps[idx].norm() > pos_div_thresh || Vs[idx].norm() > vel_div_thresh)
    {
        ROS_ERROR("[FAILURE] Divergence detected at frame %d: |P|=%.3f, |V|=%.3f (thresh: %.1f/%.1f)",
                  idx, Ps[idx].norm(), Vs[idx].norm(), pos_div_thresh, vel_div_thresh);
        return true;
    }

    // Bias explosion guardrail observed in DRT post-init failures.
    if (Bas[idx].norm() > ba_div_thresh || Bgs[idx].norm() > bg_div_thresh)
    {
        ROS_ERROR("[FAILURE] Bias explosion at frame %d: |Ba|=%.3f, |Bg|=%.6f (thresh: %.2f/%.3f)",
                  idx, Bas[idx].norm(), Bgs[idx].norm(), ba_div_thresh, bg_div_thresh);
        return true;
    }

    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
    // === END DISABLED CODE ===
    */
}

// [FIX] Moved PriorFactor class definition to global scope (anonymous namespace)
// so it can be instantiated in optimization() and cloned for marginalization.
namespace {
class PriorFactor : public ceres::SizedCostFunction<15, 7, 9> {
public:
    PriorFactor(const Eigen::Matrix<double, 15, 15>& covariance, 
                const double* para_Pose_prior, 
                const double* para_SpeedBias_prior) {
        ROS_INFO("Creating PriorFactor at %p", this);
        // Compute Square Root Information Matrix
        // Info = Cov^-1
        // We use LLT to get J such that J^T * J = Info
        // Residual r = J * e
        // Cost = r^T * r = e^T * J^T * J * e = e^T * Info * e
        
        Eigen::Matrix<double, 15, 15> info = covariance.inverse();
        // Check for validity
        if (info.hasNaN()) {
            ROS_ERROR("[PriorFactor] Inverse covariance has NaN! Using Identity.");
            sqrt_info.setIdentity();
        } else {
            Eigen::LLT<Eigen::Matrix<double, 15, 15>> llt(info);
            sqrt_info = llt.matrixL().transpose(); // Upper triangular
        }
        
        // Store priors
        P_prior = Eigen::Vector3d(para_Pose_prior[0], para_Pose_prior[1], para_Pose_prior[2]);
        Q_prior = Eigen::Quaterniond(para_Pose_prior[6], para_Pose_prior[3], para_Pose_prior[4], para_Pose_prior[5]);
        V_prior = Eigen::Vector3d(para_SpeedBias_prior[0], para_SpeedBias_prior[1], para_SpeedBias_prior[2]);
        Ba_prior = Eigen::Vector3d(para_SpeedBias_prior[3], para_SpeedBias_prior[4], para_SpeedBias_prior[5]);
        Bg_prior = Eigen::Vector3d(para_SpeedBias_prior[6], para_SpeedBias_prior[7], para_SpeedBias_prior[8]);
    }
    

    
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        // Parameters
        Eigen::Vector3d P(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        
        Eigen::Vector3d V(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d Ba(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d Bg(parameters[1][6], parameters[1][7], parameters[1][8]);
        
        // Error state (15x1)
        Eigen::Matrix<double, 15, 1> error;
        error.segment<3>(0) = P - P_prior;
        
        // Rotation error: 2 * Log(Q_prior^-1 * Q)
        // Approximation: if Q = Q_prior * [1, theta/2], then theta = 2 * (Q_prior^-1 * Q).vec
        Eigen::Quaterniond dQ = Q_prior.inverse() * Q;
        error.segment<3>(3) = 2.0 * dQ.vec(); // Valid for small errors
        if (dQ.w() < 0) {
            // [FIX] Manual negation for quaternion
            Eigen::Quaterniond neg_dQ(-dQ.w(), -dQ.x(), -dQ.y(), -dQ.z());
            error.segment<3>(3) = -2.0 * neg_dQ.vec();
        }
        
        error.segment<3>(6) = V - V_prior;
        error.segment<3>(9) = Ba - Ba_prior;
        error.segment<3>(12) = Bg - Bg_prior;
        
        // Residual = sqrt_info * error
        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        residual = sqrt_info * error;
        
        if (jacobians) {
            // Jacobian w.r.t Pose (7)
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                jacobian_pose.setZero();
                
                jacobian_pose.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                
                Eigen::Matrix3d J_r_inv = Eigen::Matrix3d::Identity();
                jacobian_pose.block<3, 3>(3, 3) = J_r_inv;
                
                Eigen::Quaterniond q_p = Q_prior.conjugate();
                Eigen::Matrix<double, 3, 4> J_q;
                J_q << q_p.w(), -q_p.z(),  q_p.y(), q_p.x(),
                       q_p.z(),  q_p.w(), -q_p.x(), q_p.y(),
                      -q_p.y(),  q_p.x(),  q_p.w(), q_p.z();
                J_q *= 2.0;
                
                Eigen::Matrix<double, 15, 7> J_error_pose;
                J_error_pose.setZero();
                J_error_pose.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                J_error_pose.block<3, 4>(3, 3) = J_q;
                
                jacobian_pose = sqrt_info * J_error_pose;
            }
            
            // Jacobian w.r.t SpeedBias (9)
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_sb(jacobians[1]);
                jacobian_sb.setZero();
                
                Eigen::Matrix<double, 15, 9> J_error_sb;
                J_error_sb.setZero();
                J_error_sb.block<3, 3>(6, 0) = Eigen::Matrix3d::Identity();
                J_error_sb.block<3, 3>(9, 3) = Eigen::Matrix3d::Identity();
                J_error_sb.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();
                
                jacobian_sb = sqrt_info * J_error_sb;
            }
        }
        
        return true;
    }
    
    ~PriorFactor() {
        ROS_WARN("Deleting PriorFactor at %p", this);
    }

private:
    Eigen::Matrix<double, 15, 15> sqrt_info;
    Eigen::Vector3d P_prior;
    Eigen::Quaterniond Q_prior;
    Eigen::Vector3d V_prior;
    Eigen::Vector3d Ba_prior;
    Eigen::Vector3d Bg_prior;
};
} // namespace

void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    
    // 调试：打印进入optimization时的状态
    ROS_DEBUG("solver costs: %f", 0.0);
    
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        
        if(USE_IMU) {
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
        }
    }
    if(!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // 调试：打印marginalization参数块
        ROS_WARN("[OPT] Adding marginalization factor with %zu parameter blocks:", 
                 last_marginalization_parameter_blocks.size());
        for (size_t i = 0; i < last_marginalization_parameter_blocks.size(); i++) {
            void* addr = last_marginalization_parameter_blocks[i];
            // 检查这个地址是para_Pose还是para_SpeedBias
            bool is_pose = false, is_speed = false;
            int pose_idx = -1, speed_idx = -1;
            for (int j = 0; j <= WINDOW_SIZE; j++) {
                if (addr == para_Pose[j]) {
                    is_pose = true;
                    pose_idx = j;
                }
                if (addr == para_SpeedBias[j]) {
                    is_speed = true;
                    speed_idx = j;
                }
            }
            if (is_pose && is_speed) {
                ROS_ERROR("[OPT] Block %zu: addr=%p is BOTH para_Pose[%d] AND para_SpeedBias[%d]!",
                          i, addr, pose_idx, speed_idx);
            } else if (is_pose) {
                ROS_INFO("[OPT]   Block %zu: para_Pose[%d] = %p", i, pose_idx, addr);
            } else if (is_speed) {
                ROS_INFO("[OPT]   Block %zu: para_SpeedBias[%d] = %p", i, speed_idx, addr);
            } else {
                ROS_INFO("[OPT]   Block %zu: other = %p", i, addr);
            }
        }
        
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    // [NEW] DRT Initial Prior Factor
    // Only add if use_initial_prior is true (set by DRTAdapter)
    // [FIX] Declare PriorFactor pointer outside to be used in marginalization
    PriorFactor* prior_factor = nullptr;

    if (use_initial_prior)
    {
        ROS_INFO("[OPT] Adding DRT Initial Prior Factor...");
        
        // Add to problem
        // Prior is on the FIRST frame (frame 0)
        
        // [FIX] Use explicit prior means set by DRTAdapter
        // This ensures we anchor to the ALIGNED state, not whatever happens to be in para_Pose[0]
        // (though they should be the same if vector2double worked correctly).
        
        double prior_pose_buf[7];
        double prior_speedbias_buf[9];
        
        prior_pose_buf[0] = initial_P_mean.x();
        prior_pose_buf[1] = initial_P_mean.y();
        prior_pose_buf[2] = initial_P_mean.z();
        prior_pose_buf[3] = initial_R_mean.x();
        prior_pose_buf[4] = initial_R_mean.y();
        prior_pose_buf[5] = initial_R_mean.z();
        prior_pose_buf[6] = initial_R_mean.w();
        
        prior_speedbias_buf[0] = initial_V_mean.x();
        prior_speedbias_buf[1] = initial_V_mean.y();
        prior_speedbias_buf[2] = initial_V_mean.z();
        prior_speedbias_buf[3] = initial_Ba_mean.x();
        prior_speedbias_buf[4] = initial_Ba_mean.y();
        prior_speedbias_buf[5] = initial_Ba_mean.z();
        prior_speedbias_buf[6] = initial_Bg_mean.x();
        prior_speedbias_buf[7] = initial_Bg_mean.y();
        prior_speedbias_buf[8] = initial_Bg_mean.z();
        
        prior_factor = new PriorFactor(initial_P, prior_pose_buf, prior_speedbias_buf);
        problem.AddResidualBlock(prior_factor, NULL, para_Pose[0], para_SpeedBias[0]);
        
        // [FIX] Only reset flag if we are about to marginalize the prior (i.e. frame_count == WINDOW_SIZE)
        // If frame_count < WINDOW_SIZE, we return early and don't marginalize, so we need to add prior again next time.
        if (frame_count == WINDOW_SIZE && marginalization_flag == MARGIN_OLD) {
            if (drt_prior_keep_marg_count_ > 0) {
                drt_prior_keep_marg_count_--;
                use_initial_prior = true;
                ROS_INFO("[OPT] DRT Initial Prior kept for stabilization, remaining marg rounds: %d",
                         drt_prior_keep_marg_count_);
            } else {
                use_initial_prior = false;
                ROS_INFO("[OPT] DRT Initial Prior Factor added and will be MARGINALIZED.");
            }
        } else {
            ROS_INFO("[OPT] DRT Initial Prior Factor added (persisting until marginalization).");
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        // 关键修复：跳过start_frame超出frame_count范围的特征
        if (imu_i > frame_count)
            continue;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            
            // 关键修复：跳过超出frame_count范围的观测
            if (imu_j > frame_count)
                break;
            
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if(STEREO && it_per_frame.is_stereo)
            {                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
               
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    double2vector();

    // 优化后检查bias变化和系统状态
    static double last_Ba_norm = -1.0;
    static double last_Bg_norm = -1.0;
    static int opt_count = 0;
    opt_count++;

    // 统计视觉约束质量
    int total_features = 0;
    int valid_features = 0;
    double avg_parallax = 0.0;
    for (auto &it_per_id : f_manager.feature) {
        total_features++;
        if (it_per_id.estimated_depth > 0) {
            valid_features++;
        }
    }

    for (int i = 0; i <= frame_count; i++) {
        double Ba_norm = Bas[i].norm();
        double Bg_norm = Bgs[i].norm();

        // 检测异常bias
        if (Ba_norm > 0.7 || Bg_norm > 0.02) {
            ROS_ERROR("[BIAS-ALERT] Frame %d (opt#%d): Ba_norm=%.3f, Bg_norm=%.6f EXCEEDS THRESHOLD",
                      i, opt_count, Ba_norm, Bg_norm);
        }

        // 检测突变（仅对最新帧）
        if (i == frame_count) {
            if (last_Ba_norm > 0) {
                double Ba_change = fabs(Ba_norm - last_Ba_norm);
                double Bg_change = fabs(Bg_norm - last_Bg_norm);

                // 记录前30次优化的详细信息
                if (opt_count <= 30 || Ba_change > 0.1 || Bg_change > 0.005) {
                    ROS_WARN("[OPT-MONITOR] opt#%d: Ba=%.3f(Δ%.3f) Bg=%.6f(Δ%.6f) |P|=%.1f |V|=%.2f features=%d/%d",
                             opt_count, Ba_norm, Ba_change, Bg_norm, Bg_change,
                             Ps[frame_count].norm(), Vs[frame_count].norm(),
                             valid_features, total_features);
                }

                if (Ba_change > 0.1) {
                    ROS_WARN("[BIAS-JUMP] Frame %d: Ba changed by %.3f (%.3f -> %.3f)",
                             i, Ba_change, last_Ba_norm, Ba_norm);
                }
                if (Bg_change > 0.005) {
                    ROS_WARN("[BIAS-JUMP] Frame %d: Bg changed by %.6f (%.6f -> %.6f)",
                             i, Bg_change, last_Bg_norm, Bg_norm);
                }
            }
            last_Ba_norm = Ba_norm;
            last_Bg_norm = Bg_norm;
        }
    }

    // 检测位姿发散
    double pos_norm = Ps[frame_count].norm();
    double vel_norm = Vs[frame_count].norm();
    if (pos_norm > 200 || vel_norm > 20) {
        ROS_ERROR("[DIVERGENCE] Frame %d (opt#%d): |P|=%.1f, |V|=%.1f",
                  frame_count, opt_count, pos_norm, vel_norm);
    }

    // 早期预警：检测潜在的不稳定迹象（使用最新帧的Ba）
    double current_Ba_norm = Bas[frame_count].norm();
    if (opt_count <= 50 && (current_Ba_norm > 0.3 || pos_norm > 50)) {
        ROS_ERROR("[EARLY-WARNING] opt#%d: System showing instability signs - Ba=%.3f, |P|=%.1f",
                  opt_count, current_Ba_norm, pos_norm);
    }
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)
        return;
    
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        // [FIX] Create a NEW instance of PriorFactor for MarginalizationInfo
        // Do NOT copy! The copy constructor corrupts the base class state.
        // Instead, create a fresh instance with the same parameters.
        if (prior_factor) {
            ROS_WARN("[MARG] Adding PriorFactor to marginalization info");
            
            // [FIX] Re-construct prior buffers to ensure consistent linearization point
            // We must use the SAME prior mean as the one used in optimization, NOT the current state in para_Pose[0]
            double prior_pose_buf[7];
            double prior_speedbias_buf[9];
            prior_pose_buf[0] = initial_P_mean.x();
            prior_pose_buf[1] = initial_P_mean.y();
            prior_pose_buf[2] = initial_P_mean.z();
            prior_pose_buf[3] = initial_R_mean.x();
            prior_pose_buf[4] = initial_R_mean.y();
            prior_pose_buf[5] = initial_R_mean.z();
            prior_pose_buf[6] = initial_R_mean.w();
            
            prior_speedbias_buf[0] = initial_V_mean.x();
            prior_speedbias_buf[1] = initial_V_mean.y();
            prior_speedbias_buf[2] = initial_V_mean.z();
            prior_speedbias_buf[3] = initial_Ba_mean.x();
            prior_speedbias_buf[4] = initial_Ba_mean.y();
            prior_speedbias_buf[5] = initial_Ba_mean.z();
            prior_speedbias_buf[6] = initial_Bg_mean.x();
            prior_speedbias_buf[7] = initial_Bg_mean.y();
            prior_speedbias_buf[8] = initial_Bg_mean.z();

            PriorFactor* prior_factor_for_marg = new PriorFactor(initial_P, prior_pose_buf, prior_speedbias_buf);
            
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(prior_factor_for_marg, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0]},
                                                                           vector<int>{0, 1}); // Drop both Pose[0] and SpeedBias[0]
            marginalization_info->addResidualBlockInfo(residual_block_info);
        } else {
            ROS_WARN("[MARG] prior_factor is NULL, skipping marg addition");
        }

        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    
                    // 关键修复：防止访问超出窗口范围的参数块
                    if (imu_j > WINDOW_SIZE) break;
                    
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if(STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if(imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        // 调试：打印marginalization后的参数块
        ROS_WARN("[MARG] After MARGIN_OLD marginalization:");
        ROS_WARN("[MARG] Got %zu parameter blocks from getParameterBlocks:", parameter_blocks.size());
        for (size_t i = 0; i < parameter_blocks.size(); i++) {
            void* addr = parameter_blocks[i];
            bool is_pose = false, is_speed = false;
            int pose_idx = -1, speed_idx = -1;
            for (int j = 0; j <= WINDOW_SIZE; j++) {
                if (addr == para_Pose[j]) {
                    is_pose = true;
                    pose_idx = j;
                }
                if (addr == para_SpeedBias[j]) {
                    is_speed = true;
                    speed_idx = j;
                }
            }
            if (is_pose && is_speed) {
                ROS_ERROR("[MARG] Block %zu: addr=%p is BOTH para_Pose[%d] AND para_SpeedBias[%d]!",
                          i, addr, pose_idx, speed_idx);
            } else if (is_pose) {
                ROS_WARN("[MARG]   Block %zu: para_Pose[%d] = %p", i, pose_idx, addr);
            } else if (is_speed) {
                ROS_WARN("[MARG]   Block %zu: para_SpeedBias[%d] = %p", i, speed_idx, addr);
            } else {
                ROS_WARN("[MARG]   Block %zu: other = %p", i, addr);
            }
        }

        if (last_marginalization_info) {
            ROS_WARN("[OPT] Deleting last_marginalization_info...");
            delete last_marginalization_info;
            ROS_WARN("[OPT] Deleted last_marginalization_info.");
        }
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_WARN("[OPT] End of optimization function. Problem will be destroyed.");
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}



void Estimator::slideWindow()
{
    TicToc t_margin;

    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                if (it_0 != all_image_frame.end()) {
                    if (it_0->second.pre_integration) {
                        delete it_0->second.pre_integration;
                        it_0->second.pre_integration = nullptr;
                    }
                    all_image_frame.erase(it_0);
                }
                // Clean up any older frames that might have been missed
                all_image_frame.erase(all_image_frame.begin(), all_image_frame.lower_bound(t_0));
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex, double threshold)
{
    //return;
    double err_thresh = 3.0; // 默认3像素
    if (threshold > 0) {
        err_thresh = threshold;
    }
    
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;

        // ACE零额外开销优化：计算运动幅度缓存
        double max_motion = 0.0;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                    Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;

                // ACE零额外开销优化：同时计算运动幅度（基于VINS已有的位姿数据）
                if (imu_i >= 0 && imu_j >= 0 && imu_i < WINDOW_SIZE + 1 && imu_j < WINDOW_SIZE + 1) {
                    Vector3d t_change = Ps[imu_j] - Ps[imu_i];
                    Matrix3d R_change = Rs[imu_j] * Rs[imu_i].transpose();
                    double translation_mag = t_change.norm();
                    double rotation_mag = acos(std::min(1.0, std::max(-1.0, (R_change.trace() - 1) / 2)));
                    double motion_mag = translation_mag + rotation_mag;
                    max_motion = std::max(max_motion, motion_mag);
                }
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(STEREO && it_per_frame.is_stereo)
            {

                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
            }
        }

        // ACE零额外开销优化：缓存计算结果供动态检测使用
        if (errCnt > 0) {
            double ave_err = err / errCnt;
            if(ave_err * FOCAL_LENGTH > err_thresh)
                removeIndex.insert(it_per_id.feature_id);
        }
    }
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
    
    // Phase 3: 更新全局IMU速度和旋转矩阵（线程安全，保持时刻一致性）
    {
        std::lock_guard<std::mutex> lock(VELOCITY_MUTEX);
        LATEST_CAMERA_VELOCITY = Vs[frame_count];      // 速度（世界坐标系）

        // Store world->camera rotation for downstream depth-rate checks.
        Matrix3d R_i_w = Rs[frame_count].transpose();
        LATEST_CAMERA_ROTATION = R_i_w;
        if (!RIC.empty()) {
            LATEST_CAMERA_ROTATION = RIC[0].transpose() * R_i_w;
        }

        // 关键新增：传递角速度
        // 从最新的IMU测量值中减去估计的陀螺仪bias
        // angular_velocity_buf[frame_count] 存储了当前帧的所有IMU角速度测量值
        if (!angular_velocity_buf[frame_count].empty()) {
            // 使用最后一个IMU测量值作为最佳估计
            Vector3d latest_raw_omega = angular_velocity_buf[frame_count].back();
            // Bgs[frame_count] 是当前帧的陀螺仪bias估计
            // NOTE: This remains in IMU frame; convert to camera frame where used.
            LATEST_CAMERA_ANGULAR_VELOCITY = latest_raw_omega - Bgs[frame_count];
        } else {
            // 如果缓冲区为空（理论上不应发生），则保持为零
            LATEST_CAMERA_ANGULAR_VELOCITY = Vector3d::Zero();
        }

        HAS_VALID_CAMERA_VELOCITY = (solver_flag == NON_LINEAR);

        ROS_INFO_THROTTLE(1.0,
            "[VelocityUpdate] Frame %d: V=%.3f m/s [%.3f, %.3f, %.3f], Omega=[%.3f, %.3f, %.3f], solver=%d",
            frame_count,
            Vs[frame_count].norm(),
            Vs[frame_count].x(), Vs[frame_count].y(), Vs[frame_count].z(),
            LATEST_CAMERA_ANGULAR_VELOCITY.x(), LATEST_CAMERA_ANGULAR_VELOCITY.y(), LATEST_CAMERA_ANGULAR_VELOCITY.z(),
            (int)solver_flag);
    }
}

// ACE代码清理：移除旧的后端DSS动态检测逻辑，现在使用前端动态检测







// ACE修复：添加前端动态检测仍需要的函数实现
double Estimator::calculateTemporalConsistencyForFeature(int feature_id)
{
    // 简化实现：返回固定值
    return 0.5;
}





// ACE代码清理：移除所有旧的DSS计算函数



















// 注意：旧版VINS动态检测函数已删除，现在使用FeatureTracker事件触发机制

// 注意：旧版动态检测函数已删除，现在使用DG-RANSAC前端处理

void Estimator::analyzeTemporalDataAssociation()
{
    if (!ENABLE_DYNAMIC_DETECTION || keyframe_suspicious_scores_.empty()) {
        return;
    }

    ROS_INFO_STREAM("=== 开始时序数据关联异常分析 ===");

    // ACE 中间帧逻辑清理：简化为纯关键帧时序分析
    int total_keyframe_suspicious = keyframe_suspicious_scores_.size();
    double max_keyframe_score = 0.0, min_keyframe_score = 1.0;

    for (const auto& [feature_id, keyframe_score] : keyframe_suspicious_scores_) {
        max_keyframe_score = std::max(max_keyframe_score, keyframe_score);
        min_keyframe_score = std::min(min_keyframe_score, keyframe_score);
    }

    ROS_INFO_STREAM("=== 关键帧时序数据统计 ===");
    ROS_INFO_STREAM("关键帧可疑特征点: " << total_keyframe_suspicious << " 个");
    ROS_INFO_STREAM("关键帧评分范围: [" << std::fixed << std::setprecision(3) << min_keyframe_score
                   << ", " << max_keyframe_score << "]");

    // 分析每个在关键帧上可疑的特征点的时序一致性
    std::vector<std::pair<int, double>> temporal_analysis_results;
    std::vector<double> consistency_scores;
    int features_with_temporal_data = 0;

    // ACE 中间帧逻辑清理：简化为关键帧时序一致性分析
    for (const auto& [feature_id, keyframe_score] : keyframe_suspicious_scores_) {
        double temporal_consistency = calculateTemporalConsistencyForFeature(feature_id);

        ROS_DEBUG_STREAM("Feature " << feature_id << " keyframe temporal analysis:");
        ROS_DEBUG_STREAM("  Keyframe score: " << std::fixed << std::setprecision(3) << keyframe_score);
        ROS_DEBUG_STREAM("  Temporal consistency: " << temporal_consistency);

        if (temporal_consistency > 0.0) {
            temporal_analysis_results.emplace_back(feature_id, temporal_consistency);
            consistency_scores.push_back(temporal_consistency);
            features_with_temporal_data++;
        }
    }

    // ==================== 新增：时序一致性分布统计 ====================
    if (!consistency_scores.empty()) {
        std::sort(consistency_scores.begin(), consistency_scores.end());
        double median_consistency = consistency_scores[consistency_scores.size() / 2];
        double avg_consistency = std::accumulate(consistency_scores.begin(), consistency_scores.end(), 0.0) / consistency_scores.size();
        double max_consistency = consistency_scores.back();
        double percentile_75 = consistency_scores[static_cast<size_t>(consistency_scores.size() * 0.75)];

        ROS_INFO_STREAM("=== 时序一致性分布 ===");
        ROS_INFO_STREAM("有时序数据特征点: " << features_with_temporal_data << "/" << total_keyframe_suspicious);
        ROS_INFO_STREAM("一致性评分范围: [0, " << std::fixed << std::setprecision(3) << max_consistency << "]");
        ROS_INFO_STREAM("一致性统计: 平均=" << avg_consistency << ", 中位=" << median_consistency
                       << ", 75%分位=" << percentile_75);
    }

    // 按时序一致性评分排序
    std::sort(temporal_analysis_results.begin(), temporal_analysis_results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // 更新鲁棒的可疑特征点列表
    updateRobustSuspiciousFeatures();

    // ACE 中间帧逻辑清理：输出关键帧时序分析结果
    ROS_INFO_STREAM("Keyframe temporal data analysis completed:");
    ROS_INFO_STREAM("  Keyframe suspicious features: " << keyframe_suspicious_scores_.size());
    ROS_INFO_STREAM("  Features with temporal data: " << temporal_analysis_results.size());
    ROS_INFO_STREAM("  Robust suspicious points after temporal verification: " << robust_suspicious_feature_ids_.size());

    // 输出前3个最可疑的时序异常特征点
    int debug_count = 0;
    for (const auto& [feature_id, consistency_score] : temporal_analysis_results) {
        if (debug_count >= 3) break;

        ROS_INFO_STREAM("  Feature " << feature_id
                       << ": temporal consistency=" << std::fixed << std::setprecision(3) << consistency_score
                       << ", keyframe score=" << keyframe_suspicious_scores_[feature_id]);
        debug_count++;
    }
}

void Estimator::updateRobustSuspiciousFeatures()
{
    robust_suspicious_feature_ids_.clear();

    // 时序一致性阈值：要求较高的一致性才认为是真正的动态特征点
    const double temporal_consistency_threshold = DYNAMIC_TEMPORAL_CONSISTENCY_THRESHOLD;  // 使用配置参数

    for (const auto& [feature_id, keyframe_score] : keyframe_suspicious_scores_) {
        double temporal_consistency = calculateTemporalConsistencyForFeature(feature_id);

        // 时序验证条件：
        // 1. 关键帧上可疑（已经满足，因为在keyframe_suspicious_scores_中）
        // 2. 时序一致性超过阈值
        if (temporal_consistency > temporal_consistency_threshold) {
            robust_suspicious_feature_ids_.push_back(feature_id);
        }
    }

    // 计算鲁棒性提升效果
    double robustness_improvement = 0.0;
    if (!suspicious_feature_ids_.empty()) {
        robustness_improvement = (1.0 - static_cast<double>(robust_suspicious_feature_ids_.size()) /
                                        suspicious_feature_ids_.size()) * 100.0;
    }

    ROS_INFO_STREAM("Temporal verification results: "
                   << "Original suspicious features: " << suspicious_feature_ids_.size()
                   << " → After temporal verification: " << robust_suspicious_feature_ids_.size()
                   << " (Robustness improvement: " << std::fixed << std::setprecision(1) << robustness_improvement << "%)");
}

// 旧使用DG-RANSAC前端处理






// 新增：获取最新的相机运动信息（用于深度置信度计算）
bool Estimator::getLatestCameraMotion(Vector3d &velocity, Matrix3d &rotation)
{
    if (solver_flag != NON_LINEAR || frame_count < WINDOW_SIZE) {
        return false;  // 系统未初始化完成
    }
    
    // Return world velocity + world->camera rotation.
    velocity = Vs[WINDOW_SIZE];
    Matrix3d R_i_w = Rs[WINDOW_SIZE].transpose();
    rotation = R_i_w;
    if (!RIC.empty()) {
        // RIC[0] is R_i_c (camera -> IMU), so R_c_w = R_c_i * R_i_w.
        rotation = RIC[0].transpose() * R_i_w;
    }
    
    // 调试日志：验证坐标系转换
    static int debug_count = 0;
    if (debug_count < 3) {
        Vector3d V_camera = rotation * velocity;  // 转换到相机坐标系
        ROS_INFO("[MotionDebug] World velocity: [%.3f, %.3f, %.3f] m/s", 
                 velocity.x(), velocity.y(), velocity.z());
        ROS_INFO("[MotionDebug] Camera velocity: [%.3f, %.3f, %.3f] m/s", 
                 V_camera.x(), V_camera.y(), V_camera.z());
        debug_count++;
    }
    
    return true;
}

Vector3d Estimator::getLatestCameraVelocity()
{
    if (solver_flag != NON_LINEAR || frame_count < WINDOW_SIZE) {
        return Vector3d::Zero();
    }
    
    return Vs[WINDOW_SIZE];
}

Matrix3d Estimator::getLatestCameraRotation()
{
    if (solver_flag != NON_LINEAR || frame_count < WINDOW_SIZE) {
        return Matrix3d::Identity();
    }
    
    return Rs[WINDOW_SIZE];
}
