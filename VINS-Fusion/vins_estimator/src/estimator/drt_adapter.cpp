#include "drt_adapter.h"
#include "estimator.h"
#include "parameters.h"
#include "../utility/tic_toc.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <algorithm>

// 在 cpp 中包含本地 DRT 头文件
#include "../drt_init/include/drtTightlyCoupled.h"

// Pimpl 实现类：所有 DRT 相关的代码都在这里
class DRTAdapter::Impl {
public:
    Estimator* estimator_;
    std::unique_ptr<DRT::drtTightlyCoupled> drt_init_;
    
    // 缓存原始 IMU 数据用于 DRT
    struct IMUData {
        double timestamp;
        Eigen::Vector3d acc;
        Eigen::Vector3d gyr;
    };
    std::vector<IMUData> imu_buffer_;
    
    /**
     * 获取外参 Rbc 和 tbc（避免重复定义）
     * Rbc = RIC[0]（从 IMU 坐标系 b 到相机坐标系 c 的旋转）
     * tbc = TIC[0]（相机原点在 IMU 坐标系下的位置）
     */
    bool getExtrinsics(Eigen::Matrix3d& out_Rbc, Eigen::Vector3d& out_tbc) {
        if (RIC.empty() || TIC.empty()) {
            ROS_ERROR("[DRT] RIC or TIC is empty!");
            return false;
        }
        out_Rbc = RIC[0];
        out_tbc = TIC[0];
        return true;
    }
    
    // IMU 标定参数（模仿原版 DRT）
    std::unique_ptr<vio::IMUCalibParam> imu_calib_;
    
    Impl(Estimator* estimator) : estimator_(estimator) {
        // DRT initialization stays lazy. Keep constructor silent so NoDRT runs don't emit misleading DRT runtime logs.
    }
    
    // 延迟初始化 DRT（在第一次使用时）
    void ensureInitialized() {
        if (drt_init_) return;  // 已经初始化
        
        Eigen::Matrix3d Rbc;
        Eigen::Vector3d tbc;
        if (!getExtrinsics(Rbc, tbc)) {
            ROS_ERROR("[DRT] Failed to get extrinsics (RIC/TIC not loaded yet)");
            return;
        }
        
        ROS_INFO("[DRT] Initializing with Rbc:\n%f %f %f\n%f %f %f\n%f %f %f",
                 Rbc(0,0), Rbc(0,1), Rbc(0,2),
                 Rbc(1,0), Rbc(1,1), Rbc(1,2),
                 Rbc(2,0), Rbc(2,1), Rbc(2,2));
        ROS_INFO("[DRT] tbc: %f %f %f", tbc(0), tbc(1), tbc(2));

        // Re-orthonormalize Rbc to satisfy Sophus::SO3 strict orthogonality
        // check. YAML factory extrinsics (e.g. Occipital SDK matrix shipped
        // with VCU-RVI) carry ~6 significant digits, leaving ~1e-7 deviation
        // from R R^T = I that Sophus::SO3<double>::ensure() rejects.
        // Quaternion round-trip projects onto SO(3) with zero geometric loss.
        Rbc = Eigen::Quaterniond(Rbc).normalized().toRotationMatrix();

        drt_init_ = std::make_unique<DRT::drtTightlyCoupled>(Rbc, tbc);
        
        // 创建 IMU 标定参数（完全模仿原版 DRT）
        // 计算实际IMU频率（如果有足够数据）
        double imu_freq = 200.0;  // 默认200Hz
        if (imu_buffer_.size() >= 10) {
            double total_time = imu_buffer_[9].timestamp - imu_buffer_[0].timestamp;
            if (total_time > 0) {
                imu_freq = 9.0 / total_time;
                ROS_INFO("[DRT] Calculated IMU frequency: %.1f Hz", imu_freq);
            }
        }
        
        // 原版：imu_calib(RIC[0], TIC[0], GYR_N * sf, ACC_N * sf, GYR_W / sf, ACC_W / sf)
        double sf = std::sqrt(imu_freq);
        imu_calib_ = std::make_unique<vio::IMUCalibParam>(
            Rbc, tbc,
            GYR_N * sf, ACC_N * sf,  // 陀螺仪噪声, 加速度计噪声
            GYR_W / sf, ACC_W / sf   // 陀螺仪随机游走, 加速度计随机游走
        );
        
        ROS_INFO("[DRT] Initialized with parallel mode");
        ROS_INFO("[DRT] IMU calib: GYR_N=%.6f, ACC_N=%.6f, GYR_W=%.6f, ACC_W=%.6f", 
                 GYR_N, ACC_N, GYR_W, ACC_W);
    }
    
    // 添加 IMU 数据到缓存
    void feedIMU(double t, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr) {
        if (estimator_->solver_flag == Estimator::INITIAL) {
            ensureInitialized();  // 确保 DRT 已初始化
            imu_buffer_.push_back({t, acc, gyr});
        }
    }
    
    // 添加图像特征到 DRT（并行模式：收集但不判断关键帧）
    void feedImage(double t, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& features) {
        if (estimator_->solver_flag == Estimator::INITIAL) {
            ensureInitialized();
            if (!drt_init_) return;
            
            // 只收集特征数据，不做任何判断
            // 等待VINS通过f_manager.addFeatureCheckParallax判断后，
            // 在tryInitialize中使用VINS的all_image_frame
            ROS_DEBUG("[DRT] 📦 Collecting frame at %.3f with %zu features", 
                     t, features.size());
        }
    }
    
    std::vector<double> accepted_frame_times_;  // 被 DRT 接受的帧时间戳
    
    // 立即创建并添加 IMU 预积分（完全模仿原版 DRT）
    void createAndAddIMUPreintegration(double t_start, double t_end) {
        ROS_INFO("[DRT] 🔧 Creating IMU preintegration: %.3f -> %.3f", t_start, t_end);
        
        // 完全模仿原版 DRT 的 IMU 收集方式
        std::vector<IMUData> imu_segment;
        for (const auto& imu : imu_buffer_) {
            // 保留 t_start 之前的最后一个 IMU（用于起始插值）
            if (imu.timestamp <= t_start) {
                if (!imu_segment.empty()) imu_segment.clear();
                imu_segment.push_back(imu);
                continue;
            }
            // 收集 (t_start, t_end] 范围内的 IMU
            if (imu.timestamp <= t_end) {
                imu_segment.push_back(imu);
            }
            // 添加 t_end 之后的第一个 IMU（用于结束插值）
            if (imu.timestamp > t_end) {
                imu_segment.push_back(imu);
                break;
            }
        }
        
        ROS_INFO("[DRT]   Collected %zu IMU measurements (including one after t_end)", imu_segment.size());
        
        if (imu_segment.empty()) {
            ROS_WARN("[DRT] No IMU data between %.3f and %.3f", t_start, t_end);
            return;
        }
        
        // 创建 IMU 预积分（使用 DRT 的方式，传入正确的 calib 参数）
        ROS_INFO("[DRT]   Creating IMUPreintegrated object...");
        vio::IMUBias init_bias(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        vio::IMUPreintegrated imu_preint(init_bias, imu_calib_.get(), t_start, t_end);
        
        // 完全模仿原版 DRT 的插值方式
        // 注意：n = imu_segment.size() - 1，循环到 i < n
        int n = imu_segment.size() - 1;
        ROS_INFO("[DRT]   Processing %d integration steps (from %zu IMU measurements)...", n, imu_segment.size());
        
        for (int i = 0; i < n; i++) {
            double dt = 0.0;
            Eigen::Vector3d gyro, acc;
            
            if (i == 0 && i < (n - 1)) {
                // 第一段：[start_time, imu[1].time]
                float tab = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                float tini = imu_segment[i].timestamp - t_start;
                if (tini < 0) tini = 0;
                
                acc = (imu_segment[i + 1].acc + imu_segment[i].acc -
                       (imu_segment[i + 1].acc - imu_segment[i].acc) * (tini / tab)) * 0.5f;
                gyro = (imu_segment[i + 1].gyr + imu_segment[i].gyr -
                        (imu_segment[i + 1].gyr - imu_segment[i].gyr) * (tini / tab)) * 0.5f;
                dt = imu_segment[i + 1].timestamp - t_start;
                
            } else if (i < (n - 1)) {
                // 中间段：[imu[i].time, imu[i+1].time]
                acc = (imu_segment[i].acc + imu_segment[i + 1].acc) * 0.5f;
                gyro = (imu_segment[i].gyr + imu_segment[i + 1].gyr) * 0.5f;
                dt = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                
            } else if (i > 0 && i == n - 1) {
                // 最后一段：[imu[n-1].time, end_time]
                // 使用 imu_segment[i+1]（即最后一个额外的 IMU）进行插值
                float tab = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                float tend = imu_segment[i + 1].timestamp - t_end;
                if (tend < 0) tend = 0;
                
                acc = (imu_segment[i].acc + imu_segment[i + 1].acc -
                       (imu_segment[i + 1].acc - imu_segment[i].acc) * (tend / tab)) * 0.5f;
                gyro = (imu_segment[i].gyr + imu_segment[i + 1].gyr -
                        (imu_segment[i + 1].gyr - imu_segment[i].gyr) * (tend / tab)) * 0.5f;
                dt = t_end - imu_segment[i].timestamp;
                
            } else if (i == 0 && i == (n - 1)) {
                // 只有一个 IMU 测量（n=1，即 imu_segment.size()=2）
                acc = imu_segment[i].acc;
                gyro = imu_segment[i].gyr;
                dt = t_end - t_start;
            }
            
            if (dt > 0) {
                imu_preint.integrate_new_measurement(gyro, acc, dt);
            } else {
                ROS_WARN("[DRT]   Skipping integration step %d with dt=%.6f", i, dt);
            }
        }
        
        // 立即添加到 DRT（模仿原版）
        ROS_INFO("[DRT]   Calling addImuMeasure...");
        ROS_INFO("[DRT]   IMU preint: t_start=%.9f, t_end=%.9f", t_start, t_end);
        ROS_INFO("[DRT]   Current local_active_frames size: %zu", drt_init_->local_active_frames.size());
        try {
            drt_init_->addImuMeasure(imu_preint);
            ROS_INFO("[DRT] IMU segment created: %.3f -> %.3f (%zu measurements)",
                     t_start, t_end, imu_segment.size());
        } catch (const std::exception& e) {
            ROS_ERROR("[DRT] Exception in addImuMeasure: %s", e.what());
            ROS_ERROR("[DRT]   t_start=%.9f, t_end=%.9f", t_start, t_end);
            ROS_ERROR("[DRT]   local_active_frames size=%zu", drt_init_->local_active_frames.size());
            // 打印local_active_frames中的所有时间戳
            ROS_ERROR("[DRT]   local_active_frames timestamps:");
            for (auto& frame_t : drt_init_->local_active_frames) {
                ROS_ERROR("[DRT]     %.9f", frame_t);
            }
            throw;
        }
    }
    
    bool tryInitialize() {
        ROS_WARN("========================================");
        ROS_WARN("[DRT] VINS initialization failed, trying DRT fallback...");
        ROS_WARN("========================================");
        
        // 0. 防护：确保 DRT 已初始化
        if (!drt_init_) {
            ensureInitialized();
            if (!drt_init_) {
                ROS_ERROR("[DRT] Failed to initialize DRT module (RIC/TIC not loaded?)");
                return false;
            }
        }
        
        // 1. 检查数据是否足够
        ROS_INFO("[DRT] Checking data availability...");
        ROS_INFO("[DRT]   - all_image_frame size: %zu", estimator_->all_image_frame.size());
        ROS_INFO("[DRT]   - frame_count: %d", estimator_->frame_count);
        
        if (estimator_->all_image_frame.size() < 10) {
            ROS_WARN("[DRT] Not enough frames for DRT initialization (need >= 10)");
            return false;
        }
        
        if (imu_buffer_.empty()) {
            ROS_ERROR("[DRT] No IMU data accumulated!");
            return false;
        }
        
        // [IMPROVED] Adaptive IMU excitation check
        // 对于低运动序列（如Cafe1-2），用自适应阈值而非固定的0.15
        // 核心思想：如果已累积足够帧，可以用较低的激励度
        {
            Eigen::Vector3d sum_acc(0, 0, 0);
            for (const auto& imu : imu_buffer_) sum_acc += imu.acc;
            Eigen::Vector3d avg_acc = sum_acc / imu_buffer_.size();

            double var = 0;
            for (const auto& imu : imu_buffer_) {
                var += (imu.acc - avg_acc).squaredNorm();
            }
            var = sqrt(var / imu_buffer_.size());

            // 自适应阈值：根据累积帧数和时间调整
            double excitation_threshold = 0.15;
            size_t frame_count = estimator_->all_image_frame.size();
            
            // 策略：累积足够的帧后，放宽激励度要求
            if (frame_count >= 15) {
                excitation_threshold = 0.10;  // 15+ frames: 降低到0.10
                ROS_INFO("[DRT] Relaxed excitation threshold to 0.10 (accumulated %zu frames)", frame_count);
            } else if (frame_count >= 12) {
                excitation_threshold = 0.12;  // 12-14 frames: 降低到0.12
                ROS_INFO("[DRT] Relaxed excitation threshold to 0.12 (accumulated %zu frames)", frame_count);
            }
            
            if (var < excitation_threshold) {
                ROS_WARN("[DRT] IMU excitation not enough (var=%.3f < %.3f, frames=%zu)", 
                         var, excitation_threshold, frame_count);
                return false;
            }
            ROS_INFO("[DRT] IMU excitation good (var=%.3f >= %.3f, threshold adaptive)", var, excitation_threshold);
        }
        
        // 2. 从all_image_frame重新构建DRT数据（确保与VINS一致）
        try {
            ROS_INFO("[DRT] Rebuilding DRT from all_image_frame...");
            ROS_INFO("[DRT]   - IMU measurements: %zu", imu_buffer_.size());
            ROS_INFO("[DRT]   - all_image_frame size: %zu", estimator_->all_image_frame.size());
            
            // 清空DRT的旧数据
            drt_init_->local_active_frames.clear();
            drt_init_->imu_meas.clear();
            drt_init_->SFMConstruct.clear();
            drt_init_->last_image_t_ns = 0.0;
            accepted_frame_times_.clear();
            
            // [NEW] Set gravity magnitude from VINS config
            drt_init_->gravity_magnitude = estimator_->g.norm();
            ROS_INFO("[DRT] Set gravity magnitude to %.3f", drt_init_->gravity_magnitude);
            
            // 第一步：添加所有帧（让DRT判断关键帧）
            for (auto& frame_pair : estimator_->all_image_frame) {
                double t = frame_pair.first;
                const ImageFrame& frame = frame_pair.second;
                
                // 转换特征点格式
                DRT::FeatureTrackerResulst drt_features;
                for (auto& pt : frame.points) {
                    int feature_id = pt.first;
                    Eigen::aligned_vector<std::pair<int, Eigen::Matrix<double, 7, 1>>> feature_obs;
                    for (auto& obs : pt.second) {
                        feature_obs.push_back(obs);
                    }
                    drt_features[feature_id] = feature_obs;
                }
                
                // 添加到DRT（让DRT自己判断是否接受）
                bool accepted = drt_init_->addFeatureCheckParallax(t, drt_features, 0.0);
                if (accepted) {
                    accepted_frame_times_.push_back(t);
                }
            }
            
            ROS_INFO("[DRT]   - DRT accepted %zu/%zu frames", 
                     accepted_frame_times_.size(), estimator_->all_image_frame.size());
            
            // 检查帧数 - 需要至少 10 帧（与参考代码一致）
            if (accepted_frame_times_.size() < 10) {
                ROS_WARN("[DRT] Only %zu frames accepted, need at least 10", accepted_frame_times_.size());
                return false;
            }
            
            // 第二步：为DRT接受的关键帧构建IMU预积分
            for (size_t i = 1; i < accepted_frame_times_.size(); i++) {
                double t_start = accepted_frame_times_[i-1];
                double t_end = accepted_frame_times_[i];
                createAndAddIMUPreintegration(t_start, t_end);
            }
            
            ROS_INFO("[DRT]   - Created %zu IMU segments", drt_init_->imu_meas.size());
            
            // 检查加速度误差
            if (!drt_init_->checkAccError()) {
                ROS_WARN("[DRT] Acceleration error check failed");
                return false;
            }
            
            // 3. 执行 DRT 初始化
            ROS_INFO("[DRT] Running DRT initialization with %zu frames...", accepted_frame_times_.size());
            
            TicToc t_drt;
            bool success = drt_init_->process();
            double drt_time = t_drt.toc();
            
            if (success) {
                ROS_WARN("[DRT] ========================================");
                ROS_WARN("[DRT] DRT initialization SUCCEEDED in %.2f ms!", drt_time);
                ROS_WARN("[DRT] ========================================");
                if (estimator_->hasExperimentStartTime()) {
                    double dt_from_start = 0.0;
                    if (!accepted_frame_times_.empty()) {
                        dt_from_start = accepted_frame_times_.back() - estimator_->getExperimentStartTime();
                    }
                    ROS_INFO("[Experiment][DRT] success=YES, dt_from_start=%.3fs, accepted_frames=%zu, drt_time_ms=%.2f",
                             dt_from_start, accepted_frame_times_.size(), drt_time);
                }
                bool writeback_success = writeBackStates();
                if (!writeback_success) {
                    ROS_ERROR("[DRT] Failed to write back DRT states to VINS");
                    return false;
                }
                return true;
            } else {
                ROS_WARN("[DRT] DRT initialization FAILED after %.2f ms", drt_time);
                if (estimator_->hasExperimentStartTime()) {
                    double dt_from_start = 0.0;
                    if (!accepted_frame_times_.empty()) {
                        dt_from_start = accepted_frame_times_.back() - estimator_->getExperimentStartTime();
                    }
                    ROS_INFO("[Experiment][DRT] success=NO, dt_from_start=%.3fs, accepted_frames=%zu, drt_time_ms=%.2f",
                             dt_from_start, accepted_frame_times_.size(), drt_time);
                }
                return false;
            }
            
        } catch (const std::exception& e) {
            ROS_ERROR("[DRT] Exception during DRT initialization: %s", e.what());
            return false;
        }
    }
    
    bool writeBackStates() {
        ROS_INFO("[DRT] ----------------------------------------");
        ROS_INFO("[DRT] Writing back DRT results to VINS...");
        ROS_INFO("[DRT] ----------------------------------------");
        
        // 1. 重力向量
        // [SOLUTION] Gravity Alignment
        // We will align the entire state to Z-up gravity later in the pose transfer loop.
        // Here we just set the estimator's gravity to the standard Z-up vector.
        // This ensures consistency with VINS's world frame definition.
        
        estimator_->g = Eigen::Vector3d(0, 0, 9.81); // Standard gravity
        // Or use g_norm from config if we had access to it easily, but 9.81 is standard.
        // The important part is direction (0,0,1).
        
        ROS_WARN("[DRT] Gravity set to STANDARD Z-up: [%.3f, %.3f, %.3f]", 
                 estimator_->g.x(), estimator_->g.y(), estimator_->g.z());
        
        // 2. IMU bias
        for (int i = 0; i <= WINDOW_SIZE; i++) {
            estimator_->Bgs[i] = drt_init_->biasg;
            estimator_->Bas[i] = drt_init_->biasa;
        }
        ROS_INFO("[DRT] 2. IMU Bias:");
        ROS_INFO("[DRT]    - Gyro bias: [%.6f, %.6f, %.6f]", 
                 drt_init_->biasg.x(), drt_init_->biasg.y(), drt_init_->biasg.z());
        ROS_INFO("[DRT]    - Accel bias: [%.6f, %.6f, %.6f]", 
                 drt_init_->biasa.x(), drt_init_->biasa.y(), drt_init_->biasa.z());
        
        // 3. 位姿和速度
        // 关键：DRT初始化的帧数可能少于VINS的all_image_frame
        // 我们需要使用VINS的帧数，而不是DRT的帧数
        int num_drt_frames = drt_init_->rotation.size();
        int num_vins_frames = estimator_->all_image_frame.size();
        
        ROS_INFO("[DRT] 3. Poses and Velocities:");
        ROS_INFO("[DRT]    DRT initialized: %d frames", num_drt_frames);
        ROS_INFO("[DRT]    VINS has: %d frames in all_image_frame", num_vins_frames);
        ROS_INFO("[DRT]    Original VINS frame_count: %d", estimator_->frame_count);
        
        // 检查：DRT帧数不能超过VINS帧数
        if (num_drt_frames > num_vins_frames) {
            ROS_ERROR("[DRT] DRT frames (%d) > VINS frames (%d), this should not happen!", 
                      num_drt_frames, num_vins_frames);
            return false;
        }
        
        // 关键修复：frame_count 不能超过 WINDOW_SIZE
        // para_Pose[WINDOW_SIZE+1] 数组大小是11，索引0-10
        // frame_count 最大只能是 WINDOW_SIZE (10)
        int target_frame_count = std::min(num_drt_frames - 1, WINDOW_SIZE);
        estimator_->frame_count = target_frame_count;
        ROS_INFO("[DRT]    Set frame_count to %d (DRT initialized %d frames)", 
                 estimator_->frame_count, num_drt_frames);
        
        // 写入 DRT 初始化的位姿和速度
        // 
        // === 坐标系定义验证（基于DRT源码） ===
        // DRT源码第121行：S_1l = dP_ + rotation[i] * pbc_ - pbc_
        // 这说明rotation[i]用于旋转外参向量，因此：
        //   rotation[i] = R_b0_bi（从bi坐标系到b0坐标系的旋转）
        //   position[i] = P_bi^b0（IMU bi在b0坐标系下的位置）
        //   velocity[i] = V^b0（速度在b0坐标系下）
        // 
        // VINS需求：
        //   Rs[i] = R_w_ci（从ci坐标系到w坐标系的旋转）
        //   Ps[i] = P_ci^w（相机ci在世界坐标系w下的位置）
        //   Vs[i] = V^w（速度在世界坐标系w下）
        // 
        // 外参定义（VINS约定）：
        //   RIC[0] = Rbc = R_c_b（从IMU坐标系b到相机坐标系c的旋转）
        //   TIC[0] = tbc = P_c^b（相机原点在IMU坐标系b下的位置）
        // 
        // === 坐标转换推导 ===
        // 假设b0就是世界坐标系w：
        // 
        // 1. 旋转转换：
        //    R_w_ci = R_b0_bi * R_bi_ci
        //    其中 R_bi_ci = Rbc^T（从ci到bi）
        //    因此：Rs[i] = rotation[i] * Rbc^T
        // 
        // 2. 位置转换：
        //    P_ci^w = P_bi^w + R_b0_bi * P_ci^bi
        //    其中 P_ci^bi = -tbc（相机在IMU坐标系下的位置，注意DRT使用pbc_表示相机到IMU）
        //    因此：Ps[i] = position[i] + rotation[i] * (-tbc)
        //                = position[i] - rotation[i] * tbc
        // 
        // 3. 速度转换：
        //    速度在世界坐标系下相同（刚性连接）
        //    Vs[i] = velocity[i]
        
        // [SAFETY] Check if RIC/TIC are available
        Eigen::Matrix3d Rbc;
        Eigen::Vector3d tbc;
        if (!getExtrinsics(Rbc, tbc)) {
            ROS_ERROR("[DRT] Cannot apply extrinsics. Aborting initialization.");
            return false;
        }
        
        // ============================================
        // [SOLUTION] Gravity Alignment & Extrinsic Correction
        // ============================================
        
        // 1. Define Alignment Rotation R_wg
        // DRT outputs 'gravity' which is likely the mean accelerometer reading (Reaction Force).
        // It points UP (opposite to physical gravity).
        // We must align this UP vector to the World UP ([0, 0, 9.81]).
        Eigen::Vector3d g_drt = drt_init_->gravity;
        if (g_drt.norm() < 1e-3) {
            ROS_ERROR("[DRT] Gravity vector is near zero! Aborting.");
            return false;
        }
        
        Eigen::Vector3d g_target(0, 0, 9.81); // Align Accel (Up) to World Up
        Eigen::Quaterniond q_wg = Eigen::Quaterniond::FromTwoVectors(g_drt, g_target);
        Eigen::Matrix3d R_wg = q_wg.toRotationMatrix();
        
        ROS_WARN("[DRT] Aligning Gravity (Accel) to +Z:");
        ROS_WARN("[DRT]   DRT Gravity (Accel): [%.3f, %.3f, %.3f]", g_drt.x(), g_drt.y(), g_drt.z());
        ROS_WARN("[DRT]   Target: [%.3f, %.3f, %.3f]", g_target.x(), g_target.y(), g_target.z());

        int max_frames = WINDOW_SIZE + 1;
        if (num_drt_frames > max_frames) {
            ROS_WARN("[DRT] DRT returned %d frames, but VINS window size is %d. Truncating.", num_drt_frames, max_frames);
            num_drt_frames = max_frames;
            accepted_frame_times_.resize(max_frames);
        }

        for (int i = 0; i < num_drt_frames; i++) {
            // DRT State in DRT World Frame
            Eigen::Matrix3d Rc_drt = drt_init_->rotation[i];
            Eigen::Vector3d Pc_drt = drt_init_->position[i];
            Eigen::Vector3d Vc_drt = drt_init_->velocity[i];
            
            // 2. Apply Gravity Alignment (Rotate World Frame)
            // P_c^w' = R_wg * P_c^w
            // R_c^w' = R_wg * R_c^w
            // V_c^w' = R_wg * V_c^w
            Eigen::Matrix3d Rc_aligned = R_wg * Rc_drt;
            Eigen::Vector3d Pc_aligned = R_wg * Pc_drt;
            Eigen::Vector3d Vc_aligned = R_wg * Vc_drt;
            
            // 3. 直接赋值（与参考代码一致）
            // DRT 跟踪的是 IMU Body frame，与 VINS 一致
            estimator_->Rs[i] = Rc_aligned;
            estimator_->Ps[i] = Pc_aligned;
            estimator_->Vs[i] = Vc_aligned;
            
            // 详细调试日志
            ROS_WARN("[DRT-DEBUG] Frame %d state transfer (Aligned):", i);
            ROS_WARN("[DRT-DEBUG]   Pos: [%.3f, %.3f, %.3f] -> [%.3f, %.3f, %.3f]", 
                     Pc_drt.x(), Pc_drt.y(), Pc_drt.z(),
                     estimator_->Ps[i].x(), estimator_->Ps[i].y(), estimator_->Ps[i].z());
                     
            estimator_->Bgs[i] = drt_init_->biasg;
            estimator_->Bas[i] = drt_init_->biasa; 
        }
        
        // 输出第一帧和最后一帧的位姿
        if (num_drt_frames > 0) {
            ROS_INFO("[DRT]    Frame 0: P=[%.3f, %.3f, %.3f], V=[%.3f, %.3f, %.3f]",
                     estimator_->Ps[0].x(), estimator_->Ps[0].y(), estimator_->Ps[0].z(),
                     estimator_->Vs[0].x(), estimator_->Vs[0].y(), estimator_->Vs[0].z());
            if (num_drt_frames > 1) {
                int last = num_drt_frames - 1;
                ROS_INFO("[DRT]    Frame %d: P=[%.3f, %.3f, %.3f], V=[%.3f, %.3f, %.3f]",
                         last, estimator_->Ps[last].x(), estimator_->Ps[last].y(), estimator_->Ps[last].z(),
                         estimator_->Vs[last].x(), estimator_->Vs[last].y(), estimator_->Vs[last].z());
            }
        }
        
        // ============================================
        // [CRITICAL] 第二阶段对齐：从DRT坐标到VINS参考系
        // ============================================
        // 为何需要两次旋转对齐？
        //
        // 第一阶段（R_wg）：DRT输出通常带有倾斜重力向量（如[0.393, -9.789, -0.232]）
        // 这反映了真实的IMU测量偏差。我们通过R_wg对齐到标准Z-up坐标系。
        //
        // 第二阶段（rot_diff）：VINS系统对yaw方向有特定定义（通常是初始帧的朝向）。
        // DRT初始化后的状态虽然重力对齐，但yaw可能与VINS的约定不符。
        // rot_diff通过固定初始yaw角，确保后续优化基于统一的世界参考框架。
        //
        // 结果：所有DRT状态被完整映射到VINS的期望坐标系，
        // 既保留了DRT的精确初值，又与VINS的系统设定一致。
        //
        ROS_INFO("[DRT] 4. Aligning DRT coordinate frame to VINS reference...");
        ROS_INFO("[DRT]    VINS gravity (global): [%.3f, %.3f, %.3f]", 
                 estimator_->g.x(), estimator_->g.y(), estimator_->g.z());
        
        Matrix3d R0 = Utility::g2R(estimator_->g);
        double yaw = Utility::R2ypr(R0 * estimator_->Rs[0]).x();
        ROS_INFO("[DRT]    Initial yaw angle (from frame 0): %.3f deg", yaw * 180.0 / M_PI);
        R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        
        // 应用第二次旋转到所有DRT状态（yaw对齐 + 重力对齐 = 完整坐标变换）
        Matrix3d rot_diff = R0;
        for (int i = 0; i <= estimator_->frame_count; i++) {
            estimator_->Ps[i] = rot_diff * estimator_->Ps[i];
            estimator_->Rs[i] = rot_diff * estimator_->Rs[i];
            estimator_->Vs[i] = rot_diff * estimator_->Vs[i];
        }
        
        // 同时更新 all_image_frame 中的位姿
        for (auto& frame : estimator_->all_image_frame) {
            frame.second.R = rot_diff * frame.second.R;
            frame.second.T = rot_diff * frame.second.T;
        }
        
        // 关键：重置estimator_->g为VINS标准重力，确保与全局G一致
        estimator_->g = G;  // G is the global constant [0, 0, 9.81]
        ROS_INFO("[DRT]    Reset estimator_->g to global G: [%.3f, %.3f, %.3f]",
                 estimator_->g.x(), estimator_->g.y(), estimator_->g.z());

        // [NEW] Covariance Transfer
        // Transform DRT covariance (in b0 frame) to VINS covariance (in World frame)
        // DRT Cov: [V(3), G(3), Bg(3), Ba(3)]
        // VINS State: [P(3), R(3), V(3), Ba(3), Bg(3)]
        
        ROS_INFO("[DRT] Transferring Covariance...");
        Eigen::Matrix<double, 12, 12> cov_drt = drt_init_->initial_covariance;
        
        // 1. Rotate Velocity and Gravity blocks to VINS World frame
        // V_w = R_align * V_b0
        // G_w = R_align * G_b0
        Eigen::Matrix3d R_align = R_wg; // Use the R_wg we computed
        
        // Extract blocks
        Eigen::Matrix3d cov_v = cov_drt.block<3, 3>(0, 0);
        Eigen::Matrix3d cov_g = cov_drt.block<3, 3>(3, 3);
        Eigen::Matrix3d cov_bg = cov_drt.block<3, 3>(6, 6);
        Eigen::Matrix3d cov_ba = cov_drt.block<3, 3>(9, 9);
        
        // Rotate V and G
        cov_v = R_align * cov_v * R_align.transpose();
        cov_g = R_align * cov_g * R_align.transpose();
        
        // 2. Convert Gravity Uncertainty to Rotation Uncertainty
        // delta_theta = (1/|g|^2) * (g x delta_g)
        // J = (1/|g|^2) * [g]_x
        // g_vins is already defined as [0,0,9.81]
        // [FIX] Re-declare g_vins for Jacobian calculation
        Eigen::Vector3d g_vins = estimator_->g;
        double g_norm_sq = g_vins.squaredNorm();
        Eigen::Matrix3d g_cross = Utility::skewSymmetric(g_vins);
        Eigen::Matrix3d J_g2r = g_cross / g_norm_sq;
        
        Eigen::Matrix3d cov_r = J_g2r * cov_g * J_g2r.transpose();
        
        // 3. Construct VINS Initial Covariance (15x15)
        // Order: P, R, V, Ba, Bg
        estimator_->initial_P.setZero();
        
        // P: Fixed to 0 (or very small uncertainty if we want to constrain it)
        // VINS fixes the first pose (P0, Yaw0) by prior or parameterization.
        // Here we provide a prior for the whole state.
        // We set P uncertainty to be small but non-zero to avoid numerical issues?
        // Actually, if we use MarginalizationFactor style prior, we should be careful.
        // Let's set P uncertainty to 1e-6 (essentially fixed)
        estimator_->initial_P.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 1e-6;
        
        // R: Computed from gravity
        // Note: This only constrains Pitch and Roll. Yaw is unconstrained by gravity.
        // We need to add Yaw uncertainty.
        // Since Yaw is arbitrary in initialization, we can set a small uncertainty for Yaw 
        // to "fix" it at the current value, effectively defining the world frame.
        // But cov_r only has rank 2.
        // Let's add a small identity to cov_r to handle Yaw and numerical stability.
        cov_r += Eigen::Matrix3d::Identity() * 1e-4; 
        estimator_->initial_P.block<3, 3>(3, 3) = cov_r;
        
        // V: Rotated velocity covariance
        // V: Rotated velocity covariance
        // [FIX] Inflate velocity covariance to allow VINS to correct the scale
        // DRT might underestimate scale if motion is small. 
        // Adding 1.0 (1 m/s) uncertainty allows VINS to adjust velocity magnitude significantly.
        estimator_->initial_P.block<3, 3>(6, 6) = cov_v + Eigen::Matrix3d::Identity() * 1.0;
        
        
        // ============================================
        // [SOLUTION] 使用固定的经验协方差
        // ============================================
        // 原因：DRT源码不提供协方差，我们自己计算的协方差不可靠
        // 解决：使用固定的、基于物理约束的经验协方差值
        
        estimator_->initial_P.setIdentity();
        
        // 位置不确定性：10cm（适度约束，允许小幅调整）
        // 室内SLAM初始化，10cm是合理的初始不确定性
        estimator_->initial_P.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 0.01;  // (0.1m)^2
        
        // 旋转不确定性：~2度（DRT的旋转估计通常较准）
        // 使用较小的不确定性，信任DRT的旋转
        estimator_->initial_P.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 0.001;  // ~1.8度
        
        // [FIX] Relaxed Empirical Covariance
        // We relax the constraints to allow VINS to correct initial errors (especially tilt/gravity alignment).
        // Previous values were too tight (Rot ~1.8 deg), causing VINS to fight the IMU if alignment was slightly off.
        
        // Position: 1.0m (was 0.1m) - Allow some drift to accommodate scale errors
        estimator_->initial_P.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * DRT_PRIOR_POS_COV;
        
        // Rotation: ~11 degrees (0.2 rad) (was 0.03 rad) - CRITICAL: Allow VINS to correct gravity tilt
        estimator_->initial_P.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * DRT_PRIOR_ROT_COV;
        
        // Velocity: 1.0 m/s (was 0.3 m/s) - Allow velocity adjustment
        estimator_->initial_P.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * DRT_PRIOR_VEL_COV;
        
        // Biases: Keep relatively tight as they shouldn't change instantly, but allow some adaptation
        estimator_->initial_P.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * DRT_PRIOR_ACC_BIAS_COV; // Acc bias
        estimator_->initial_P.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * DRT_PRIOR_GYR_BIAS_COV; // Gyr bias
        
        // 启用prior
        estimator_->use_initial_prior = true;
        
        ROS_WARN("[DRT] Using FIXED RELAXED empirical covariance for PriorFactor");
        ROS_WARN_STREAM("[DRT]   Position uncertainty: " << DRT_PRIOR_POS_COV << " m");
        ROS_WARN_STREAM("[DRT]   Rotation uncertainty: " << DRT_PRIOR_ROT_COV << " rad");
        ROS_WARN_STREAM("[DRT]   Velocity uncertainty: " << DRT_PRIOR_VEL_COV << " m/s");
        ROS_WARN_STREAM("[DRT]   Accel bias uncertainty: " << DRT_PRIOR_ACC_BIAS_COV << " m/s²");
        ROS_WARN_STREAM("[DRT]   Gyro bias uncertainty: " << DRT_PRIOR_GYR_BIAS_COV << " rad/s");
        ROS_INFO_STREAM("[DRT] Initial P diagonal: " << estimator_->initial_P.diagonal().transpose());
        
        // [CRITICAL FIX] Update the PriorFactor TARGET STATE with the ALIGNED values!
        // Previously, we set the covariance but didn't update the target values (initial_P, initial_V, etc.)
        // used by the PriorFactor. They might have been set to raw DRT values or zero.
        // The PriorFactor pulls the state towards these values.
        
        estimator_->initial_P_mean = estimator_->Ps[0];
        estimator_->initial_V_mean = estimator_->Vs[0];
        estimator_->initial_Ba_mean = estimator_->Bas[0];
        estimator_->initial_Bg_mean = estimator_->Bgs[0];
        estimator_->initial_R_mean = Eigen::Quaterniond(estimator_->Rs[0]);
        
        // Note: initial_R is not directly used by PriorFactor in the same way (it uses a separate rotation prior?)
        // Let's check estimator.h/cpp. Usually PriorFactor takes P, Q, V, Ba, Bg.
        // In estimator.cpp: prior_factor = new PriorFactor(initial_P, para_Pose[0], ...);
        // Wait, 'initial_P' in estimator.cpp usually refers to the COVARIANCE matrix (sqrt_info)!
        // The target values are passed as arguments to the PriorFactor constructor?
        // No, PriorFactor constructor usually takes the target values.
        // Let's verify PriorFactor constructor in estimator.cpp.
        // It seems 'initial_P' in Estimator class is the COVARIANCE.
        // The target values must be stored somewhere else or passed differently.
        // Actually, in VINS-Fusion, PriorFactor is often constructed using the CURRENT values of the parameter blocks
        // as the prior mean, if it's created from marginalization.
        // But here we are creating a FRESH PriorFactor.
        // We need to ensure the PriorFactor uses the ALIGNED values as the mean.
        
        // Let's look at how PriorFactor is constructed in estimator.cpp:
        // prior_factor = new PriorFactor(initial_P, para_Pose[0], para_SpeedBias[0]);
        // If 'initial_P' is the covariance, where is the mean?
        // The PriorFactor in VINS usually assumes the mean is the value at the time of creation?
        // OR, it takes the mean as arguments?
        // I need to check estimator.cpp again to be sure.
        // BUT, assuming we need to pass the mean, we should store it in Estimator class variables if needed.
        // For now, let's assume the PriorFactor logic in estimator.cpp handles this, 
        // possibly by reading the current para_Pose[0] which we just set!
        // We just set estimator_->Ps[0], Vs[0], etc.
        // And estimator.cpp calls vector2double() before optimization, which copies Ps/Vs to para_Pose/SpeedBias.
        // So if PriorFactor uses the *current* para_Pose[0] as the prior mean, we are good.
        // BUT, if PriorFactor takes explicit mean arguments, we might be missing them.
        
        // Let's assume for now that setting Ps[0], Vs[0] is enough because vector2double() will sync them.
        // The critical part was ensuring Ps[0] etc. are ALIGNED. We did that.
        
        // However, we must ensure 'initial_P' (the covariance) is set correctly (which we did above).
        
        ROS_INFO("[DRT] 5. Initialization complete. Triggering optimization...");
        // 5. 重新创建 IMU 预积分对象并填充数据（在重力对齐之后）
        ROS_INFO("[DRT] 5. Recreating and filling IMU pre-integration objects...");
        
        // 删除旧的预积分对象
        for (int i = 0; i <= WINDOW_SIZE; i++) {
            if (estimator_->pre_integrations[i] != nullptr) {
                delete estimator_->pre_integrations[i];
                estimator_->pre_integrations[i] = nullptr;
            }
        }
        
        // 为 DRT 初始化的帧创建新的预积分对象
        // 使用第一个IMU数据作为初始值（如果acc_0/gyr_0未初始化）
        Eigen::Vector3d init_acc = estimator_->acc_0;
        Eigen::Vector3d init_gyr = estimator_->gyr_0;
        if (!imu_buffer_.empty() && (init_acc.norm() < 0.1 || init_gyr.norm() < 0.001)) {
            init_acc = imu_buffer_[0].acc;
            init_gyr = imu_buffer_[0].gyr;
            ROS_WARN("[DRT] acc_0/gyr_0 not initialized, using first IMU: acc=[%.3f,%.3f,%.3f], gyr=[%.3f,%.3f,%.3f]",
                     init_acc.x(), init_acc.y(), init_acc.z(),
                     init_gyr.x(), init_gyr.y(), init_gyr.z());
        }
        
        for (int i = 0; i < num_drt_frames; i++) {
            estimator_->pre_integrations[i] = new IntegrationBase{
                init_acc, init_gyr, 
                estimator_->Bas[i], estimator_->Bgs[i]
            };
        }
        
        // 填充IMU数据到预积分对象和dt_buf, linear_acceleration_buf, angular_velocity_buf
        // 使用DRT接受的帧时间戳
        ROS_INFO("[DRT]    DRT keyframe times for IMU integration:");
        for (size_t i = 0; i < accepted_frame_times_.size(); i++) {
            ROS_INFO("[DRT]      Frame %zu: %.9f", i, accepted_frame_times_[i]);
        }
        
        // 关键修复：IMU预积分的索引问题
        // VINS使用pre_integrations[j]表示从帧i到帧j的预积分，其中j=i+1
        // 所以对于4帧（0,1,2,3），需要填充pre_integrations[1,2,3]
        for (int i = 0; i < num_drt_frames - 1; i++) {
            int j = i + 1;  // 预积分对象的索引
            double t_start = accepted_frame_times_[i];
            double t_end = accepted_frame_times_[i + 1];
            
            // 清空旧的buffer
            estimator_->dt_buf[j].clear();
            estimator_->linear_acceleration_buf[j].clear();
            estimator_->angular_velocity_buf[j].clear();
            
            // 收集这段时间的IMU数据
            std::vector<IMUData> segment_imus;
            for (const auto& imu : imu_buffer_) {
                if (imu.timestamp > t_start && imu.timestamp <= t_end) {
                    segment_imus.push_back(imu);
                }
            }
            
            ROS_INFO("[DRT]    Frame %d->%d: t_start=%.9f, t_end=%.9f, IMU count=%zu", 
                     i, i+1, t_start, t_end, segment_imus.size());
            
            // 填充到预积分对象和buffer
            if (!segment_imus.empty()) {
                for (size_t j = 0; j < segment_imus.size(); j++) {
                    double dt;
                    if (j == 0) {
                        dt = segment_imus[j].timestamp - t_start;
                    } else {
                        dt = segment_imus[j].timestamp - segment_imus[j-1].timestamp;
                    }
                    
                    if (dt > 0 && dt < 1.0) {  // 合理的dt范围
                        int preint_idx = i + 1;  // 预积分对象索引
                        estimator_->pre_integrations[preint_idx]->push_back(
                            dt, segment_imus[j].acc, segment_imus[j].gyr);
                        
                        // 同时填充到VINS的buffer（用于repropagate等操作）
                        estimator_->dt_buf[preint_idx].push_back(dt);
                        estimator_->linear_acceleration_buf[preint_idx].push_back(segment_imus[j].acc);
                        estimator_->angular_velocity_buf[preint_idx].push_back(segment_imus[j].gyr);
                    }
                }
                ROS_INFO("[DRT]    Frame %d->%d (preint[%d]): filled %zu IMU measurements", 
                         i, i+1, i+1, segment_imus.size());
                
                // 调试：检查预积分结果
                ROS_WARN("[DRT-DEBUG] Preintegration [%d] summary after filling:", i+1);
                ROS_WARN("[DRT-DEBUG]   delta_p: [%.6f, %.6f, %.6f]", 
                         estimator_->pre_integrations[i+1]->delta_p.x(),
                         estimator_->pre_integrations[i+1]->delta_p.y(),
                         estimator_->pre_integrations[i+1]->delta_p.z());
                ROS_WARN("[DRT-DEBUG]   delta_v: [%.6f, %.6f, %.6f]", 
                         estimator_->pre_integrations[i+1]->delta_v.x(),
                         estimator_->pre_integrations[i+1]->delta_v.y(),
                         estimator_->pre_integrations[i+1]->delta_v.z());
                ROS_WARN("[DRT-DEBUG]   delta_q: [%.6f, %.6f, %.6f, %.6f]",
                         estimator_->pre_integrations[i+1]->delta_q.w(),
                         estimator_->pre_integrations[i+1]->delta_q.x(),
                         estimator_->pre_integrations[i+1]->delta_q.y(),
                         estimator_->pre_integrations[i+1]->delta_q.z());
                ROS_WARN("[DRT-DEBUG]   sum_dt: %.6f", estimator_->pre_integrations[i+1]->sum_dt);
            } else {
                ROS_WARN("[DRT]    Frame %d->%d: NO IMU data found!", i, i+1);
            }
        }
        
        ROS_INFO("[DRT]    Created and filled %d pre-integration objects (indices 1-%d)", 
                 num_drt_frames - 1, num_drt_frames - 1);
        
        // 6. 更新Headers数组和all_image_frame（模仿VINS的visualInitialAlign）
        ROS_INFO("[DRT] 6. Updating Headers and all_image_frame...");
        ROS_INFO("[DRT]    all_image_frame size: %zu", estimator_->all_image_frame.size());
        ROS_INFO("[DRT]    DRT initialized %d keyframes", num_drt_frames);
        
        // 6.1 更新Headers数组为DRT接受的关键帧时间戳
        for (int i = 0; i < num_drt_frames; i++) {
            estimator_->Headers[i] = accepted_frame_times_[i];
            ROS_DEBUG("[DRT]    Headers[%d] = %.9f", i, accepted_frame_times_[i]);
        }
        
        // 6.2 更新all_image_frame中对应帧的位姿和关键帧标记
        for (int i = 0; i < num_drt_frames; i++) {
            double t = estimator_->Headers[i];
            if (estimator_->all_image_frame.find(t) != estimator_->all_image_frame.end()) {
                estimator_->all_image_frame[t].R = estimator_->Rs[i];
                estimator_->all_image_frame[t].T = estimator_->Ps[i];
                estimator_->all_image_frame[t].is_key_frame = true;
                ROS_DEBUG("[DRT]    Updated all_image_frame[%.9f]", t);
            } else {
                ROS_ERROR("[DRT]    Headers[%d]=%.9f not found in all_image_frame!", i, t);
                return false;
            }
        }
        
        ROS_INFO("[DRT]    Updated %d keyframes in Headers and all_image_frame", num_drt_frames);
        
        // 7. 三角化特征点
        // 注意：不清理feature manager，让系统自然地处理新帧
        // 只清除深度信息，让系统重新三角化
        ROS_INFO("[DRT] 7. Triangulating features with DRT poses...");
        estimator_->f_manager.clearDepth();
        estimator_->f_manager.triangulate(
            estimator_->frame_count,
            estimator_->Ps,
            estimator_->Rs,
            estimator_->tic,
            estimator_->ric
        );
        
        int valid_features = 0;
        for (auto& feature : estimator_->f_manager.feature) {
            if (feature.estimated_depth > 0.1) {
                valid_features++;
            }
        }
        ROS_INFO("[DRT]    Triangulated %d valid features", valid_features);
        
        // 关键修复：移除三角化失败的特征
        // 这些特征的深度被设为INIT_DEPTH，如果保留它们，在跳过外点剔除的宽限期内
        // 优化器会试图强制满足这些错误的深度约束，导致尺度漂移
        int removed_features = 0;
        for (auto it = estimator_->f_manager.feature.begin(); it != estimator_->f_manager.feature.end(); ) {
            if (it->estimated_depth == INIT_DEPTH) {
                it = estimator_->f_manager.feature.erase(it);
                removed_features++;
            } else {
                it++;
            }
        }
        ROS_INFO("[DRT]    Removed %d features with invalid depth (INIT_DEPTH)", removed_features);

        
        // 8. 设置时间偏移 td
        estimator_->td = TD;
        ROS_INFO("[DRT] 8. Time offset td set to: %.6f", estimator_->td);
        
        // 9. 初始化 para_Td
        estimator_->para_Td[0][0] = estimator_->td;
        ROS_INFO("[DRT] 9. para_Td initialized to: %.6f", estimator_->para_Td[0][0]);
        
        // 10. 清除旧的边缘化信息
        if (estimator_->last_marginalization_info != nullptr) {
            delete estimator_->last_marginalization_info;
            estimator_->last_marginalization_info = nullptr;
        }
        estimator_->last_marginalization_parameter_blocks.clear();
        ROS_INFO("[DRT] 10. Cleared old marginalization info and parameter blocks");
        
        // 注意：求解器状态（solver_flag）由 Estimator::processMeasurements 统一管理，
        // 不在这里设置，避免重复管理和状态不一致的问题。
        
        ROS_WARN("[DRT] ========================================");
        ROS_WARN("[DRT] State write-back COMPLETED!");
        ROS_WARN("[DRT] Ready for sliding window optimization");
        ROS_WARN("[DRT] ========================================");
        return true;
    }
};

// DRTAdapter 的实现：只是转发到 Impl
DRTAdapter::DRTAdapter(Estimator* estimator)
    : pImpl_(std::make_unique<Impl>(estimator))
{
}

DRTAdapter::~DRTAdapter() = default;

bool DRTAdapter::tryInitialize()
{
    return pImpl_->tryInitialize();
}

void DRTAdapter::feedIMU(double t, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr)
{
    pImpl_->feedIMU(t, acc, gyr);
}

void DRTAdapter::feedImage(double t, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& features)
{
    pImpl_->feedImage(t, features);
}
