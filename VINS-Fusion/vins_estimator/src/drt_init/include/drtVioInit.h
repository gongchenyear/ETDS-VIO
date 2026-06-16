//
// Created by xubo on 22-8-21.
//

#ifndef VIO_INIT_SYS_ROBUST_INITIALIZE_VIO_H
#define VIO_INIT_SYS_ROBUST_INITIALIZE_VIO_H

#include <string>
#include <unordered_map>
// #include <glog/logging.h>
// #include <pangolin/pangolin.h>

// 简单的日志宏替代
#include <iostream>
#define LOG(level) std::cout

// 使用本地 DRT 头文件
#include "utils/eigenTypes.h"
#include "utils/ticToc.h"
#include "IMU/basicTypes.hpp"
#include "IMU/imuPreintegrated.hpp"
#include "featureManager.h"
#include "optimization.hpp"
// 不再包含 featureTracker/parameters.h，避免与 VINS 冲突

namespace DRT {

    using namespace Eigen;
    using namespace std;

    // [MODIFIED] Removed static const G to allow configurable gravity
    // static const Eigen::Vector3d G(0.0, 0.0, 9.8);

    class drtVioInit {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        drtVioInit(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &tbc);

        virtual ~drtVioInit() = default;

        virtual bool process() = 0;

        bool gravityRefine(const Eigen::MatrixXd &M,
                           const Eigen::VectorXd &m,
                           double Q,
                           double gravity_mag,
                           Eigen::VectorXd &rhs);


        // 用于标定gyr bias
        bool gyroBiasEstimator();

        bool addFeatureCheckParallax(TimeFrameId frame_id, const FeatureTrackerResulst &image, double td);


        double compensatedParallax2(const Eigen::Vector3d &p_i, const Eigen::Vector3d &p_j);

        void addImuMeasure(const vio::IMUPreintegrated &imuData);

        void recomputeFrameId();

        bool checkAccError();

        inline Eigen::Matrix3d cross_product_matrix(const Eigen::Vector3d &x) {
            Eigen::Matrix3d X;
            X << 0, -x(2), x(1),
                    x(2), 0, -x(0),
                    -x(1), x(0), 0;
            return X;
        }

        using Ptr = std::shared_ptr<drtVioInit>;
    public:

        std::set<double> local_active_frames;
        std::map<int, TimeFrameId> int_frameid2_time_frameid;
        std::map<TimeFrameId, int> time_frameid2_int_frameid;
        Eigen::aligned_map<TimeFrameId, Eigen::Matrix3d> frame_rot;

        Eigen::Vector3d biasg;
        Eigen::Vector3d biasa;
        Eigen::Vector3d gravity;
        // [NEW] Covariance matrix for the initialized state
        // Structure: 
        // [0-2]: Velocity (v0)
        // [3-5]: Gravity (g)
        // [6-8]: Gyro Bias (bg)
        // [9-11]: Accel Bias (ba)
        Eigen::Matrix<double, 12, 12> initial_covariance;
        
        double avg_observation;
        std::vector<Eigen::Vector3d> velocity;
        std::vector<Eigen::Vector3d> position;
        std::vector<Eigen::Matrix3d> rotation;

        Eigen::Matrix3d Rbc_;
        Eigen::Vector3d pbc_;

        Eigen::aligned_unordered_map<FeatureID, SFMFeature>
                SFMConstruct;
        Eigen::aligned_vector<vio::IMUPreintegrated> imu_meas;

        TimeFrameId last_image_t_ns;
        
        // [NEW] Configurable gravity magnitude (default 9.81)
        double gravity_magnitude = 9.81;

        int frame_num_;
    };
}

#endif //VIO_INIT_SYS_ROBUST_INITIALIZE_VIO_H
