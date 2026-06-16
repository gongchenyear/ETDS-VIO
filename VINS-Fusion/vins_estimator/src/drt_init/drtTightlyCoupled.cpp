//
// Created by xubo on 23-8-30.
//
// 使用本地 DRT 头文件
#include "include/drtTightlyCoupled.h"

namespace DRT
{

    drtTightlyCoupled::drtTightlyCoupled(const Eigen::Matrix3d &Rbc, const Eigen::Vector3d &pbc)
    : drtVioInit(Rbc, pbc) {}

    bool drtTightlyCoupled::process()
    {

        cout << "drt tightly process" << endl;
        ticToc t_solve;
        rotation.resize(local_active_frames.size());
        position.resize(local_active_frames.size());
        velocity.resize(local_active_frames.size());

        ticToc t_biasg;

        if(!gyroBiasEstimator())
            return false;

        double time_biasg = t_biasg.toc();

        std::cout << "biasa: " << biasa.transpose() << std::endl;

        ticToc t_velocity_gravity;
        vio::IMUBias solved_bias(biasg, biasa);

        // the reference frame is b0
        rotation[0] = Eigen::Matrix3d::Identity();
        frame_rot.clear();
        frame_rot[int_frameid2_time_frameid.at(0)] = Rbc_.transpose() * rotation[0] * Rbc_;

        std::vector<vio::IMUPreintegrated::integrable> accum_imu_meas;
        for (int i = 0; i < imu_meas.size(); i++) {
            accum_imu_meas.insert(accum_imu_meas.end(), imu_meas[i].imu_measurements_.begin(),
                                  imu_meas[i].imu_measurements_.end());
            imu_meas[i].reintegrate(solved_bias, accum_imu_meas);
            rotation[i + 1] = imu_meas[i].dR_.matrix();
            frame_rot[int_frameid2_time_frameid.at(i + 1)] = Rbc_.transpose() * rotation[i + 1] * Rbc_;
        }

        int num_obs = 0;
        for (const auto &pt: SFMConstruct) {
            if (pt.second.obs.size() < 3) continue;
            num_obs += (int) pt.second.obs.size() - 1;
        }

        Eigen::MatrixXd A_H{6, 6};
        Eigen::MatrixXd b_H{6, 1};
        A_H.setZero();
        b_H.setZero();
        double Q_H = 0.;

        num_obs = 0;
        for (const auto &pt: SFMConstruct) {

            const auto &obs = pt.second.obs;

            // the number of obs must greater than 3
            if (obs.size() < 3) continue;


            TimeFrameId lbase_view_id = 0;
            TimeFrameId rbase_view_id = 0;

            // select two frame with max parallax
            select_base_views(obs, lbase_view_id, rbase_view_id);

            for (const auto &frame: obs) {

                TimeFrameId i_view_id = frame.first;

                if (i_view_id != rbase_view_id) {

                    Eigen::Matrix3d xi_cross = cross_product_matrix(frame.second.normalpoint);

                    Eigen::Matrix3d R_cicl =
                            frame_rot.at(i_view_id).transpose() * frame_rot.at(lbase_view_id);

                    Eigen::Matrix3d R_crcl =
                            frame_rot.at(rbase_view_id).transpose() * frame_rot.at(lbase_view_id);

                    Eigen::Vector3d a_lr_tmp_t =
                            cross_product_matrix(R_crcl * obs.at(lbase_view_id).normalpoint) *
                            obs.at(rbase_view_id).normalpoint;

                    Eigen::RowVector3d a_lr_t =
                            a_lr_tmp_t.transpose() * cross_product_matrix(obs.at(rbase_view_id).normalpoint);

                    Eigen::Vector3d theta_lr_vector = cross_product_matrix(obs.at(rbase_view_id).normalpoint)
                                                      * R_crcl
                                                      * obs.at(lbase_view_id).normalpoint;

                    double theta_lr = theta_lr_vector.squaredNorm();

                    Eigen::Matrix3d B =
                            xi_cross * R_cicl * obs.at(lbase_view_id).normalpoint * a_lr_t *
                            frame_rot.at(rbase_view_id).transpose();

                    Eigen::Matrix3d B_prime = B * Rbc_.transpose();


                    Eigen::Matrix3d C = theta_lr * cross_product_matrix(obs.at(i_view_id).normalpoint) *
                                        frame_rot.at(i_view_id).transpose();

                    Eigen::Matrix3d C_prime = C * Rbc_.transpose();

                    Eigen::Matrix3d D = -(B + C);
                    Eigen::Matrix3d D_prime = D * Rbc_.transpose();

                    Eigen::Vector3d S_1r, S_1i, S_1l;
                    double t_1r, t_1i, t_1l;

                    if (time_frameid2_int_frameid.at(lbase_view_id) > 0) {
                        S_1l = imu_meas[time_frameid2_int_frameid.at(lbase_view_id) - 1].dP_ +
                               rotation[time_frameid2_int_frameid.at(lbase_view_id)] * pbc_ - pbc_;

                        t_1l = imu_meas[time_frameid2_int_frameid.at(lbase_view_id) - 1].sum_dt_;
                    } else {
                        S_1l = Eigen::Vector3d::Zero();
                        t_1l = 0;
                    }

                    if (time_frameid2_int_frameid.at(i_view_id) > 0) {
                        S_1i = imu_meas[time_frameid2_int_frameid.at(i_view_id) - 1].dP_ +
                               rotation[time_frameid2_int_frameid.at(i_view_id)] * pbc_ - pbc_;

                        t_1i = imu_meas[time_frameid2_int_frameid.at(i_view_id) - 1].sum_dt_;
                    } else {
                        S_1i = Eigen::Vector3d::Zero();
                        t_1i = 0;
                    }

                    if (time_frameid2_int_frameid.at(rbase_view_id) > 0) {
                        S_1r = imu_meas[time_frameid2_int_frameid.at(rbase_view_id) - 1].dP_ +
                               rotation[time_frameid2_int_frameid.at(rbase_view_id)] * pbc_ - pbc_;
                        t_1r = imu_meas[time_frameid2_int_frameid.at(rbase_view_id) - 1].sum_dt_;
                    } else {
                        S_1r = Eigen::Vector3d::Zero();
                        t_1r = 0;
                    }

                    Eigen::Matrix<double, 3, 6> A_tmp;
                    A_tmp.block(0, 0, 3, 3) = (B_prime * t_1r + C_prime * t_1i + D_prime * t_1l);
                    A_tmp.block(0, 3, 3, 3) =
                            -(B_prime * t_1r * t_1r / 2. + C_prime * t_1i * t_1i / 2. + D_prime * t_1l * t_1l / 2.) *
                            gravity_magnitude;
                    Eigen::Vector3d b_tmp = (-B_prime * S_1r - C_prime * S_1i - D_prime * S_1l);

                    A_H += A_tmp.transpose() * A_tmp;
                    b_H += A_tmp.transpose() * b_tmp;
                    Q_H += b_tmp.transpose() * b_tmp;

                    num_obs++;
                }
            }
        }

        Eigen::MatrixXd A2TA2 = A_H.block(3, 3, 3, 3);

        double mean_value = (A2TA2(0, 0) + A2TA2(1, 1) + A2TA2(2, 2)) / 3.0;

        double scale = 1. / mean_value;

        A_H = A_H * scale;
        b_H = b_H * scale;
        Q_H = Q_H * scale;

        Eigen::VectorXd rhs;
        if (!gravityRefine(A_H, -2. * b_H, Q_H, 1, rhs))
        {
            return false;
        }

        gravity = rhs.tail(3).normalized() * gravity_magnitude;
        velocity[0] = rhs.head(3);
        position[0].setZero();
        for (int i = 0; i < int_frameid2_time_frameid.size() - 1; i++) {
            int j = i + 1;
            double dt = imu_meas[i].sum_dt_;
            Eigen::Vector3d dP = imu_meas[i].dP_;
            Eigen::Vector3d dV = imu_meas[i].dV_;
            velocity[j] = velocity[0] - gravity * dt + dV;
            position[j] = velocity[0] * dt - 0.5 * gravity * dt * dt + dP;

        }

        // [NEW] Compute Covariance
        // A_H is the Hessian (J^T J). Covariance is inverse of Hessian.
        // Note: A_H was scaled by 'scale'. We need to handle this.
        // H_scaled = scale * H_real
        // Cov_real = H_real^-1 = (H_scaled / scale)^-1 = scale * H_scaled^-1
        
        // Initialize covariance to identity * large value (high uncertainty)
        initial_covariance.setIdentity();
        initial_covariance *= 100.0; 
        
        // 1. Velocity and Gravity Covariance (from Linear Solver)
        Eigen::MatrixXd Cov_vg = A_H.inverse() * scale; // 6x6
        
        // [FIX] Cap covariance to reasonable physical limits
        // If Hessian is poorly conditioned (few observations, small parallax),
        // the scale factor can be huge (e.g. 1e8), resulting in unrealistic covariance.
        // This makes the prior useless in VINS optimization.
        // Physical limits:
        // - Velocity uncertainty: max 10 m/s (generous for initialization)
        // - Gravity uncertainty: max 1 m/s^2 (corresponds to ~5 deg tilt uncertainty)
        const double MAX_V_COV = 10.0 * 10.0;  // (10 m/s)^2
        const double MAX_G_COV = 1.0 * 1.0;    // (1 m/s^2)^2
        
        // Check if inversion was successful (not NaN/Inf)
        if (Cov_vg.allFinite()) {
            // Cap velocity covariance (first 3x3 block)
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (std::abs(Cov_vg(i, j)) > MAX_V_COV) {
                        Cov_vg(i, j) = (Cov_vg(i, j) > 0 ? MAX_V_COV : -MAX_V_COV);
                    }
                }
            }
            // Cap gravity covariance (second 3x3 block)
            for (int i = 3; i < 6; i++) {
                for (int j = 3; j < 6; j++) {
                    if (std::abs(Cov_vg(i, j)) > MAX_G_COV) {
                        Cov_vg(i, j) = (Cov_vg(i, j) > 0 ? MAX_G_COV : -MAX_G_COV);
                    }
                }
            }
            
            // Map to [0-2] Velocity, [3-5] Gravity
            initial_covariance.block<6, 6>(0, 0) = Cov_vg;
            std::cout << "[DRT] Velocity-Gravity Covariance computed (capped to physical limits)." << std::endl;
            // std::cout << Cov_vg << std::endl;
        } else {
            std::cout << "[DRT] WARNING: Hessian inversion failed or singular. Using default high covariance." << std::endl;
        }

        // 2. Gyro Bias Covariance
        // For now, we set a fixed reasonable covariance for biasg as we didn't extract it from Ceres
        // Typical IMU gyro bias random walk is around 2.0e-5 to 1.0e-4
        // Initial uncertainty can be larger.
        initial_covariance.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 1e-4;

        // 3. Accel Bias Covariance
        // We didn't estimate biasa, so set a large uncertainty
        initial_covariance.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 0.1; // 0.1 is large for bias

        return true;
    }

    void drtTightlyCoupled::select_base_views(const Eigen::aligned_map<TimeFrameId, FeaturePerFrame> &track,
                                              TimeFrameId &lbase_view_id,
                                              TimeFrameId &rbase_view_id) {
        double best_criterion_value = -1.;
        std::vector<int> track_id;
        // track_id.reserve(track.size());

        for (const auto &frame: track) {
            int id = time_frameid2_int_frameid.at(frame.first);
            track_id.push_back(id);
        }

        size_t track_size = track_id.size(); //num_pts_

        // [Step.2 in Pose-only Algorithm]: select the left/right-base views
        for (int i = 0; i < track_size - 1; ++i) {
            for (int j = i + 1; j < track_size; ++j) {

                const TimeFrameId &i_view_id = int_frameid2_time_frameid.at(track_id[i]);
                const TimeFrameId &j_view_id = int_frameid2_time_frameid.at(track_id[j]);

                const Eigen::Vector3d &i_coord = track.at(i_view_id).normalpoint;
                const Eigen::Vector3d &j_coord = track.at(j_view_id).normalpoint;

                // R_i is world to camera i
                const Eigen::Matrix3d &R_i = frame_rot.at(i_view_id);
                const Eigen::Matrix3d &R_j = frame_rot.at(j_view_id);
                // camera i to camera j
                // Rcjw *  Rwci
                const Eigen::Matrix3d R_ij = R_j.transpose() * R_i;
                const Eigen::Vector3d theta_ij = j_coord.cross(R_ij * i_coord);

                double criterion_value = theta_ij.norm();

                if (criterion_value > best_criterion_value) {

                    best_criterion_value = criterion_value;

                    if (i_view_id < j_view_id) {
                        lbase_view_id = i_view_id;
                        rbase_view_id = j_view_id;
                    } else {
                        lbase_view_id = j_view_id;
                        rbase_view_id = i_view_id;
                    }

                }
            }
        }
    }

}