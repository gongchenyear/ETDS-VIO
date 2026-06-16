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

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <limits>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/image_encodings.h>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"

Estimator estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
queue<sensor_msgs::ImageConstPtr> depth_buf;
std::mutex m_buf;


// 全局变量：追踪最新的RGB时间戳
static double g_last_rgb_timestamp = 0;
static std::mutex g_rgb_timestamp_mutex;

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
    
    // 🔍 OpenLORIS诊断：更新最新RGB时间戳
    {
        std::lock_guard<std::mutex> lock(g_rgb_timestamp_mutex);
        g_last_rgb_timestamp = img_msg->header.stamp.toSec();
    }
    
    static bool first_rgb_logged = false;
    if(!first_rgb_logged) {
        ROS_WARN("🎉 [RGBDiag] First RGB frame: timestamp=%.6f, frame_id=%s", 
                 img_msg->header.stamp.toSec(), img_msg->header.frame_id.c_str());
        first_rgb_logged = true;
    }
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}

void img0_compressed_callback(const sensor_msgs::CompressedImageConstPtr &img_msg)
{
    // 将压缩图像转换为普通图像后再放入队列
    cv_bridge::CvImageConstPtr ptr;
    try
    {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        
        // 创建一个普通的Image消息以维持代码兼容性
        sensor_msgs::ImagePtr image_ptr = boost::make_shared<sensor_msgs::Image>();
        image_ptr->header = img_msg->header;
        ptr->toImageMsg(*image_ptr);
        
        m_buf.lock();
        img0_buf.push(image_ptr);
        m_buf.unlock();
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception in compressed callback: %s", e.what());
    }
}

void img1_compressed_callback(const sensor_msgs::CompressedImageConstPtr &img_msg)
{
    // 将压缩图像转换为普通图像后再放入队列
    cv_bridge::CvImageConstPtr ptr;
    try
    {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        
        // 创建一个普通的Image消息以维持代码兼容性
        sensor_msgs::ImagePtr image_ptr = boost::make_shared<sensor_msgs::Image>();
        image_ptr->header = img_msg->header;
        ptr->toImageMsg(*image_ptr);
        
        m_buf.lock();
        img1_buf.push(image_ptr);
        m_buf.unlock();
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception in compressed callback: %s", e.what());
    }
}

void depth_callback(const sensor_msgs::ImageConstPtr &depth_msg)
{
    static bool first_depth_received = false;
    static int depth_frame_count = 0;
    static double last_log_time = 0;
    static double first_depth_timestamp = 0;
    
    m_buf.lock();
    
    // 🔍 OpenLORIS诊断：首帧深度图信息
    if (!first_depth_received) {
        first_depth_timestamp = depth_msg->header.stamp.toSec();
        ROS_WARN("🎉 [DepthDiag] ✓✓✓ FIRST DEPTH FRAME RECEIVED ✓✓✓");
        ROS_WARN("[DepthDiag]   Topic: %s", depth_msg->header.frame_id.c_str());
        ROS_WARN("[DepthDiag]   Size: %dx%d", depth_msg->width, depth_msg->height);
        ROS_WARN("[DepthDiag]   Encoding: %s", depth_msg->encoding.c_str());
        ROS_WARN("[DepthDiag]   Timestamp: %.6f (bag time)", first_depth_timestamp);
        ROS_WARN("[DepthDiag]   ROS wall time: %.6f", ros::Time::now().toSec());
        first_depth_received = true;
    }
    
    depth_frame_count++;
    
    depth_buf.push(depth_msg);
    while(depth_buf.size() > 50) depth_buf.pop();
    
    double current_time = ros::Time::now().toSec();
    if(current_time - last_log_time > 5.0) {
        double depth_span = depth_buf.back()->header.stamp.toSec() - depth_buf.front()->header.stamp.toSec();
        ROS_WARN("[DepthDiag] Frames: %d, buffer: %zu, depth_span: %.3fs, oldest: %.3f, newest: %.3f", 
                 depth_frame_count, depth_buf.size(), depth_span,
                 depth_buf.front()->header.stamp.toSec(),
                 depth_buf.back()->header.stamp.toSec());
        
        // 🔍 关键诊断：深度与RGB时间戳差异
        double last_rgb_ts = 0;
        {
            std::lock_guard<std::mutex> lock(g_rgb_timestamp_mutex);
            last_rgb_ts = g_last_rgb_timestamp;
        }
        
        if(last_rgb_ts > 0) {
            double time_diff = depth_msg->header.stamp.toSec() - last_rgb_ts;
            ROS_WARN("[DepthDiag] ⏱️ Depth-RGB time diff: %.4fs (depth %.6f - rgb %.6f)", 
                     time_diff, depth_msg->header.stamp.toSec(), last_rgb_ts);
            
            if(std::abs(time_diff) > 0.1) {
                ROS_ERROR("[DepthDiag] ❌ TIME SYNC PROBLEM! Depth-RGB diff %.4fs > 0.1s threshold!", time_diff);
            }
        }
        
        last_log_time = current_time;
    }
    
    m_buf.unlock();
}

cv::Mat getSyncedDepthImage(double rgb_timestamp)
{
    m_buf.lock();
    
    if (depth_buf.empty()) {
        m_buf.unlock();
        return cv::Mat();
    }
    
    // 找到时间戳最接近的深度图
    sensor_msgs::ImageConstPtr best_depth_msg = nullptr;
    double min_time_diff = std::numeric_limits<double>::max();
    
    // 遍历深度缓冲区找到最佳匹配
    std::queue<sensor_msgs::ImageConstPtr> temp_buf = depth_buf;
    while (!temp_buf.empty()) {
        auto depth_msg = temp_buf.front();
        temp_buf.pop();
        
        double time_diff = std::abs(depth_msg->header.stamp.toSec() - rgb_timestamp);
        if (time_diff < min_time_diff) {
            min_time_diff = time_diff;
            best_depth_msg = depth_msg;
        }
    }
    
    m_buf.unlock();
    
    // 添加时间同步调试信息
    static int sync_debug_count = 0;
    if (sync_debug_count < 10) {
        ROS_INFO("[DepthSync] RGB=%.6f, Depth=%.6f, diff=%.4fs, tolerance=0.05s",
                 rgb_timestamp, best_depth_msg ? best_depth_msg->header.stamp.toSec() : 0.0, min_time_diff);
        sync_debug_count++;
    }
    
    // 添加深度数据质量检查
    if (best_depth_msg && sync_debug_count < 10) {
        try {
            cv_bridge::CvImageConstPtr depth_ptr = cv_bridge::toCvCopy(best_depth_msg, sensor_msgs::image_encodings::MONO16);
            cv::Mat depth_raw = depth_ptr->image;
            
            // 统计非零像素
            cv::Mat nonzero_mask = depth_raw > 0;
            int nonzero_pixels = cv::countNonZero(nonzero_mask);
            int total_pixels = depth_raw.rows * depth_raw.cols;
            double nonzero_ratio = (double)nonzero_pixels / total_pixels;
            
            // 计算深度范围
            double min_depth, max_depth;
            cv::minMaxLoc(depth_raw, &min_depth, &max_depth, nullptr, nullptr, nonzero_mask);
            
            ROS_INFO("[DepthQuality] Frame %d: nonzero_ratio=%.1f%%, depth_range=[%.0f, %.0f]mm, encoding=%s",
                     sync_debug_count + 1, nonzero_ratio * 100.0, min_depth, max_depth, 
                     best_depth_msg->encoding.c_str());
        } catch (cv_bridge::Exception& e) {
            ROS_WARN("[DepthQuality] Failed to analyze depth quality: %s", e.what());
        }
    }
    
    // 如果时间差太大，返回空图像
    if (min_time_diff > 0.05) { // 50ms tolerance
        ROS_WARN("[DepthSync] Time diff too large: %.4fs", min_time_diff);
        return cv::Mat();
    }
    
    if (best_depth_msg) {
        try {
            cv_bridge::CvImageConstPtr depth_ptr = cv_bridge::toCvCopy(best_depth_msg, sensor_msgs::image_encodings::MONO16);
            return depth_ptr->image.clone();
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("[DepthSync] cv_bridge exception: %s", e.what());
            return cv::Mat();
        }
    }
    
    return cv::Mat();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    // 打印第一帧的编码信息用于调试
    static bool first_frame = true;
    if (first_frame) {
        ROS_WARN("[ImageConvert] First frame encoding: %s, size: %dx%d", 
                 img_msg->encoding.c_str(), img_msg->width, img_msg->height);
        first_frame = false;
    }
    
    cv_bridge::CvImageConstPtr ptr;
    cv::Mat gray_img;

    try {
        // VCU-RVI Structure Core 发布的彩色图编码为 8UC3（裸三通道无颜色顺序标注），
        // cv_bridge 无法直接 8UC3->MONO8，因为不知道 BGR/RGB 顺序。
        // 在 toCvShare 之前显式标为 bgr8（与 SC driver 与 OpenCV 默认一致）。
        if (img_msg->encoding == "8UC3") {
            sensor_msgs::Image relabeled;
            relabeled.header = img_msg->header;
            relabeled.height = img_msg->height;
            relabeled.width = img_msg->width;
            relabeled.is_bigendian = img_msg->is_bigendian;
            relabeled.step = img_msg->step;
            relabeled.data = img_msg->data;
            relabeled.encoding = "bgr8";
            ptr = cv_bridge::toCvCopy(relabeled, sensor_msgs::image_encodings::MONO8);
        } else {
            // 使用cv_bridge自动转换为灰度图
            ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
        }
        gray_img = ptr->image.clone();
        
        static int success_count = 0;
        success_count++;
        if (success_count <= 3) {
            ROS_WARN("[ImageConvert] Frame %d converted successfully to %dx%d MONO8", 
                     success_count, gray_img.cols, gray_img.rows);
        }
        
        return gray_img;
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("[ImageConvert] cv_bridge exception: %s", e.what());
        ROS_ERROR("[ImageConvert] Failed to convert encoding: %s", img_msg->encoding.c_str());
        return cv::Mat();
    }
}

// extract images with same timestamp from two topics
void sync_process()
{
    while(1)
    {
        if(STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                // 0.003s sync tolerance
                if(time0 < time1 - 0.003)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if(time0 > time1 + 0.003)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                    //printf("find img0 and img1\n");
                }
            }
            m_buf.unlock();
            if(!image0.empty())
                estimator.inputImage(time, image0, image1, nullptr);
        }
        else
        {
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if(!img0_buf.empty())
            {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                
                static int frame_count = 0;
                frame_count++;
                if(frame_count <= 3) {
                    ROS_WARN("[SyncDebug] Processing frame %d, encoding: %s, size: %dx%d", 
                             frame_count, 
                             img0_buf.front()->encoding.c_str(),
                             img0_buf.front()->width,
                             img0_buf.front()->height);
                }
                
                image = getImageFromMsg(img0_buf.front());
                
                if(frame_count <= 3) {
                    if(image.empty()) {
                        ROS_ERROR("[SyncDebug] Frame %d: getImageFromMsg returned empty image!", frame_count);
                    } else {
                        ROS_WARN("[SyncDebug] Frame %d: Converted to %dx%d, channels=%d", 
                                 frame_count, image.cols, image.rows, image.channels());
                    }
                }
                
                img0_buf.pop();
            }
            m_buf.unlock();
            if(!image.empty())
            {
                static int input_count = 0;
                input_count++;
                if(input_count <= 3) {
                    ROS_WARN("[SyncDebug] Calling estimator.inputImage for frame %d, time=%.6f", input_count, time);
                }
                
                try {
                    // 获取同步的深度图
                    cv::Mat depth_img = getSyncedDepthImage(time);
                    estimator.inputImage(time, image, cv::Mat(), nullptr, depth_img);
                    if(input_count <= 3) {
                        ROS_WARN("[SyncDebug] estimator.inputImage completed for frame %d", input_count);
                    }
                } catch (const std::exception& e) {
                    ROS_ERROR("[SyncDebug] Exception in estimator.inputImage: %s", e.what());
                } catch (...) {
                    ROS_ERROR("[SyncDebug] Unknown exception in estimator.inputImage");
                }
            }
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x * IMU_ACC_SCALE;
    double dy = imu_msg->linear_acceleration.y * IMU_ACC_SCALE;
    double dz = imu_msg->linear_acceleration.z * IMU_ACC_SCALE;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        //ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        //ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_drt_dyn");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 2)
    {
        printf("please intput: rosrun vins_drt_dyn vins_drt_dyn_node [config file] \n"
               "for example: rosrun vins_drt_dyn vins_drt_dyn_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    readParameters(config_file);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu;
    if(USE_IMU)
    {
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker_dyn/feature", 2000, feature_callback);
    
    // 图像订阅 - 简化逻辑，直接订阅
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
    ROS_WARN("Subscribed to image topic: %s", IMAGE0_TOPIC.c_str());
    
    ros::Subscriber sub_img1;
    if(STEREO)
    {
        sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);
        ROS_WARN("Subscribed to stereo image topic: %s", IMAGE1_TOPIC.c_str());
    }
    // Phase 3: 深度图订阅 - 修复作用域问题
    ros::Subscriber sub_depth;
    if(!DEPTH_TOPIC.empty()) {
        sub_depth = n.subscribe(DEPTH_TOPIC, 100, depth_callback);
        ROS_INFO("Subscribed to depth: %s", DEPTH_TOPIC.c_str());
    }
    
    ros::Subscriber sub_restart = n.subscribe("/vins_estimator_drt_dyn/restart_drt_dyn", 100, restart_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_estimator_drt_dyn/imu_switch_drt_dyn", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_estimator_drt_dyn/cam_switch_drt_dyn", 100, cam_switch_callback);

    std::thread sync_thread{sync_process};
    ros::spin();

    return 0;
}
