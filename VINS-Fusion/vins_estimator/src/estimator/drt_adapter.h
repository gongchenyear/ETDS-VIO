#pragma once

#include <memory>
#include <map>
#include <vector>
#include <eigen3/Eigen/Dense>

class Estimator;  // 前向声明

/**
 * DRT 初始化适配器
 * 
 * 使用 Pimpl 模式完全隔离 DRT 头文件，避免符号冲突
 */
class DRTAdapter {
public:
    explicit DRTAdapter(Estimator* estimator);
    ~DRTAdapter();
    
    // 禁止拷贝和移动（因为使用了 Pimpl）
    DRTAdapter(const DRTAdapter&) = delete;
    DRTAdapter& operator=(const DRTAdapter&) = delete;
    
    /**
     * 尝试 DRT 初始化
     * @return true: 成功, false: 失败
     */
    bool tryInitialize();
    
    /**
     * 喂入 IMU 数据（并行模式）
     */
    void feedIMU(double t, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr);
    
    /**
     * 喂入图像特征（并行模式）
     */
    void feedImage(double t, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>& features);
    
private:
    // Pimpl: 所有实现细节都在 Impl 中
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};
