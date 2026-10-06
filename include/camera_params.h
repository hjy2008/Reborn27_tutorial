#pragma once

#include <vector>

#include <opencv2/core.hpp>

// =============================================================================
// 相机参数与装甲板物理尺寸
//   数值来自作业四题目：相机内参矩阵按行展开 + 5 个畸变系数
//   装甲板为 135mm x 56mm 的矩形（135 = 板宽，56 = 灯条长度），单位统一用 mm
// =============================================================================

struct CameraParams
{
    double fx = 1789.2913422305087;
    double fy = 1789.214342198638;
    double cx = 702.93444417420312;
    double cy = 557.65168675325629;
    // 畸变系数：k1, k2, p1, p2, k3
    std::vector<double> dist = {-0.075551988063684988, 0.13117079840026666,
                                0.0014612753636792362, -0.001318090407814957, 0.0};

    cv::Mat cameraMatrix() const
    {
        return (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
                                          0.0, fy, cy,
                                          0.0, 0.0, 1.0);
    }

    cv::Mat distCoeffs() const
    {
        return cv::Mat(dist, true);   // 5x1
    }
};

// 装甲板物理尺寸（mm）：宽 = 两灯条外缘间距，高 = 灯条长度
struct ArmorSize
{
    double width  = 135.0;
    double height = 56.0;
};

inline CameraParams assignmentCameraParams() { return CameraParams(); }
inline ArmorSize    assignmentArmorSize()    { return ArmorSize(); }
