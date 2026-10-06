#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "armor_detector.h"
#include "camera_params.h"

// =============================================================================
// PnP 模块：由装甲板角点的像素坐标解算位姿（结构参考 rm_auto_aim 的 PnPSolver）
//
// 世界坐标系（装甲板坐标系）约定：
//   原点 O = 装甲板中心
//   X 轴   = 沿装甲板宽度方向向右（135mm 方向）
//   Y 轴   = 沿灯条方向向上（56mm 方向）
//   Z 轴   = 垂直于装甲板指向相机（右手系）
//
// solvePnP 得到的是"世界 -> 相机"的 R, t，本文同时给出两个方向的位姿：
//   1) armorPosition / roll,pitch,yaw : 装甲板在相机系下的位姿（相机看到的目标状态）
//   2) cameraPosition / camRoll,...   : 相机在装甲板系下的位姿（作业问题所求）
// 两者互为逆变换，显示时按作业参考图用 1) 的 Tx/Ty/Tz/Roll/Pitch/Yaw。
// =============================================================================

struct CameraPose
{
    bool valid = false;

    // ---- 装甲板在相机坐标系下的位姿（= solvePnP 的 tvec 与 R_wc）----
    cv::Vec3d armorPosition{0, 0, 0};   // 装甲板中心在相机系下的坐标 (mm)
    double distance    = 0.0;           // 相机到装甲板中心的距离 (mm)
    double roll        = 0.0;           // 装甲板相对相机的姿态角 (度, ZYX)
    double pitch       = 0.0;
    double yaw         = 0.0;

    // ---- 相机在装甲板（世界）坐标系下的位姿（互为逆变换）----
    cv::Vec3d cameraPosition{0, 0, 0};  // 相机光心在装甲板系下的坐标 (mm)
    double camRoll     = 0.0;           // 相机相对装甲板的姿态角 (度, ZYX)
    double camPitch    = 0.0;
    double camYaw      = 0.0;

    double reprojError = 0.0;           // 四角点重投影 RMS 误差（像素）
    cv::Mat rvec, tvec;                 // solvePnP 原始输出：世界 -> 相机
};

class PnpSolver
{
public:
    // 小/大装甲板的世界坐标尺寸（mm）：宽 = 两灯条外缘间距，高 = 灯条长度
    explicit PnpSolver(const CameraParams& camera = CameraParams(),
                       const ArmorSize&   smallArmor = ArmorSize{135.0, 56.0},
                       const ArmorSize&   largeArmor = ArmorSize{230.0, 56.0});

    // 主要接口：直接给一块装甲板，按其类型（小/大）选择三维模型
    bool solve(const Armor& armor, CameraPose& pose) const;

    // 底层接口：给定四角点像素坐标 + 装甲板类型
    bool solve(const std::vector<cv::Point2f>& corners, ArmorType type, CameraPose& pose) const;

    // 到光心的像素距离（rm_auto_aim 用它挑选目标装甲板）
    double distanceToCenter(const cv::Point2f& imagePoint) const;

    // 世界坐标下的角点（顺序：左下 左上 右上 右下，与 Armor 的四个端点一致）
    const std::vector<cv::Point3f>& objectPoints(ArmorType type) const;

    const CameraParams& camera() const { return camera_; }

    // 重投影误差上限（像素）。灯条端点本身有若干像素的系统偏差，
    // 这里只作为"解算明显失败"的兜底判据，不作为精度指标。
    double maxReprojError = 30.0;

private:
    // 世界 -> 相机 的旋转矩阵 -> 相机 -> 世界（R_cw）以及 ZYX 欧拉角
    static void rotationToEuler(const cv::Mat& R, double& roll, double& pitch, double& yaw);

    CameraParams camera_;
    std::vector<cv::Point3f> smallArmorPoints_;
    std::vector<cv::Point3f> largeArmorPoints_;
};
