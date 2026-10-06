// =============================================================================
// PnP 模块实现
//   solvePnP(SOLVEPNP_IPPE) : 四个共面角点 -> 世界到相机的 R, t
//   装甲板在相机系的位姿   : tvec, R_wc
//   相机在装甲板系的位姿   : C = -R_cw * tvec, R_cw = R_wc^T
// =============================================================================
#include "pnp_solver.h"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <iostream>

namespace
{
constexpr double kRad2Deg = 180.0 / CV_PI;

// 归一化到 (-180, 180]
double normalizeAngle(double deg)
{
    while (deg <= -180.0) deg += 360.0;
    while (deg >   180.0) deg -= 360.0;
    return deg;
}

// 由装甲板尺寸生成四个角点的世界坐标
//   顺序与 rm_auto_aim 一致：左下 左上 右上 右下
//   X 向右、Y 向上、Z 指向相机
std::vector<cv::Point3f> makeObjectPoints(const ArmorSize& size)
{
    const float hw = (float)(size.width  * 0.5);
    const float hh = (float)(size.height * 0.5);
    return {
        cv::Point3f(-hw, -hh, 0.0f),   // 左下
        cv::Point3f(-hw,  hh, 0.0f),   // 左上
        cv::Point3f( hw,  hh, 0.0f),   // 右上
        cv::Point3f( hw, -hh, 0.0f),   // 右下
    };
}
}  // namespace

PnpSolver::PnpSolver(const CameraParams& camera, const ArmorSize& smallArmor, const ArmorSize& largeArmor)
    : camera_(camera),
      smallArmorPoints_(makeObjectPoints(smallArmor)),
      largeArmorPoints_(makeObjectPoints(largeArmor))
{
}

const std::vector<cv::Point3f>& PnpSolver::objectPoints(ArmorType type) const
{
    return type == ArmorType::LARGE ? largeArmorPoints_ : smallArmorPoints_;
}

double PnpSolver::distanceToCenter(const cv::Point2f& imagePoint) const
{
    const cv::Point2f center((float)camera_.cx, (float)camera_.cy);
    return cv::norm(imagePoint - center);
}

bool PnpSolver::solve(const Armor& armor, CameraPose& pose) const
{
    // 图像点顺序与 objectPoints 对应：左下 左上 右上 右下
    const std::vector<cv::Point2f> corners = {
        armor.leftLight.bottom,
        armor.leftLight.top,
        armor.rightLight.top,
        armor.rightLight.bottom,
    };
    return solve(corners, armor.type, pose);
}

bool PnpSolver::solve(const std::vector<cv::Point2f>& corners, ArmorType type, CameraPose& pose) const
{
    pose = CameraPose();
    if (corners.size() != 4)
    {
        return false;
    }

    const std::vector<cv::Point3f>& object = objectPoints(type);
    const cv::Mat K    = camera_.cameraMatrix();
    const cv::Mat dist = camera_.distCoeffs();

    cv::Mat rvec, tvec;
    bool ok = false;
    try
    {
        // IPPE 专门针对"四个共面点"，比通用迭代法更稳
        ok = cv::solvePnP(object, corners, K, dist, rvec, tvec, false, cv::SOLVEPNP_IPPE);
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "solvePnP(IPPE) 失败: " << e.what() << "，改用迭代法" << std::endl;
        ok = false;
    }
    if (!ok)
    {
        ok = cv::solvePnP(object, corners, K, dist, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    }
    if (!ok)
    {
        return false;
    }

    cv::Mat R_wc;
    cv::Rodrigues(rvec, R_wc);
    const cv::Mat R_cw = R_wc.t();
    const cv::Mat C_w  = -R_cw * tvec;

    // ---- 装甲板在相机系下的位姿 ----
    pose.armorPosition = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    pose.distance      = cv::norm(pose.armorPosition);
    rotationToEuler(R_wc, pose.roll, pose.pitch, pose.yaw);

    // ---- 相机在装甲板系下的位姿 ----
    pose.cameraPosition = cv::Vec3d(C_w.at<double>(0), C_w.at<double>(1), C_w.at<double>(2));
    rotationToEuler(R_cw, pose.camRoll, pose.camPitch, pose.camYaw);

    // ---- 重投影误差 ----
    std::vector<cv::Point2f> reproj;
    cv::projectPoints(object, rvec, tvec, K, dist, reproj);
    double sqSum = 0.0;
    for (size_t i = 0; i < reproj.size(); ++i)
    {
        sqSum += cv::norm(reproj[i] - corners[i]) * cv::norm(reproj[i] - corners[i]);
    }
    pose.reprojError = std::sqrt(sqSum / (double)reproj.size());
    pose.rvec = rvec.clone();
    pose.tvec = tvec.clone();

    // ---- 可信性检查：装甲板必须在相机前方，数值必须有限 ----
    const double z = tvec.at<double>(2);
    pose.valid = std::isfinite(z) && z > 0.0 &&
                 std::isfinite(pose.cameraPosition[0]) && std::isfinite(pose.cameraPosition[1]) &&
                 std::isfinite(pose.cameraPosition[2]) &&
                 pose.reprojError <= maxReprojError;
    return pose.valid;
}

void PnpSolver::rotationToEuler(const cv::Mat& R, double& roll, double& pitch, double& yaw)
{
    // R = Rz(yaw) * Ry(pitch) * Rx(roll)   （ZYX 顺序）
    const double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                                R.at<double>(1, 0) * R.at<double>(1, 0));
    if (sy > 1e-9)
    {
        roll  = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        pitch = std::atan2(-R.at<double>(2, 0), sy);
        yaw   = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else   // 万向锁
    {
        roll  = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        pitch = std::atan2(-R.at<double>(2, 0), sy);
        yaw   = 0.0;
    }
    roll  = normalizeAngle(roll  * kRad2Deg);
    pitch = normalizeAngle(pitch * kRad2Deg);
    yaw   = normalizeAngle(yaw   * kRad2Deg);
}
