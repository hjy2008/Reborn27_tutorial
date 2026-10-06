// =============================================================================
// 可视化模块实现
// =============================================================================
#include "visualizer.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace
{
const cv::Scalar kGreen (0, 255, 0);
const cv::Scalar kYellow(0, 255, 255);
const cv::Scalar kGray  (140, 140, 140);
const cv::Scalar kPanel (32, 32, 32);

// 各轴长度一致，便于观察姿态
const cv::Scalar kAxisColor[3] = {cv::Scalar(0, 0, 255),    // X 红
                                  cv::Scalar(0, 255, 0),    // Y 绿
                                  cv::Scalar(255, 0, 0)};   // Z 蓝
}  // namespace

void Visualizer::render(cv::Mat& frame,
                        const std::vector<Armor>& armors,
                        const Armor* target,
                        const CameraPose& pose,
                        bool poseIsCurrent,
                        const PnpSolver& solver,
                        double fps) const
{
    (void)armors;   // 所有候选装甲板由识别模块的 drawResults() 画出，这里只强调参与解算的目标
    if (target != nullptr)
    {
        drawTarget(frame, *target);
    }
    if (showAxes && poseIsCurrent && pose.valid && !pose.rvec.empty())
    {
        drawAxes(frame, pose, solver);
    }
    const int hudHeight = drawHud(frame, pose, poseIsCurrent, target, fps);
    if (poseIsCurrent && pose.valid)
    {
        drawCameraPanel(frame, pose, 12 + hudHeight + 8);
    }
}

void Visualizer::drawTarget(cv::Mat& frame, const Armor& armor) const
{
    armor.draw(frame, cv::Scalar(0, 0, 255), kYellow);
    cv::circle(frame, armor.center, 4, kGreen, -1);
}

void Visualizer::drawAxes(cv::Mat& frame, const CameraPose& pose, const PnpSolver& solver) const
{
    const double L = axisLength;
    const std::vector<cv::Point3f> axisPoints = {
        cv::Point3f(0, 0, 0),
        cv::Point3f((float)L, 0, 0),
        cv::Point3f(0, (float)L, 0),
        cv::Point3f(0, 0, (float)L),
    };
    std::vector<cv::Point2f> projected;
    try
    {
        cv::projectPoints(axisPoints, pose.rvec, pose.tvec,
                          solver.camera().cameraMatrix(), solver.camera().distCoeffs(), projected);
    }
    catch (const cv::Exception&)
    {
        return;
    }
    if (projected.size() != 4) return;
    for (int i = 0; i < 3; ++i)
    {
        cv::arrowedLine(frame, projected[0], projected[i + 1], kAxisColor[i], 2, cv::LINE_AA, 0, 0.15);
    }
}

int Visualizer::drawHud(cv::Mat& frame, const CameraPose& pose, bool poseIsCurrent,
                        const Armor* target, double fps) const
{
    char lines[10][64];
    int  count = 0;
    if (poseIsCurrent && pose.valid)
    {
        std::snprintf(lines[count++], 64, "Tx:    %7.1f mm", pose.armorPosition[0]);
        std::snprintf(lines[count++], 64, "Ty:    %7.1f mm", pose.armorPosition[1]);
        std::snprintf(lines[count++], 64, "Tz:    %7.1f mm", pose.armorPosition[2]);
        std::snprintf(lines[count++], 64, "Dist:  %7.1f mm", pose.distance);
        std::snprintf(lines[count++], 64, "Roll:  %7.1f",    pose.roll);
        std::snprintf(lines[count++], 64, "Pitch: %7.1f",    pose.pitch);
        std::snprintf(lines[count++], 64, "Yaw:   %7.1f",    pose.yaw);
    }
    else
    {
        std::snprintf(lines[count++], 64, "No armor detected");
    }
    // 识别结果（作业三的数字识别输出，type 由两灯条中心距/灯条长判定）
    if (target != nullptr)
    {
        if (!target->classificationResult.empty())
        {
            std::snprintf(lines[count++], 64, "Armor: %s %s",
                          ARMOR_TYPE_STR[(int)target->type], target->classificationResult.c_str());
        }
        else
        {
            std::snprintf(lines[count++], 64, "Armor: %s", ARMOR_TYPE_STR[(int)target->type]);
        }
    }
    std::snprintf(lines[count++], 64, "FPS:   %7.1f", fps);

    const char* title = (poseIsCurrent && pose.valid) ? "== Camera Pose ==" : "== Camera Pose (lost) ==";

    // 面板尺寸：由最长一行文本决定，放在画面右上角
    const double fontScale = 0.55;
    const int    lineH     = 22;
    int textW = (int)std::strlen(title) * 11;
    for (int i = 0; i < count; ++i)
    {
        textW = std::max(textW, cv::getTextSize(lines[i], cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, nullptr).width);
    }
    const int padX = 12, padY = 10;
    const int panelW = textW + padX * 2;
    const int panelH = lineH * (count + 1) + padY * 2;
    const int x0 = frame.cols - panelW - 12;
    const int y0 = 12;

    cv::Mat roi = frame(cv::Rect(x0, y0, panelW, panelH));
    cv::Mat panel(panelH, panelW, frame.type(), kPanel);
    cv::addWeighted(panel, 0.55, roi, 0.45, 0, roi);

    const cv::Scalar textColor = (poseIsCurrent && pose.valid) ? kGreen : kYellow;
    cv::putText(frame, title, cv::Point(x0 + padX, y0 + padY + 14),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, 1, cv::LINE_AA);
    for (int i = 0; i < count; ++i)
    {
        cv::putText(frame, lines[i], cv::Point(x0 + padX, y0 + padY + 14 + lineH * (i + 1)),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, 1, cv::LINE_AA);
    }
    return panelH;
}

void Visualizer::drawCameraPanel(cv::Mat& frame, const CameraPose& pose, int topY) const
{
    char title[64];
    char body[96];
    std::snprintf(title, 64, "Cam in Armor Frame (mm)");
    std::snprintf(body, 96, "X:%7.1f Y:%7.1f Z:%7.1f",
                  pose.cameraPosition[0], pose.cameraPosition[1], pose.cameraPosition[2]);

    const double fontScale = 0.5;
    const int    lineH     = 20;
    int textW = std::max(cv::getTextSize(title, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, nullptr).width,
                         cv::getTextSize(body,  cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, nullptr).width);
    const int padX = 12, padY = 8;
    const int panelW = textW + padX * 2;
    const int panelH = lineH * 2 + padY * 2;
    const int x0 = frame.cols - panelW - 12;
    const int y0 = topY;   // 主面板下方

    cv::Mat roi = frame(cv::Rect(x0, y0, panelW, panelH));
    cv::Mat panel(panelH, panelW, frame.type(), kPanel);
    cv::addWeighted(panel, 0.55, roi, 0.45, 0, roi);

    const cv::Scalar color(255, 200, 0);
    cv::putText(frame, title, cv::Point(x0 + padX, y0 + padY + 13),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, color, 1, cv::LINE_AA);
    cv::putText(frame, body, cv::Point(x0 + padX, y0 + padY + 13 + lineH),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, color, 1, cv::LINE_AA);
}
