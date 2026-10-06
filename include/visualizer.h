#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "armor_detector.h"
#include "pnp_solver.h"

// =============================================================================
// 可视化模块：把识别结果与 PnP 位姿画到画面上
//   - 装甲板框 + 四个角点
//   - 装甲板坐标轴（X 红 / Y 绿 / Z 蓝），直观展示解出来的姿态
//   - 右上角位姿面板：x, y, z, dist, yaw, pitch, roll（对应作业要求的显示）
// =============================================================================
class Visualizer
{
public:
    bool   showAxes   = true;    // 是否画装甲板坐标轴
    double axisLength = 60.0;    // 坐标轴长度 (mm)
    int    thickness  = 2;

    // armors        : 本帧所有候选装甲板
    // target        : 选中用于解算的装甲板（可为 nullptr）
    // pose          : 相机位姿（target 为空时传上一次的有效位姿，配合 poseIsCurrent=false 显示）
    // poseIsCurrent : 该位姿是否来自当前帧
    void render(cv::Mat& frame,
                const std::vector<Armor>& armors,
                const Armor* target,
                const CameraPose& pose,
                bool poseIsCurrent,
                const PnpSolver& solver,
                double fps) const;

private:
    void drawTarget(cv::Mat& frame, const Armor& armor) const;
    void drawAxes(cv::Mat& frame, const CameraPose& pose, const PnpSolver& solver) const;
    // 主面板：位姿 + 识别结果；返回面板高度（像素），供副面板定位
    int  drawHud(cv::Mat& frame, const CameraPose& pose, bool poseIsCurrent,
                 const Armor* target, double fps) const;
    // 副面板：相机在装甲板坐标系下的位置（作业问题所求的"相机在世界系下的位姿"）
    void drawCameraPanel(cv::Mat& frame, const CameraPose& pose, int topY) const;
};
