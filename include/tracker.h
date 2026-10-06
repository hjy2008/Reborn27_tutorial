#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "armor_detector.h"

// =============================================================================
// 目标追踪（代码取自作业三 l3/main.cpp 的跟踪段，封装成一个类）
//   update()：
//     1) 用本帧识别结果与已有追踪器做"全局一对一"匹配（先算所有可行组合并打分，
//        再按分数从小到大分配，避免两块板抢同一个检测或互换身份）；
//     2) 漏检时按速度外推角点位置；
//     3) 连续丢失超过 maxLostFrames 帧的追踪器删除，并限制追踪器数量。
//   命中计数达到 minHitCount 的才算"确认目标"，可交给 PnP 使用。
// =============================================================================
class Tracker
{
public:
    // 参数默认值来自 detector_params.h
    int   maxTrackers    = MAX_TRACKERS;
    int   minHitCount    = MIN_HIT_COUNT;
    int   maxLostFrames  = MAX_LOST_FRAMES;
    int   hitResetFrames = HIT_RESET_FRAMES;
    float maxDistRatio   = TRACK_MAX_DIST_RATIO;
    float maxAngleGap    = TRACK_MAX_ANGLE_GAP;
    float maxScaleRatio  = TRACK_MAX_SCALE_RATIO;

    // 用本帧识别结果更新追踪器
    void update(const std::vector<Armor>& currentArmors);

    void reset() { trackedArmors_.clear(); }

    const std::vector<Armor>& armors() const { return trackedArmors_; }

    // 已确认（hitCount >= minHitCount）且本帧被识别到的目标中，离光心最近的一个；
    // 没有确认目标时返回 nullptr
    const Armor* bestTarget(const cv::Point2f& imageCenter) const;

private:
    std::vector<Armor> trackedArmors_;
};
