// =============================================================================
// 目标追踪实现（逻辑与作业三 l3/main.cpp 一致）
// =============================================================================
#include "tracker.h"

#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

void Tracker::update(const vector<Armor>& currentArmors)
{
    // ================= 1) 本帧识别结果与追踪器的全局一对一匹配 =================
    //  先算出所有"可以匹配"的 (追踪器, 当前装甲板) 组合并打分，
    //  再从小到大做全局一对一分配：既不会两块板抢同一个检测，也不会两块板互换身份。
    for (auto& armor : trackedArmors_) armor.isTracking = false;

    {
        struct Match
        {
            size_t ti, ci;
            float  score;
        };
        vector<Match> matches;
        for (size_t ti = 0; ti < trackedArmors_.size(); ++ti) {
            const Armor& tr = trackedArmors_[ti];
            const float   scale = tr.width() + tr.height();
            if (scale <= 0) continue;
            for (size_t ci = 0; ci < currentArmors.size(); ++ci) {
                const Armor& cur = currentArmors[ci];

                // 门限随"丢失帧数"渐进放宽：刚开始跟丢时门限很紧（不会误配到旁边那块板），
                // 丢得越久门限越松（最多 1.0 x 装甲板尺度），这样目标重新出现时能接回同一条轨迹。
                const float base     = maxDistRatio * scale;
                const float gapGrow  = min(base * 1.5f * (float)tr.lostCount, scale * 0.6f - base);
                const float motion   = (float)norm(tr.velocity) * (float)(tr.lostCount + 1);
                const float gate     = base + max(0.0f, gapGrow) + min(motion, base);

                const float dist = (float)norm(cur.center - tr.center);
                if (dist > gate) continue;

                // 取两条"线"之间的最小夹角（0~90 度）
                float angGap = fmod(fabs(cur.angle() - tr.angle()), 180.0f);
                if (angGap > 90.0f) angGap = 180.0f - angGap;
                if (angGap > maxAngleGap) continue;

                // 尺寸不能差太多：差得多说明是远近不同的另一块装甲板
                const float curScale = cur.width() + cur.height();
                const float ratio    = curScale > scale ? curScale / scale : scale / curScale;
                if (ratio > maxScaleRatio) continue;

                matches.push_back({ti, ci, dist / scale + (ratio - 1.0f) * 0.5f});
            }
        }
        sort(matches.begin(), matches.end(),
             [](const Match& a, const Match& b) { return a.score < b.score; });

        vector<bool> trackUsed(trackedArmors_.size(), false);
        vector<bool> armorUsed(currentArmors.size(), false);
        for (const auto& m : matches) {
            if (trackUsed[m.ti] || armorUsed[m.ci]) continue;
            trackUsed[m.ti] = true;
            armorUsed[m.ci] = true;

            Armor& tracked = trackedArmors_[m.ti];
            const Armor& cur = currentArmors[m.ci];

            // 命中计数：短暂漏检（lostCount 不大）继续累加，只有长时间丢失才重新计数。
            tracked.hitCount = (tracked.lostCount <= hitResetFrames)
                                   ? tracked.hitCount + 1 : 1;

            // 用本次匹配的位置差估计速度（低通平滑，避免抖动放大）
            if (tracked.lostCount == 0) {
                tracked.velocity = tracked.velocity * 0.5f + (cur.center - tracked.center) * 0.5f;
            } else {
                const float k = 1.0f / (float)(tracked.lostCount + 1);
                tracked.velocity = tracked.velocity * (1.0f - k) +
                                   (cur.center - tracked.center) * k;
            }

            tracked.leftLight  = cur.leftLight;
            tracked.rightLight = cur.rightLight;
            tracked.center     = cur.center;
            tracked.type       = cur.type;
            tracked.number     = cur.number;
            tracked.confidence = cur.confidence;
            tracked.classificationResult = cur.classificationResult;
            tracked.isTracking = true;
            tracked.lostCount  = 0;
        }

        // 没匹配上的识别结果作为新目标
        for (size_t ci = 0; ci < currentArmors.size(); ++ci) {
            if (!armorUsed[ci]) trackedArmors_.push_back(currentArmors[ci]);
        }
    }

    // ================= 2) 漏检时按速度外推位置 =================
    for (auto& armor : trackedArmors_) {
        if (armor.isTracking) continue;      // 本帧被识别命中，位置已经是实测值

        armor.lostCount++;
        if (armor.lostCount > 3) armor.velocity = Point2f(0, 0);   // 陈旧速度不可信

        armor.leftLight.top     += armor.velocity;
        armor.leftLight.bottom  += armor.velocity;
        armor.rightLight.top    += armor.velocity;
        armor.rightLight.bottom += armor.velocity;
        armor.leftLight.center  += armor.velocity;
        armor.rightLight.center += armor.velocity;
        armor.center            += armor.velocity;
    }

    // 连续丢失过多帧就删除
    trackedArmors_.erase(
        remove_if(trackedArmors_.begin(), trackedArmors_.end(),
                  [this](const Armor& a) { return a.lostCount > maxLostFrames; }),
        trackedArmors_.end());

    // 限制追踪器数量，防止 FPS 崩溃
    if ((int)trackedArmors_.size() > maxTrackers) {
        sort(trackedArmors_.begin(), trackedArmors_.end(),
             [](const Armor& a, const Armor& b) {
                 if (a.isTracking != b.isTracking) return a.isTracking;
                 if (a.lostCount != b.lostCount) return a.lostCount < b.lostCount;
                 return a.hitCount > b.hitCount;
             });
        trackedArmors_.resize(maxTrackers);
    }
}

const Armor* Tracker::bestTarget(const Point2f& imageCenter) const
{
    const Armor* best = nullptr;
    double bestDistance = 1e18;
    for (const Armor& armor : trackedArmors_) {
        if (armor.hitCount < minHitCount) continue;   // 未确认
        if (!armor.isTracking) continue;              // 本帧没被识别到（角点是外推的，不能用于 PnP）
        const double d = norm(armor.center - imageCenter);
        if (d < bestDistance) {
            bestDistance = d;
            best = &armor;
        }
    }
    return best;
}
