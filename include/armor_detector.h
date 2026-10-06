#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "detector_params.h"

// =============================================================================
// 识别模块（代码取自作业三 l3/main.cpp，按面向对象结构拆到独立文件，算法未改）
//   流程：灰度二值化 -> 找灯条轮廓 -> 合并断裂碎片 -> 灯条配对成装甲板
//         -> 数字识别（mlp.onnx）
//   输出：Armor（左右灯条、中心、四个角点，角点顺序 左上/右上/右下/左下）
// =============================================================================

// 装甲板类型：按两灯条中心距 / 灯条长 的比值区分大小装甲板
enum class ArmorType { SMALL, LARGE, INVALID };
inline const char* ARMOR_TYPE_STR[3] = {"SMALL", "LARGE", "INVALID"};

// -----------------------------------------------------------------------------
// Light：灯条
//   length     : 灯条长度（top 到 bottom 的欧氏距离）
//   width      : 等效宽度 = 轮廓面积 / length
//   ratio      : width / length
//   tilt       : 灯条轴线与竖直方向的夹角
//   top/bottom : 灯条轴线与外接矩形上下边界的交点，即灯条的上下端点
// -----------------------------------------------------------------------------
struct Light
{
    cv::Rect    box;              // 轮廓外接正矩形
    cv::Point2f top, bottom;      // 灯条上下端点
    cv::Point2f center;           // 灯条中心
    double  length = 0;           // 灯条长度
    double  width  = 0;           // 等效宽度
    double  ratio  = 0;           // width / length
    double  tilt   = 0;           // 与竖直方向夹角(度)
    double  fill   = 0;           // 填充率
    double  area   = 0;           // 轮廓像素点数（面积）
    int     color  = -1;          // RED / BLUE
};

// -----------------------------------------------------------------------------
// Armor：装甲板 = 左右两根灯条
// -----------------------------------------------------------------------------
class Armor
{
public:
    Light     leftLight, rightLight;
    cv::Point2f center;
    ArmorType type = ArmorType::INVALID;

    // 数字识别结果
    cv::Mat     numberImg;              // 送入网络的 20x28 二值图（调试用）
    std::string number;                 // 识别出的数字/类别
    float       confidence = 0;         // 置信度
    std::string classificationResult;   // "2: 98.5%" 供显示

    // 追踪相关
    cv::Point2f velocity;               // 像素/帧（用最近两次命中估计，漏检时按它外推位置）
    bool    isTracking = false;         // 本帧是否被识别命中
    int     lostCount  = 0;             // 连续未匹配帧数
    int     hitCount   = 0;             // 连续命中次数

    Armor();
    Armor(const Light& l1, const Light& l2);

    // 装甲板宽度 = 两灯条中心距；高度 = 两灯条平均长度
    float width()  const;
    float height() const;

    // 装甲板倾角 = 左灯条中心 -> 右灯条中心 的方向（度）
    float angle() const;

    // 四个角点（物理上就是两根灯条的四个端点），顺序：左上 右上 右下 左下
    std::vector<cv::Point2f> getCorners() const;

    // 画出角点框（红框 + 黄色角点 + 绿色中心）
    void draw(cv::Mat& frame,
              cv::Scalar boxColor    = cv::Scalar(0, 0, 255),
              cv::Scalar cornerColor = cv::Scalar(0, 255, 255)) const;
};

// -----------------------------------------------------------------------------
// ArmorDetector：识别器
// -----------------------------------------------------------------------------
class ArmorDetector
{
public:
    // 候选配对：两根灯条下标 + 打分 + 大小类型
    struct Candidate
    {
        size_t    i = 0, j = 0;
        double    score = 0;
        ArmorType type = ArmorType::SMALL;
    };

    int    binaryThres    = BINARY_THRES;
    int    detectColor    = BLUE;                    // 0=红 1=蓝
    double minLightLength = MIN_LIGHT_LENGTH;
    double minContourArea = MIN_CONTOUR_AREA;
    double maxScaleRatio  = ARMOR_MAX_SCALE_RATIO;   // 中心距 / 较大灯条长 上限

    cv::Mat binaryImg;                               // 调试用：最近一帧二值图
    cv::Ptr<cv::dnn::Net> numberNet;                 // 数字识别网络（可为空）
    std::vector<std::string> classNames;             // 类别名
    double numberThreshold = NUMBER_THRESHOLD;       // 置信度阈值
    bool   filterByNumberEnabled = false;            // 是否按数字识别结果过滤

    // 加载 ONNX 模型与类别表；失败返回 false（此时只做几何识别）
    bool loadNumberModel(const std::string& modelPath, const std::string& labelPath);

    // 主入口：识别一帧
    std::vector<Armor> detect(const cv::Mat& img);

    const std::vector<Light>& lights() const;
    const std::vector<Armor>& armors() const;

    // 可视化：灯条轴线 + 装甲板对角线；可选标注灯条序号
    void drawResults(cv::Mat& img, bool showIndex = false) const;

private:
    // ---- 1) 预处理：灰度 + 固定阈值二值化 ----
    cv::Mat preprocessImage(const cv::Mat& img);

    // ---- 2) 找灯条 ----
    std::vector<Light> findLights(const cv::Mat& img, const cv::Mat& binaryImg);

    // ---- 2.5) 合并被阈值截断成几段的同一根灯条 ----
    std::vector<Light> mergeFragments(const std::vector<Light>& lights) const;

    // 单根灯条的几何校验
    bool isLight(const Light& light) const;

    // ---- 3) 灯条配对 ----
    std::vector<Armor> matchLights(const std::vector<Light>& lights);
    bool isArmorPair(const Light& light1, const Light& light2, const std::vector<Light>& lights,
                     Candidate& c) const;
    double spreadOfPair(const Light& light1, const Light& light2) const;
    bool hasLightBetween(const Light& light1, const Light& light2,
                         const std::vector<Light>& lights) const;

    // ---- 4) 数字识别 ----
    void extractNumbers(const cv::Mat& src, std::vector<Armor>& armors) const;
    void filterByNumber(std::vector<Armor>& armors) const;
    void classify(std::vector<Armor>& armors);

    std::vector<Light> lights_;
    std::vector<Armor> armors_;
};
