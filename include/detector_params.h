#pragma once

// =============================================================================
// 识别模块可调参数（直接取自作业三 l3/main.cpp，集中在此便于调整）
// =============================================================================

// ---- 1) 预处理 ----
const int    BINARY_THRES    = 160;   // 灰度二值化阈值（灯条是画面中最亮区域）

// ---- 2) 灯条校验 (armor_detector LightParams) ----
const double LIGHT_MIN_RATIO = 0.08;  // 短边/长边 下限
const double LIGHT_MAX_RATIO = 0.50;  // 短边/长边 上限（放宽：灯条正对/斜视时会显得很粗）
const double LIGHT_MAX_ANGLE = 40.0;  // 灯条与竖直方向夹角上限（度）
const double LIGHT_MIN_FILL  = 0.80;  // 轮廓面积 / 最小外接矩形面积 下限

// ---- 尺寸过滤（分辨率相关，用于滤掉远处小目标与噪点）----
const double MIN_LIGHT_LENGTH = 22.0; // 灯条最短长度(px)
const double MIN_CONTOUR_AREA = 35.0; // 轮廓最小面积(px^2)
const int    MIN_CONTOUR_PTS  = 5;    // 轮廓最少点数（fillPoly / fitLine 需要）

// ---- 3) 装甲板校验 (armor_detector ArmorParams) ----
const double ARMOR_MIN_LIGHT_RATIO = 0.60;  // 两灯条长度比 下限
// 两根灯条中心距（以平均灯条长为单位）的下限：同一根宽灯条被阈值截断成几段时，
// 这些碎片会彼此靠近（间距≈自身长度）并被误配成"装甲板"，用这个下限把它们挡掉。
const double ARMOR_MIN_CENTER_DIST = 0.8;   // 中心距/平均灯条长 下限
const double ARMOR_MIN_SMALL_DIST  = 0.8;   // 小装甲板：中心距/平均灯条长 下限
const double ARMOR_MAX_SMALL_DIST  = 3.2;   // 小装甲板：上限
const double ARMOR_MIN_LARGE_DIST  = 3.2;   // 大装甲板：下限
const double ARMOR_MAX_LARGE_DIST  = 5.5;   // 大装甲板：上限
const double ARMOR_MAX_ANGLE       = 35.0;  // 两灯条连线与水平方向夹角上限（度）

// ---- 灯条分组（多灯条时按距离/尺度判断是否属于同一块装甲板）----
const double ARMOR_MAX_SCALE_RATIO = 5.5;   // 中心距 / 较大灯条长 上限
const double ARMOR_MAX_TILT_GAP    = 15.0;  // 同一装甲板两灯条倾角差上限（度）
const double ARMOR_BETWEEN_HALF    = 0.25;  // "夹在中间"判定：侧向偏移 < 该系数 x 平均灯条长

// ---- 灯条碎片合并 ----
const double FRAGMENT_AXIS_TOL     = 0.8;   // 碎片中心到主灯条轴线的垂距 < 该系数 x 碎片长
const double FRAGMENT_GAP_TOL      = 0.5;   // 沿轴线方向的间隙 < 该系数 x 已累积长度

// ---- 配对打分权重 ----
const double W_TILT_GAP = 3.0, W_LEN_GAP = 2.0, W_DIST = 1.0, W_ANGLE = 0.3;
const double W_SPREAD   = 1.5;   // 平行四边形一致性权重

// ---- 4) 数字识别（推理模型 mlp.onnx）----
const double NUMBER_THRESHOLD  = 0.7;    // 置信度阈值
inline const char* const IGNORE_CLASSES[] = {"negative"};   // 直接丢弃的类别
const int    WARP_LIGHT_LENGTH = 12;     // 透视变换时灯条在目标图中的长度
const int    WARP_HEIGHT       = 28;     // 透视变换后的高度
const int    WARP_SMALL_WIDTH  = 32;     // 小装甲板透视变换后的宽度
const int    WARP_LARGE_WIDTH  = 54;     // 大装甲板透视变换后的宽度
const int    ROI_WIDTH         = 20;     // 送入网络的 ROI 宽
const int    ROI_HEIGHT        = 28;     // 送入网络的 ROI 高

// ---- 5) 目标追踪 ----
const int    MAX_TRACKERS     = 5;     // 最多同时追踪的装甲板数量
const int    MIN_HIT_COUNT    = 3;     // 连续命中帧数门槛（确认目标）
const int    MAX_LOST_FRAMES  = 6;     // 连续多少帧没被识别到就删除追踪器
const int    HIT_RESET_FRAMES = 10;    // 连续丢失超过该帧数才把命中计数清零
const float  TRACK_MAX_DIST_RATIO  = 0.35f;  // 中心位移 / (宽+高) 上限
const float  TRACK_MAX_ANGLE_GAP   = 15.0f;  // 装甲板倾角差上限（度）
const float  TRACK_MAX_SCALE_RATIO = 1.8f;   // 尺寸变化倍数上限

// 颜色编号（与 armor_detector 一致：0=红 1=蓝）
const int RED  = 0;
const int BLUE = 1;
