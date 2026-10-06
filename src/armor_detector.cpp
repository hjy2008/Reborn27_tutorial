// =============================================================================
// 识别模块实现（取自作业三 l3/main.cpp，算法未改）
//   1) preprocessImage : 灰度 + 固定阈值二值化（灯条是画面中最亮的区域）
//   2) findLights      : 轮廓筛选 + 轴线拟合，得到每根灯条的上下端点
//   3) mergeFragments  : 把被阈值截断的同一根灯条合并回去
//   4) matchLights     : 灯条两两配对打分 + 贪心一对一分配，得到装甲板
//   5) 数字识别        : warpPerspective 出数字 ROI，送 mlp.onnx 分类
// =============================================================================
#include "armor_detector.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

Armor::Armor() : center(0, 0) {}

Armor::Armor(const Light& l1, const Light& l2)
{
    if (l1.center.x < l2.center.x) {
        leftLight = l1, rightLight = l2;
    } else {
        leftLight = l2, rightLight = l1;
    }
    center   = (leftLight.center + rightLight.center) * 0.5f;
    hitCount = 1;
}

float Armor::width()  const { return (float)norm(leftLight.center - rightLight.center); }

float Armor::height() const { return (float)((leftLight.length + rightLight.length) * 0.5); }

float Armor::angle() const
{
    const Point2f d = rightLight.center - leftLight.center;
    return (float)(atan2(d.y, d.x) * 180.0 / CV_PI);
}

vector<Point2f> Armor::getCorners() const
{
    vector<Point2f> pts(4);
    pts[0] = leftLight.top;       // 左上
    pts[1] = rightLight.top;      // 右上
    pts[2] = rightLight.bottom;   // 右下
    pts[3] = leftLight.bottom;    // 左下
    return pts;
}

void Armor::draw(Mat& frame, Scalar boxColor, Scalar cornerColor) const
{
    vector<Point2f> corners = getCorners();
    for (int i = 0; i < 4; i++) {
        line(frame, corners[i], corners[(i + 1) % 4], boxColor, 2);
    }
    // 四个物理角点（黄色实心圆）
    for (int i = 0; i < 4; i++) {
        circle(frame, corners[i], 4, cornerColor, -1);
    }
    // 中心点（绿色）
    circle(frame, center, 3, Scalar(0, 255, 0), -1);
}

bool ArmorDetector::loadNumberModel(const string& modelPath, const string& labelPath)
{
    try {
        numberNet = makePtr<dnn::Net>(dnn::readNetFromONNX(modelPath));
    } catch (const exception& e) {
        cerr << "加载模型失败 " << modelPath << ": " << e.what() << endl;
        numberNet.release();
        return false;
    }
    ifstream labelFile(labelPath);
    if (!labelFile) {
        cerr << "无法读取类别表 " << labelPath << endl;
        numberNet.release();
        return false;
    }
    classNames.clear();
    string line;
    while (getline(labelFile, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        classNames.push_back(line);
    }
    return !classNames.empty();
}

vector<Armor> ArmorDetector::detect(const Mat& img)
{
    binaryImg = preprocessImage(img);
    lights_   = findLights(img, binaryImg);
    vector<Armor> armors = matchLights(lights_);
    if (numberNet && !armors.empty()) {
        extractNumbers(img, armors);
        classify(armors);
        if (filterByNumberEnabled) filterByNumber(armors);
    }
    armors_ = armors;   // 同步成员，便于 armors() 取用
    return armors;
}

const vector<Light>& ArmorDetector::lights() const { return lights_; }

const vector<Armor>& ArmorDetector::armors() const { return armors_; }

Mat ArmorDetector::preprocessImage(const Mat& img)
{
    Mat grayImg, binary;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    threshold(grayImg, binary, binaryThres, 255, THRESH_BINARY);
    return binary;
}

vector<Light> ArmorDetector::findLights(const Mat& img, const Mat& binaryImg)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binaryImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Light> lights;
    for (const auto& contour : contours) {
        if ((int)contour.size() < MIN_CONTOUR_PTS) continue;

        const Rect bRect        = boundingRect(contour);
        const RotatedRect rRect = minAreaRect(contour);

        // 把轮廓画到局部 mask 上，统计轮廓内部真实像素数（= 轮廓面积）
        Mat mask = Mat::zeros(bRect.size(), CV_8UC1);
        vector<Point> maskContour;
        maskContour.reserve(contour.size());
        for (const auto& p : contour) {
            maskContour.emplace_back(p - Point(bRect.x, bRect.y));
        }
        fillPoly(mask, {maskContour}, 255);
        vector<Point> points;
        findNonZero(mask, points);

        if ((double)points.size() < minContourArea) continue;

        // 填充率 = 轮廓面积 / 最小外接矩形面积（灯条是实心矩形，填充率接近 1）
        const double rRectArea = rRect.size.width * rRect.size.height;
        const double fillRatio = rRectArea > 0 ? points.size() / rRectArea : 0.0;

        // 最小二乘拟合灯条轴线：结果是过 (x0,y0)、方向 (vx,vy) 的直线
        Vec4f param;
        fitLine(points, param, DIST_L2, 0, 0.01, 0.01);
        const double vx = param[0], vy = param[1];

        Point2f top, bottom;
        double  tiltAngle;
        if (fabs(vy) < 1e-6) {
            // 轴线水平（躺倒的灯条）：直接取外接矩形左右边界
            top    = Point2f((float)bRect.x, (float)(bRect.y + bRect.height / 2));
            bottom = Point2f((float)(bRect.x + bRect.width),
                             (float)(bRect.y + bRect.height / 2));
            tiltAngle = 90.0;
        } else {
            // 直线 y = k*x + b，与外接矩形上/下边界求交点，得到灯条端点
            const double k = vy / (fabs(vx) < 1e-9 ? 1e-9 : vx);
            const double b = (param[3] + bRect.y) - k * (param[2] + bRect.x);
            top    = Point2f((float)((bRect.y - b) / k), (float)bRect.y);
            bottom = Point2f((float)((bRect.y + bRect.height - b) / k),
                             (float)(bRect.y + bRect.height));

            // 近似水平的轮廓会让 k 极小，交点被推到画面外（曾算出过 30 万像素的"灯条"）。
            // 轮廓本身一定是灯条，其端点到轴线的横向偏移不会超过外接矩形宽度，据此做约束。
            const float margin = bRect.width * 0.1f + 2.0f;
            const float xLow   = bRect.x - margin;
            const float xHigh  = bRect.x + bRect.width + margin;
            if (top.x < xLow || top.x > xHigh || bottom.x < xLow || bottom.x > xHigh) continue;
            // 与竖直方向的夹角 = |atan(|vy|/|vx|) - 90°|
            tiltAngle = fabs(fabs(atan2(fabs(vy), fabs(vx)) * 180.0 / CV_PI) - 90.0);
        }

        Light light;
        light.box    = bRect;
        light.top    = top;
        light.bottom = bottom;
        light.center = (top + bottom) * 0.5f;
        light.length = norm(top - bottom);
        light.area   = (double)points.size();
        light.width  = light.length > 0 ? light.area / light.length : 0.0;
        light.ratio  = light.length > 0 ? light.width / light.length : 0.0;
        light.tilt   = tiltAngle;
        light.fill   = fillRatio;

        // 长度硬上界：灯条不可能比自身外接矩形的对角线还长
        const double maxPossible = sqrt((double)bRect.width * bRect.width +
                                        (double)bRect.height * bRect.height) * 1.05 + 1.0;
        if (light.length > maxPossible) continue;

        if (!isLight(light)) continue;

        // 颜色判定需要访问原图 ROI，先做边界检查避免越界
        if (bRect.x < 0 || bRect.y < 0 ||
            bRect.x + bRect.width  > img.cols ||
            bRect.y + bRect.height > img.rows) {
            continue;
        }
        // 统计轮廓内部像素的 R、B 通道之和判断灯条颜色（输入为 BGR）
        long long sumR = 0, sumB = 0;
        const Mat roi = img(bRect);
        for (int i = 0; i < roi.rows; i++) {
            for (int j = 0; j < roi.cols; j++) {
                if (mask.at<uchar>(i, j) == 0) continue;   // 非轮廓内部像素直接跳过
                const Vec3b& px = roi.at<Vec3b>(i, j);     // BGR
                sumB += px[0];
                sumR += px[2];
            }
        }
        light.color = sumR > sumB ? RED : BLUE;
        lights.emplace_back(light);
    }

    // 阈值化会把一根很亮的宽灯条截断成几段，导致一根灯条被当成多根：
    // 这里把"位于同一条轴线上、沿轴间隙很小"的碎片合并回一根灯条。
    return mergeFragments(lights);
}

vector<Light> ArmorDetector::mergeFragments(const vector<Light>& lights) const
{
    const size_t n = lights.size();
    vector<bool> used(n, false);
    vector<Light> merged;

    for (size_t i = 0; i < n; ++i) {
        if (used[i]) continue;
        vector<size_t> group{i};
        used[i] = true;

        bool changed = true;
        while (changed) {
            changed = false;
            double top    = lights[group[0]].top.y;
            double bottom = lights[group[0]].bottom.y;
            double axisX  = 0;
            for (size_t g : group) {
                top     = min(top, (double)lights[g].top.y);
                bottom  = max(bottom, (double)lights[g].bottom.y);
                axisX  += lights[g].center.x;
            }
            axisX /= (double)group.size();
            const double accLen = max(bottom - top, 1.0);

            for (size_t k = 0; k < n; ++k) {
                if (used[k]) continue;
                const Light& o = lights[k];
                const double axisDev = fabs(o.center.x - axisX);      // 轴线偏差
                const double gap     = max(0.0, max(top, (double)o.top.y) -
                                                min(bottom, (double)o.bottom.y));
                if (axisDev < FRAGMENT_AXIS_TOL * o.length &&
                    gap     < FRAGMENT_GAP_TOL * accLen) {
                    group.push_back(k);
                    used[k] = true;
                    changed = true;
                }
            }
        }

        if (group.size() == 1) {
            merged.push_back(lights[i]);
            continue;
        }

        // 用各碎片端点的平均值作为轴线，上下端取极值，得到合并后的灯条
        Point2f top(0, 0), bottom(0, 0);
        double minTop = 1e9, maxBottom = -1e9;
        for (size_t g : group) {
            top    += lights[g].top;
            bottom += lights[g].bottom;
            minTop    = min(minTop, (double)lights[g].top.y);
            maxBottom = max(maxBottom, (double)lights[g].bottom.y);
        }
        top    *= 1.0f / (float)group.size();
        bottom *= 1.0f / (float)group.size();
        top.y    = (float)minTop;
        bottom.y = (float)maxBottom;

        Light light;
        light.top    = top;
        light.bottom = bottom;
        light.center = (top + bottom) * 0.5f;
        light.length = norm(top - bottom);
        for (size_t g : group) light.area += lights[g].area;   // 面积累加
        light.width  = light.length > 0 ? light.area / light.length : 0.0;
        light.ratio  = light.length > 0 ? light.width / light.length : 0.0;
        light.tilt   = fabs(fabs(atan2(bottom.x - top.x, bottom.y - top.y) * 180.0 / CV_PI));
        if (light.tilt > 90.0) light.tilt = 180.0 - light.tilt;
        light.fill   = lights[group[0]].fill;
        light.box    = boundingRect(vector<Point2f>{light.top, light.bottom});
        light.color  = lights[group[0]].color;

        // 合并后仍要满足灯条校验（太宽/太斜的直接丢弃）
        if (isLight(light)) merged.push_back(light);
    }
    return merged;
}

bool ArmorDetector::isLight(const Light& light) const
{
    const bool ratioOk = LIGHT_MIN_RATIO < light.ratio && light.ratio < LIGHT_MAX_RATIO;
    const bool angleOk = light.tilt < LIGHT_MAX_ANGLE;
    const bool fillOk  = light.fill > LIGHT_MIN_FILL;
    const bool sizeOk  = light.length >= minLightLength;
    return ratioOk && angleOk && fillOk && sizeOk;
}

vector<Armor> ArmorDetector::matchLights(const vector<Light>& lights)
{
    armors_.clear();

    // 3.1 枚举候选对并打分
    vector<Candidate> cands;
    for (size_t i = 0; i < lights.size(); ++i) {
        for (size_t j = i + 1; j < lights.size(); ++j) {
            if (lights[i].color != detectColor || lights[j].color != detectColor) continue;

            Candidate c;
            if (!isArmorPair(lights[i], lights[j], lights, c)) continue;
            c.i = i;
            c.j = j;
            cands.push_back(c);
        }
    }

    // 3.2 按分数贪心分配：每根灯条最多属于一块装甲板
    sort(cands.begin(), cands.end(),
         [](const Candidate& a, const Candidate& b) { return a.score < b.score; });

    vector<bool> used(lights.size(), false);
    for (const auto& c : cands) {
        if (used[c.i] || used[c.j]) continue;
        used[c.i] = used[c.j] = true;

        Armor armor(lights[c.i], lights[c.j]);
        armor.type = c.type;
        armors_.emplace_back(armor);
    }
    return armors_;
}

bool ArmorDetector::isArmorPair(const Light& light1, const Light& light2, const vector<Light>& lights,
                     Candidate& c) const
{
    // (1) 两根灯条长度应接近
    const double maxLen = max(light1.length, light2.length);
    const double minLen = min(light1.length, light2.length);
    if (maxLen <= 0) return false;
    const double lightLengthRatio = minLen / maxLen;
    if (lightLengthRatio <= ARMOR_MIN_LIGHT_RATIO) return false;

    // (2) 同一块装甲板的两根灯条应近似平行
    const double tiltGap = fabs(light1.tilt - light2.tilt);
    if (tiltGap > ARMOR_MAX_TILT_GAP) return false;

    // (3) 两灯条连线接近水平
    const Point2f diff  = light1.center - light2.center;
    const double  angle = fabs(atan2(fabs(diff.y), fabs(diff.x))) * 180.0 / CV_PI;
    if (angle >= ARMOR_MAX_ANGLE) return false;

    // (4) 距离判据：以灯条自身长度为尺度，量纲一化后与远近无关
    const double centerDist     = norm(light1.center - light2.center);
    const double meanLength     = (light1.length + light2.length) * 0.5;
    const double centerDistance = centerDist / meanLength;   // 中心距 / 平均灯条长
    const double scaleDistance  = centerDist / maxLen;       // 中心距 / 较大灯条长

    // 两根灯条不能贴在一起：同一根宽灯条的碎片间距 ≈ 自身长度，据此排除
    if (centerDistance < ARMOR_MIN_CENTER_DIST) return false;

    const bool smallOk = ARMOR_MIN_SMALL_DIST <= centerDistance &&
                         centerDistance < ARMOR_MAX_SMALL_DIST;
    const bool largeOk = ARMOR_MIN_LARGE_DIST <= centerDistance &&
                         centerDistance < ARMOR_MAX_LARGE_DIST;
    if (!smallOk && !largeOk) return false;
    if (scaleDistance > maxScaleRatio) return false;          // 隔太远，不是同一块装甲板

    // (5) 两灯条之间不应夹着第三根灯条（防止把不同装甲板的灯条连起来）
    if (hasLightBetween(light1, light2, lights)) return false;

    c.score = W_TILT_GAP * tiltGap / ARMOR_MAX_TILT_GAP +
              W_LEN_GAP  * (1.0 - lightLengthRatio) +
              W_DIST     * scaleDistance +
              W_ANGLE    * angle / ARMOR_MAX_ANGLE +
              W_SPREAD   * spreadOfPair(light1, light2);
    c.type  = largeOk ? ArmorType::LARGE : ArmorType::SMALL;
    return true;
}

double ArmorDetector::spreadOfPair(const Light& light1, const Light& light2) const
{
    const Point2f vTop = light2.top - light1.top;
    const Point2f vBot = light2.bottom - light1.bottom;
    const Point2f vCen = light2.center - light1.center;
    const Point2f vMean = (vTop + vBot + vCen) / 3.0f;
    const double spread = max(max(norm(vTop - vMean), norm(vBot - vMean)),
                              norm(vCen - vMean));
    const double meanLen = (light1.length + light2.length) * 0.5;
    return meanLen > 0 ? spread / meanLen : 0.0;
}

bool ArmorDetector::hasLightBetween(const Light& light1, const Light& light2,
                         const vector<Light>& lights) const
{
    const Point2f c1   = light1.center;
    const Point2f axis = light2.center - c1;
    const double  len  = norm(axis);
    if (len <= 1e-6) return false;
    const Point2f u = axis / (float)len;      // 连线方向
    const Point2f n(-u.y, u.x);               // 连线法向

    const auto proj = [&](const Point2f& p) { return (double)(p - c1).dot(u); };
    const auto lat  = [&](const Point2f& p) { return fabs((double)(p - c1).dot(n)); };

    const double lo = max(min(proj(light1.top), proj(light1.bottom)),
                          min(proj(light2.top), proj(light2.bottom)));
    const double hi = min(max(proj(light1.top), proj(light1.bottom)),
                          max(proj(light2.top), proj(light2.bottom)));
    if (lo >= hi) return false;               // 两灯条沿连线不重叠，中间没有空间

    const double halfBand = ARMOR_BETWEEN_HALF * (light1.length + light2.length) * 0.5;
    for (const auto& other : lights) {
        if (other.center == light1.center || other.center == light2.center) continue;
        const double p = proj(other.center);
        if (lo < p && p < hi && lat(other.center) < halfBand) return true;
    }
    return false;
}

void ArmorDetector::extractNumbers(const Mat& src, vector<Armor>& armors) const
{
    for (auto& armor : armors) {
        const Point2f lightsVertices[4] = {
            armor.leftLight.bottom, armor.leftLight.top,
            armor.rightLight.top,   armor.rightLight.bottom};

        const int topLightY    = (WARP_HEIGHT - WARP_LIGHT_LENGTH) / 2 - 1;
        const int bottomLightY = topLightY + WARP_LIGHT_LENGTH;
        const int warpWidth = armor.type == ArmorType::SMALL ? WARP_SMALL_WIDTH
                                                             : WARP_LARGE_WIDTH;
        const Point2f targetVertices[4] = {
            Point2f(0, (float)bottomLightY),
            Point2f(0, (float)topLightY),
            Point2f((float)(warpWidth - 1), (float)topLightY),
            Point2f((float)(warpWidth - 1), (float)bottomLightY)};

        const Mat rotationMatrix =
            getPerspectiveTransform(lightsVertices, targetVertices);
        Mat numberImage;
        warpPerspective(src, numberImage, rotationMatrix,
                        Size(warpWidth, WARP_HEIGHT));

        // 取中间 ROI，灰度化 + OTSU 二值化
        const Rect roi(Point((warpWidth - ROI_WIDTH) / 2, 0), Size(ROI_WIDTH, ROI_HEIGHT));
        numberImage = numberImage(roi).clone();
        cvtColor(numberImage, numberImage, COLOR_BGR2GRAY);
        threshold(numberImage, numberImage, 0, 255, THRESH_BINARY | THRESH_OTSU);

        armor.numberImg = numberImage;
    }
}

void ArmorDetector::filterByNumber(vector<Armor>& armors) const
{
    if (!numberNet) return;
    armors.erase(
        remove_if(armors.begin(), armors.end(),
                  [this](const Armor& armor) {
                      if (armor.confidence < numberThreshold) return true;
                      for (const char* ignore : IGNORE_CLASSES) {
                          if (armor.number == ignore) return true;
                      }
                      // 大装甲板不可能是 1/2(哨兵)/guard，小装甲板不可能是 outpost/base
                      if (armor.type == ArmorType::LARGE) {
                          return armor.number == "outpost" || armor.number == "2" ||
                                 armor.number == "guard";
                      }
                      if (armor.type == ArmorType::SMALL) {
                          return armor.number == "1" || armor.number == "base";
                      }
                      return false;
                  }),
        armors.end());
}

void ArmorDetector::classify(vector<Armor>& armors)
{
    for (auto& armor : armors) {
        if (armor.numberImg.empty()) continue;

        Mat image = armor.numberImg.clone();
        image.convertTo(image, CV_32F, 1.0 / 255.0);   // 归一化到 [0,1]

        Mat blob = dnn::blobFromImage(image);
        numberNet->setInput(blob);
        Mat outputs = numberNet->forward();

        // softmax
        double maxVal = 0;
        minMaxLoc(outputs.reshape(1, 1), nullptr, &maxVal, nullptr, nullptr);
        Mat softmaxProb;
        exp(outputs - maxVal, softmaxProb);
        softmaxProb /= (float)sum(softmaxProb)[0];

        Point classIdPoint;
        double confidence = 0;
        minMaxLoc(softmaxProb.reshape(1, 1), nullptr, &confidence, nullptr, &classIdPoint);

        armor.confidence = (float)confidence;
        const int labelId = classIdPoint.x;
        armor.number = (labelId >= 0 && labelId < (int)classNames.size())
                           ? classNames[labelId] : string("?");
        armor.classificationResult = format("%s: %.1f%%", armor.number.c_str(),
                                            armor.confidence * 100.0);
    }
}

void ArmorDetector::drawResults(Mat& img, bool showIndex) const
{
    for (size_t k = 0; k < lights_.size(); k++) {
        const auto& light = lights_[k];
        const Scalar lineColor = light.color == RED ? Scalar(255, 255, 0) : Scalar(255, 0, 255);
        line(img, light.top, light.bottom, lineColor, 1);
        circle(img, light.top, 3, Scalar(255, 255, 255), 1);
        circle(img, light.bottom, 3, Scalar(255, 255, 255), 1);
        if (showIndex) {
            const string tag = format("#%zu L%.0f", k, light.length);
            putText(img, tag, light.top + Point2f(4, -4), FONT_HERSHEY_SIMPLEX, 0.5,
                    Scalar(255, 255, 255), 1);
        }
    }
    for (const auto& armor : armors_) {
        line(img, armor.leftLight.top, armor.rightLight.bottom, Scalar(0, 255, 0), 2);
        line(img, armor.leftLight.bottom, armor.rightLight.top, Scalar(0, 255, 0), 2);
    }
}
