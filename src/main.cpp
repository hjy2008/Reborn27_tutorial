// =============================================================================
// 作业四：装甲板 PnP 解算主程序
//   读视频 -> 识别模块给出装甲板角点 -> PnP 模块解算相机位姿 -> 右上角显示
//
//   模块划分：
//     include/armor_detector.h + src/armor_detector.cpp   识别模块
//     include/pnp_solver.h     + src/pnp_solver.cpp       PnP 模块
//     include/visualizer.h     + src/visualizer.cpp       可视化模块
//     include/camera_params.h                             相机与装甲板参数
//     src/main.cpp                                        流程控制
// =============================================================================
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "armor_detector.h"
#include "camera_params.h"
#include "pnp_solver.h"
#include "tracker.h"
#include "visualizer.h"

namespace
{

struct Options
{
    std::string videoPath;              // 输入视频/图片
    std::string savePath;               // 输出视频（或图片）
    std::string csvPath;                // 位姿 CSV
    int    color      = BLUE;           // 0=红 1=蓝
    bool   display    = true;           // 是否显示窗口
    bool   showAxes   = true;           // 是否画装甲板坐标轴
    bool   showBinary = false;          // 是否显示二值图
    bool   quiet      = false;          // 不打印逐帧位姿
    double startSec   = 0.0;            // 起始时间
    double endSec     = -1.0;           // 结束时间（<0 表示到结尾）
    int    maxFrames  = -1;             // 最多处理帧数
    std::string modelPath;              // 数字识别模型（默认自动查找 model/mlp.onnx）
    bool   useDigits     = true;        // 是否启用数字识别（作业三的 mlp.onnx）
    bool   filterByNumber = false;      // 是否按数字识别结果过滤装甲板
};

void printUsage(const char* prog)
{
    std::cout
        << "用法: " << prog << " [视频/图片路径] [选项]\n"
        << "选项:\n"
        << "  --color red|blue    识别红色/蓝色灯条（默认 blue）\n"
        << "  --save <路径>       保存带标注的结果视频（图片输入则存为图片）\n"
        << "  --csv  <路径>       保存逐帧位姿 (x,y,z,dist,roll,pitch,yaw,重投影误差)\n"
        << "  --no-display        不显示窗口\n"
        << "  --no-axes           不画装甲板坐标轴\n"
        << "  --binary            额外显示二值图窗口\n"
        << "  --start <秒>        从第几秒开始\n"
        << "  --end   <秒>        处理到第几秒\n"
        << "  --frames <N>        最多处理 N 帧\n"
        << "  --quiet             不打印逐帧信息\n"
        << "  --model <路径>      数字识别模型 mlp.onnx（默认自动查找 model/ 下）\n"
        << "  --no-digits         关闭数字识别（只做几何识别）\n"
        << "  --filter-number     按数字识别结果过滤（丢弃 negative / 低置信度目标）\n"
        << "  -h, --help          显示帮助\n";
}

bool isImageFile(const std::string& path)
{
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" || ext == "webp";
}

// 无图形界面（SSH / offscreen）时自动关掉窗口，避免 HighGUI 阻塞
bool hasDisplay()
{
    const char* platform = std::getenv("QT_QPA_PLATFORM");
    if (platform != nullptr && std::string(platform) == "offscreen") return false;
    return std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr;
}

// 在几个常见位置里找模型与类别表（相对当前工作目录，取自作业三 main.cpp）
std::string findModelFile(const std::string& relative)
{
    const std::string candidates[] = {
        "model/" + relative,                                   // 从 armor_pnp/ 运行
        "../model/" + relative,                                // 从 armor_pnp/build 运行
        "armor_pnp/model/" + relative,
        "../armor_pnp/model/" + relative,
        "../l3/model/" + relative,                             // 回退：作业三的模型目录
        "../../l3/model/" + relative,
        "../rm_auto_aim/armor_detector/model/" + relative,     // 回退：armor_detector 自带模型
    };
    for (const auto& c : candidates)
    {
        std::ifstream f(c);
        if (f.good()) return c;
    }
    return std::string();
}

struct Statistics
{
    int    frames = 0;
    int    detectFrames = 0;
    int    poseFrames = 0;
    double reprojSum = 0, reprojMax = 0;
    double distMin = 1e18, distMax = 0, distSum = 0;
};

}  // namespace

int main(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&](double& v) { if (i + 1 < argc) v = std::atof(argv[++i]); };
        auto nextInt = [&](int& v) { if (i + 1 < argc) v = std::atoi(argv[++i]); };
        if (arg == "-h" || arg == "--help") { printUsage(argv[0]); return 0; }
        else if (arg == "--color") { if (i + 1 < argc) opt.color = (std::string(argv[++i]) == "red") ? RED : BLUE; }
        else if (arg == "--save")  { if (i + 1 < argc) opt.savePath = argv[++i]; }
        else if (arg == "--csv")   { if (i + 1 < argc) opt.csvPath  = argv[++i]; }
        else if (arg == "--no-display") opt.display = false;
        else if (arg == "--no-axes")    opt.showAxes = false;
        else if (arg == "--binary")     opt.showBinary = true;
        else if (arg == "--quiet")      opt.quiet = true;
        else if (arg == "--model")      { if (i + 1 < argc) opt.modelPath = argv[++i]; }
        else if (arg == "--no-digits")  opt.useDigits = false;
        else if (arg == "--filter-number") opt.filterByNumber = true;
        else if (arg == "--start")      next(opt.startSec);
        else if (arg == "--end")        next(opt.endSec);
        else if (arg == "--frames")     nextInt(opt.maxFrames);
        else if (!arg.empty() && arg[0] != '-') opt.videoPath = arg;
        else { std::cerr << "未知参数: " << arg << "\n"; printUsage(argv[0]); return 1; }
    }
    if (opt.videoPath.empty()) opt.videoPath = "../raw.mp4";
    if (!opt.display || !hasDisplay()) opt.display = false;
    opt.savePath = "./result.mp4";

    // ---- 打开输入 ----
    const bool singleImage = isImageFile(opt.videoPath);
    cv::VideoCapture capture;
    cv::Mat frame;
    double fps = 30.0;
    if (singleImage)
    {
        frame = cv::imread(opt.videoPath);
        if (frame.empty()) { std::cerr << "无法读取图片: " << opt.videoPath << std::endl; return 1; }
    }
    else
    {
        capture.open(opt.videoPath);
        if (!capture.isOpened()) { std::cerr << "无法打开视频: " << opt.videoPath << std::endl; return 1; }
        fps = capture.get(cv::CAP_PROP_FPS);
        if (!(fps > 1.0)) fps = 30.0;
        if (opt.startSec > 0) capture.set(cv::CAP_PROP_POS_MSEC, opt.startSec * 1000.0);
        capture >> frame;
        if (frame.empty()) { std::cerr << "视频没有可读帧: " << opt.videoPath << std::endl; return 1; }
    }

    // ---- 各模块 ----
    // 1) 识别模块（作业三：灯条检测 + 碎片合并 + 配对 + 数字识别）
    ArmorDetector detector;
    detector.detectColor = opt.color;
    if (opt.useDigits)
    {
        const std::string modelPath = opt.modelPath.empty() ? findModelFile("mlp.onnx") : opt.modelPath;
        std::string labelPath = findModelFile("label.txt");
        if (!opt.modelPath.empty())
        {
            const size_t slash = opt.modelPath.find_last_of('/');
            labelPath = (slash == std::string::npos) ? "label.txt"
                                                     : opt.modelPath.substr(0, slash + 1) + "label.txt";
        }
        if (!modelPath.empty() && !labelPath.empty() && detector.loadNumberModel(modelPath, labelPath))
        {
            detector.filterByNumberEnabled = opt.filterByNumber;
            std::cout << "数字识别模型: " << modelPath
                      << "  类别数: " << detector.classNames.size() << std::endl;
        }
        else
        {
            std::cout << "未找到 mlp.onnx / label.txt，跳过数字识别（只输出 SMALL/LARGE）" << std::endl;
        }
    }
    // 2) 追踪模块（作业三的全局一对一匹配 + 漏检外推）
    Tracker tracker;

    const CameraParams camera = assignmentCameraParams();
    const ArmorSize    size   = assignmentArmorSize();
    PnpSolver solver(camera, size);

    Visualizer visualizer;
    visualizer.showAxes = opt.showAxes;

    cv::VideoWriter writer;
    if (!opt.savePath.empty() && !singleImage)
    {
        writer.open(opt.savePath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frame.size());
        if (!writer.isOpened())
        {
            std::cerr << "无法创建输出视频: " << opt.savePath << "（编码器不可用？）" << std::endl;
        }
    }

    std::ofstream csv;
    if (!opt.csvPath.empty())
    {
        csv.open(opt.csvPath);
        csv << "frame,time_s,tx_mm,ty_mm,tz_mm,dist_mm,roll_deg,pitch_deg,yaw_deg,"
               "camx_mm,camy_mm,camz_mm,reproj_px\n";
    }

    std::cout << "输入: " << opt.videoPath
              << "  颜色: " << (opt.color == RED ? "RED" : "BLUE")
              << "  装甲板尺寸: " << size.width << " x " << size.height << " mm"
              << "  相机: fx=" << camera.fx << " fy=" << camera.fy
              << " cx=" << camera.cx << " cy=" << camera.cy << std::endl;
    std::cout << "按键: ESC/q 退出 | 空格 暂停 | d 二值图 | a 坐标轴" << std::endl;

    Statistics stats;
    CameraPose lastPose;                 // 最近一次有效位姿（丢帧时继续显示）
    int      frameIdx   = 0;
    bool     paused     = false;
    bool     showBinary = opt.showBinary;
    double   fpsNow     = 0.0;
    auto     tPrev      = std::chrono::steady_clock::now();

    while (true)
    {
        if (!singleImage && !paused)
        {
            capture >> frame;
            if (frame.empty()) break;
        }

        // ---------------- 1) 识别模块：装甲板 + 角点 ----------------
        const std::vector<Armor> armors = detector.detect(frame);
        tracker.update(armors);                        // 更新追踪器
        // 目标：已确认（连续命中 minHitCount 帧）且本帧确实被识别到、离光心最近的那块
        const Armor* target = tracker.bestTarget(cv::Point2f((float)camera.cx, (float)camera.cy));

        // ---------------- 2) PnP 模块：相机位姿 ----------------
        CameraPose pose;
        bool poseIsCurrent = false;
        if (target != nullptr)
        {
            // 按装甲板类型（小/大）自动选择三维模型
            poseIsCurrent = solver.solve(*target, pose);
            if (poseIsCurrent) lastPose = pose;
        }

        // ---------------- 3) 统计 ----------------
        stats.frames++;
        if (target != nullptr) stats.detectFrames++;
        if (poseIsCurrent)
        {
            stats.poseFrames++;
            stats.reprojSum += pose.reprojError;
            stats.reprojMax = std::max(stats.reprojMax, pose.reprojError);
            stats.distSum  += pose.distance;
            stats.distMin   = std::min(stats.distMin, pose.distance);
            stats.distMax   = std::max(stats.distMax, pose.distance);
        }

        const double timeSec = singleImage ? 0.0 : frameIdx / fps;
        if (poseIsCurrent && !opt.quiet)
        {
            std::cout << cv::format("frame %4d t=%5.2fs | 装甲板: T=(%7.1f,%7.1f,%7.1f)mm "
                                    "dist=%6.1f | RPY=(%7.1f,%6.1f,%7.1f) | 相机: (%7.1f,%7.1f,%7.1f)mm "
                                    "| %s | reproj=%.2fpx\n",
                                    frameIdx, timeSec,
                                    pose.armorPosition[0], pose.armorPosition[1], pose.armorPosition[2],
                                    pose.distance, pose.roll, pose.pitch, pose.yaw,
                                    pose.cameraPosition[0], pose.cameraPosition[1], pose.cameraPosition[2],
                                    target->type == ArmorType::LARGE ? "LARGE" : "SMALL",
                                    pose.reprojError);
        }
        if (csv.is_open())
        {
            csv << frameIdx << ',' << timeSec << ',';
            if (poseIsCurrent)
            {
                csv << pose.armorPosition[0] << ',' << pose.armorPosition[1] << ',' << pose.armorPosition[2] << ','
                    << pose.distance << ',' << pose.roll << ',' << pose.pitch << ',' << pose.yaw << ','
                    << pose.cameraPosition[0] << ',' << pose.cameraPosition[1] << ',' << pose.cameraPosition[2] << ','
                    << pose.reprojError;
            }
            else
            {
                csv << ",,,,,,,,,,,,";
            }
            csv << '\n';
        }

        // ---------------- 4) 可视化 ----------------
        const auto tNow = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(tNow - tPrev).count();
        tPrev = tNow;
        if (dt > 0) fpsNow = 0.9 * fpsNow + 0.1 / dt;   // 简单平滑

        detector.drawResults(frame, false);   // 灯条轴线 + 各装甲板对角线（作业三的画法）
        visualizer.render(frame, armors, target,
                          poseIsCurrent ? pose : lastPose, poseIsCurrent, solver, fpsNow);

        // ---------------- 5) 输出 ----------------
        if (writer.isOpened()) writer.write(frame);
        if (opt.display)
        {
            cv::imshow("Armor PnP Pose", frame);
            if (showBinary && !detector.binaryImg.empty()) cv::imshow("Binary", detector.binaryImg);
            const int key = cv::waitKey(singleImage ? 0 : (paused ? 0 : 1));
            if (key == 27 || key == 'q') break;
            if (key == ' ') paused = !paused;
            if (key == 'd') showBinary = !showBinary;
            if (key == 'a') visualizer.showAxes = !visualizer.showAxes;
        }

        frameIdx++;
        if (singleImage) break;
        if (opt.maxFrames > 0 && frameIdx >= opt.maxFrames) break;
        if (opt.endSec > 0 && timeSec >= opt.endSec) break;
    }

    if (capture.isOpened()) capture.release();
    if (writer.isOpened()) writer.release();
    if (csv.is_open()) csv.close();
    if (opt.display) cv::destroyAllWindows();

    // 图片输入：直接把标注结果存成图片
    if (singleImage && !opt.savePath.empty())
    {
        cv::imwrite(opt.savePath, frame);
        std::cout << "已保存标注结果: " << opt.savePath << std::endl;
    }

    // ---------------- 6) 汇总 ----------------
    std::cout << "\n================ 统计 ================\n";
    std::cout << "处理帧数       : " << stats.frames << std::endl;
    if (stats.frames > 0)
    {
        std::cout << cv::format("检测到装甲板   : %d 帧 (%.1f%%)\n",
                                stats.detectFrames, 100.0 * stats.detectFrames / stats.frames);
        std::cout << cv::format("PnP 有效位姿   : %d 帧 (%.1f%%)\n",
                                stats.poseFrames, 100.0 * stats.poseFrames / stats.frames);
    }
    if (stats.poseFrames > 0)
    {
        std::cout << cv::format("重投影误差     : 平均 %.2f px, 最大 %.2f px\n",
                                stats.reprojSum / stats.poseFrames, stats.reprojMax);
        std::cout << cv::format("相机距离       : 最小 %.1f mm, 最大 %.1f mm, 平均 %.1f mm\n",
                                stats.distMin, stats.distMax, stats.distSum / stats.poseFrames);
        std::cout << cv::format("平均重投影误差 : %.2f px（灯条端点自身有系统偏差，仅作兜底检查）\n",
                                stats.reprojSum / stats.poseFrames);
    }
    if (!opt.savePath.empty() && !singleImage) std::cout << "结果视频       : " << opt.savePath << std::endl;
    if (!opt.csvPath.empty())                  std::cout << "位姿 CSV       : " << opt.csvPath  << std::endl;
    return 0;
}
