// data.hpp —— 数据读取与写出
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace kalman {

// 一帧装甲板位姿观测（相机坐标系）
struct ArmorSample {
    double x = 0.0;    // position_x, m
    double y = 0.0;    // position_y, m
    double z = 0.0;    // position_z, m
    double yaw = 0.0;  // yaw, rad
};

struct CsvLoadReport {
    std::size_t total_lines = 0;    // 非空行数（不含表头）
    std::size_t valid_rows = 0;     // 成功解析的行数
    std::size_t malformed_rows = 0; // 列数不足等
    std::size_t non_finite = 0;     // 解析成功但含 NaN/Inf
    std::vector<std::string> header;
};

// 读取 armor_data.csv。要求至少有 position_x/position_y/position_z/yaw 四列，
// 列顺序通过表头解析得到，因此列可以任意排列、也可以有多余列。
// 返回 false 表示致命错误（文件打不开 / 表头缺列），err 里给原因。
bool load_armor_csv(const std::string& path,
                    std::vector<ArmorSample>& out,
                    CsvLoadReport& report,
                    std::string& err);

}  // namespace kalman
