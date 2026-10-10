#include "data.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace kalman {

namespace {

std::string trim(const std::string& s) {
    const char* ws = " \t\r\n\f\v\"";
    const std::size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) {
        return "";
    }
    const std::size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

std::vector<std::string> split(const std::string& line, char sep) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == sep && !in_quotes) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    for (auto& f : out) {
        f = trim(f);
    }
    return out;
}

bool to_double(const std::string& s, double& out) {
    if (s.empty()) {
        return false;
    }
    try {
        std::size_t pos = 0;
        const double v = std::stod(s, &pos);
        if (pos != s.size()) {
            return false;
        }
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

bool load_armor_csv(const std::string& path,
                    std::vector<ArmorSample>& out,
                    CsvLoadReport& report,
                    std::string& err) {
    out.clear();
    report = CsvLoadReport{};
    err.clear();

    std::ifstream in(path);
    if (!in) {
        err = "无法打开文件: " + path;
        return false;
    }

    std::string line;
    if (!std::getline(in, line)) {
        err = "文件为空: " + path;
        return false;
    }
    report.header = split(line, ',');

    // 按表头定位列索引，不依赖固定列序
    const std::vector<std::string> wanted = {"position_x", "position_y", "position_z", "yaw"};
    std::vector<std::size_t> col(wanted.size(), std::string::npos);
    for (std::size_t w = 0; w < wanted.size(); ++w) {
        for (std::size_t c = 0; c < report.header.size(); ++c) {
            if (report.header[c] == wanted[w]) {
                col[w] = c;
                break;
            }
        }
        if (col[w] == std::string::npos) {
            err = "表头缺少列: " + wanted[w];
            return false;
        }
    }

    while (std::getline(in, line)) {
        if (trim(line).empty()) {
            continue;
        }
        ++report.total_lines;
        const std::vector<std::string> f = split(line, ',');
        std::size_t need = 0;
        for (std::size_t c : col) {
            need = std::max(need, c);
        }
        if (f.size() <= need) {
            ++report.malformed_rows;
            continue;
        }

        double v[4];
        bool ok = true;
        for (std::size_t w = 0; w < 4; ++w) {
            if (!to_double(f[col[w]], v[w])) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            ++report.malformed_rows;
            continue;
        }

        ArmorSample s{v[0], v[1], v[2], v[3]};
        if (!std::isfinite(s.x) || !std::isfinite(s.y) || !std::isfinite(s.z) ||
            !std::isfinite(s.yaw)) {
            ++report.non_finite;
        }
        out.push_back(s);
        ++report.valid_rows;
    }

    if (out.empty()) {
        err = "未从文件解析到任何数据行: " + path;
        return false;
    }
    return true;
}

}  // namespace kalman
