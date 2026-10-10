// main.cpp —— 作业六：卡尔曼滤波 · 装甲板位姿滤波
//
// 用法示例：
//   ./l6                                   # 用默认参数跑 armor_data.csv
//   ./l6 --data armor_data.csv --out filtered.csv
//   ./l6 --sigma-a-yaw 12 --sigma-meas-yaw 0.25 --yaw-mode wrap+gate
//   ./l6 --yaw-mode plain                  # 对照组：不做角度处理，观察滤波器被拉飞
//
// 所有 Q/R 参数都可用命令行覆盖，便于做参数扫描。
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "data.hpp"
#include "kalman.hpp"
#include "mat.hpp"

using kalman::angle_diff;
using kalman::ArmorKalmanFilter;
using kalman::ArmorSample;
using kalman::FilterParams;
using kalman::FrameEstimate;
using kalman::Mat;
using kalman::YawMode;

namespace {

const double kPi = 3.14159265358979323846;

struct Options {
    std::string data_path = "armor_data.csv";
    std::string out_csv = "filtered.csv";
    std::string summary_path = "summary.txt";
    bool quiet = false;
    bool print_likelihood = false;
    std::size_t dump_target_rows = 1200;  // 自动抽稀目标行数，0 表示全量输出
};

void print_usage() {
    std::cout <<
        R"(作业六 · 卡尔曼滤波（装甲板位姿）—— 参数说明

基本参数
  --data <path>              输入 CSV（默认 armor_data.csv）
  --out <path>               输出 CSV（默认 filtered.csv）
  --summary <path>           输出指标摘要（默认 summary.txt）
  --dt <s>                   帧间隔，默认 0.01

过程噪声 Q（离散白噪声加速度模型，参数为加速度标准差）
  --sigma-a-xy <m/s^2>       水平加速度噪声，默认 60.0
  --sigma-a-z  <m/s^2>       垂直加速度噪声，默认 36.0
  --sigma-a-yaw <rad/s^2>    偏航角加速度噪声，默认 2.0
  --sigma-jerk <v>           仅 CA：加速度通道抖动噪声，默认 200.0

观测噪声 R（参数为量测标准差）
  --sigma-meas-x   <m>       默认 0.10
  --sigma-meas-y   <m>       默认 0.06
  --sigma-meas-z   <m>       默认 0.02
  --sigma-meas-yaw <rad>     默认 0.14

初始协方差 P0
  --p0-pos <v>               默认 0.25
  --p0-yaw <v>               默认 0.25
  --p0-vel <v>               默认 4.0
  --p0-acc <v>               仅 CA，默认 100.0

运动模型
  --model <cv|ca>            cv = 匀速 8 维（默认，作业推荐）
                             ca = 匀加速 12 维

角度与野值处理
  --yaw-mode <plain|wrap|wrap+gate>
                             plain: 不处理角度（对照组）
                             wrap:  只做角度新息归一化
                             wrap+gate: 归一化 + 逐轴卡方门控（默认）
  --gate <chi2>              单轴卡方门限，默认 10.828（1 自由度 99.9%）
  --gate-position            让位置三轴也参与门控（默认关闭，会饥饿发散）
  --print-likelihood         只输出对数似然（供脚本调参扫描）

自适应观测噪声
  --adaptive-r               开启自适应 R（默认关闭）
  --adaptive-window <n>      默认 30
  --adaptive-forget <v>      默认 0.05

其他
  --dump-rows <n>            输出 CSV 最大行数（自动抽稀），0 = 全部，默认 1200
  --quiet                    只打印一行摘要
  -h, --help                 显示本帮助
)";
}

bool parse_double_arg(const std::string& s, double& out) {
    try {
        std::size_t pos = 0;
        out = std::stod(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

// 支持 --key value 与 --key=value 两种写法
bool parse_args(int argc, char** argv, Options& opt, FilterParams& p, std::string& err) {
    auto next = [&](int& i, const std::string& key) -> std::string {
        if (i + 1 >= argc) {
            err = "参数 " + key + " 缺少取值";
            return "";
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        std::string key = a, val;
        const std::size_t eq = a.find('=');
        const bool has_inline = (eq != std::string::npos);
        if (has_inline) {
            key = a.substr(0, eq);
            val = a.substr(eq + 1);
        }
        auto need = [&](const std::string& k) -> std::string {
            return has_inline ? val : next(i, k);
        };
        auto need_double = [&](const std::string& k, double& dst) -> bool {
            const std::string v = need(k);
            if (err.size()) {
                return false;
            }
            if (!parse_double_arg(v, dst)) {
                err = "参数 " + k + " 需要数值，得到: " + v;
                return false;
            }
            return true;
        };

        if (key == "-h" || key == "--help") {
            print_usage();
            std::exit(0);
        } else if (key == "--data") {
            opt.data_path = need(key);
        } else if (key == "--out") {
            opt.out_csv = need(key);
        } else if (key == "--summary") {
            opt.summary_path = need(key);
        } else if (key == "--quiet") {
            opt.quiet = true;
        } else if (key == "--print-likelihood") {
            opt.print_likelihood = true;
        } else if (key == "--dt") {
            if (!need_double(key, p.dt)) return false;
        } else if (key == "--sigma-a-xy") {
            if (!need_double(key, p.sigma_a_xy)) return false;
        } else if (key == "--sigma-a-z") {
            if (!need_double(key, p.sigma_a_z)) return false;
        } else if (key == "--sigma-a-yaw") {
            if (!need_double(key, p.sigma_a_yaw)) return false;
        } else if (key == "--sigma-jerk") {
            if (!need_double(key, p.sigma_jerk)) return false;
        } else if (key == "--sigma-meas-x") {
            if (!need_double(key, p.sigma_meas_x)) return false;
        } else if (key == "--sigma-meas-y") {
            if (!need_double(key, p.sigma_meas_y)) return false;
        } else if (key == "--sigma-meas-z") {
            if (!need_double(key, p.sigma_meas_z)) return false;
        } else if (key == "--sigma-meas-yaw") {
            if (!need_double(key, p.sigma_meas_yaw)) return false;
        } else if (key == "--p0-pos") {
            if (!need_double(key, p.p0_pos)) return false;
        } else if (key == "--p0-yaw") {
            if (!need_double(key, p.p0_yaw)) return false;
        } else if (key == "--p0-vel") {
            if (!need_double(key, p.p0_vel)) return false;
        } else if (key == "--p0-acc") {
            if (!need_double(key, p.p0_acc)) return false;
        } else if (key == "--model") {
            const std::string v = need(key);
            kalman::MotionModel m;
            if (!kalman::parse_motion_model(v, m)) {
                err = "未知 --model: " + v + "（可用 cv | ca）";
                return false;
            }
            p.model = m;
        } else if (key == "--gate") {
            if (!need_double(key, p.gate_chi2_axis)) return false;
        } else if (key == "--gate-position") {
            p.gate_position = true;
        } else if (key == "--adaptive-r") {
            p.adaptive_R = true;
        } else if (key == "--adaptive-forget") {
            if (!need_double(key, p.adaptive_forget)) return false;
        } else if (key == "--adaptive-window") {
            double v = 0;
            if (!need_double(key, v)) return false;
            p.adaptive_window = static_cast<std::size_t>(std::max(2.0, v));
        } else if (key == "--yaw-mode") {
            const std::string v = need(key);
            YawMode m;
            if (!kalman::parse_yaw_mode(v, m)) {
                err = "未知 --yaw-mode: " + v + "（可用 plain | wrap | wrap+gate）";
                return false;
            }
            p.yaw_mode = m;
        } else if (key == "--dump-rows") {
            double v = 0;
            if (!need_double(key, v)) return false;
            opt.dump_target_rows = static_cast<std::size_t>(std::max(0.0, v));
        } else {
            err = "未知参数: " + a + "（用 --help 查看说明）";
            return false;
        }
        if (err.size()) {
            return false;
        }
    }
    return true;
}

struct Stats {
    double mean = 0.0;
    double sd = 0.0;
    double min = 0.0;
    double max = 0.0;
    double mean_abs_diff = 0.0;   // 相邻帧一阶差分绝对值的均值
    double mean_abs_diff2 = 0.0;  // 二阶差分绝对值的均值（抖动指标）
};

Stats compute_stats(const std::vector<double>& v) {
    Stats s;
    if (v.empty()) {
        return s;
    }
    const double n = static_cast<double>(v.size());
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    double var = 0.0;
    for (double x : v) {
        var += (x - s.mean) * (x - s.mean);
    }
    s.sd = std::sqrt(var / n);
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    if (v.size() > 1) {
        double d1 = 0.0;
        for (std::size_t i = 1; i < v.size(); ++i) {
            d1 += std::abs(v[i] - v[i - 1]);
        }
        s.mean_abs_diff = d1 / static_cast<double>(v.size() - 1);
    }
    if (v.size() > 2) {
        double d2 = 0.0;
        for (std::size_t i = 2; i < v.size(); ++i) {
            d2 += std::abs(v[i] - 2.0 * v[i - 1] + v[i - 2]);
        }
        s.mean_abs_diff2 = d2 / static_cast<double>(v.size() - 2);
    }
    return s;
}

// 把角度序列按最短路径解卷绕，得到连续曲线（用于统计与绘图）
std::vector<double> unwrap_series(const std::vector<double>& v) {
    std::vector<double> out(v.size());
    if (v.empty()) {
        return out;
    }
    out[0] = kalman::wrap_angle(v[0]);
    for (std::size_t i = 1; i < v.size(); ++i) {
        out[i] = out[i - 1] + angle_diff(v[i], out[i - 1]);
    }
    return out;
}

// 三点中心滑动平均：y_i = (x_{i-1} + x_i + x_{i+1}) / 3
// 它只用相邻帧、不需要调参，可以作为"真值"的粗略代理，
// 用来在缺少 ground truth 的情况下估计观测噪声量级。
std::vector<double> moving_average3(const std::vector<double>& v) {
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i == 0) {
            out[i] = (v[0] + v[1]) / 2.0;
        } else if (i + 1 == v.size()) {
            out[i] = (v[i - 1] + v[i]) / 2.0;
        } else {
            out[i] = (v[i - 1] + v[i] + v[i + 1]) / 3.0;
        }
    }
    return out;
}

std::vector<double> residual_series(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> out(a.size(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

std::string fmt(double v, int prec = 6) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    FilterParams params;
    std::string err;
    if (!parse_args(argc, argv, opt, params, err)) {
        std::cerr << "参数错误: " << err << "\n\n";
        print_usage();
        return 2;
    }
    if (!(params.dt > 0.0)) {
        std::cerr << "参数错误: --dt 必须为正\n";
        return 2;
    }

    std::ostringstream log;

    // ---------------- 1. 读取数据 ----------------
    std::vector<ArmorSample> samples;
    kalman::CsvLoadReport rep;
    if (!kalman::load_armor_csv(opt.data_path, samples, rep, err)) {
        std::cerr << "读取数据失败: " << err << "\n";
        return 1;
    }
    log << "================ 作业六 · 卡尔曼滤波（装甲板位姿） ================\n\n";
    log << "[1] 数据读取\n";
    log << "    文件            : " << opt.data_path << "\n";
    log << "    表头            : ";
    for (std::size_t i = 0; i < rep.header.size(); ++i) {
        log << rep.header[i] << (i + 1 < rep.header.size() ? ", " : "");
    }
    log << "\n";
    log << "    数据行          : " << rep.total_lines << "  (有效 " << rep.valid_rows << ", 格式错误 "
        << rep.malformed_rows << ", 非有限值 " << rep.non_finite << ")\n";
    log << "    帧数 N          : " << samples.size() << "\n";
    log << "    帧间隔 dt       : " << fmt(params.dt, 6) << " s   (总时长 "
        << fmt(params.dt * static_cast<double>(samples.size() - 1), 3) << " s)\n\n";

    // 后续的统计、绘图与指标都基于"至少能形成一条轨迹"，这里先守住最小帧数。
    // 单帧输入没有帧间量可算，直接给出滤波值即可，不要去做差分/平滑。
    if (samples.size() < 2) {
        std::cerr << "数据不足: 只有 " << samples.size()
                  << " 帧，至少需要 2 帧才能做滤波（已中止）\n";
        return 1;
    }

    // yaw 的分支跳变统计（本数据集最需要注意的地方）
    std::size_t sign_flips = 0, near_pi = 0, pos_cnt = 0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (samples[i].yaw > 0.0) {
            ++pos_cnt;
        }
        if (std::abs(std::abs(samples[i].yaw) - kPi) < 0.4) {
            ++near_pi;
        }
        if (i > 0 && std::abs(angle_diff(samples[i].yaw, samples[i - 1].yaw)) > 0.5 * kPi) {
            ++sign_flips;
        }
    }
    log << "    yaw 数据特征    : 正值 " << pos_cnt << " 帧 / 负值 " << (samples.size() - pos_cnt)
        << " 帧；贴近 ±pi 的帧 " << near_pi << "；相邻帧跨分支跳变 " << sign_flips << " 次\n";
    log << "    ⇒ 观测 yaw 同时存在 +1.56 与 -1.60 两个簇，二者物理上相差 2pi，\n";
    log << "      滤波器必须做角度新息归一化，否则每跳变一次就注入约 6.28 rad 的虚假新息。\n\n";

    // ---------------- 2. 构造滤波器 ----------------
    log << "[2] 滤波器配置（Q / R 均为可配置参数）\n";
    log << "    运动模型        : " << kalman::to_string(params.model) << "\n";
    log << "    状态向量        : "
        << (params.model == kalman::MotionModel::CA
                ? "[x, y, z, yaw, vx, vy, vz, vyaw, ax, ay, az, ayaw]^T   (12 维)"
                : "[x, y, z, yaw, vx, vy, vz, vyaw]^T   (8 维, 匀速)")
        << "\n";
    log << "    观测向量        : [x, y, z, yaw]^T                     (4 维)\n";
    log << "    yaw 处理模式    : " << kalman::to_string(params.yaw_mode) << "\n";
    log << "    Q 加速度噪声    : sigma_a_xy=" << fmt(params.sigma_a_xy, 4)
        << " m/s^2, sigma_a_z=" << fmt(params.sigma_a_z, 4)
        << " m/s^2, sigma_a_yaw=" << fmt(params.sigma_a_yaw, 4) << " rad/s^2\n";
    log << "    R 量测噪声      : sigma_xyz=[" << fmt(params.sigma_meas_x, 4) << ", "
        << fmt(params.sigma_meas_y, 4) << ", " << fmt(params.sigma_meas_z, 4) << "] m, sigma_yaw="
        << fmt(params.sigma_meas_yaw, 4) << " rad\n";
    log << "    新息门控        : " << (params.yaw_mode == YawMode::WrapAndGate ? "开启" : "关闭")
        << "（逐轴卡方门限 " << fmt(params.gate_chi2_axis, 3) << ", 1 自由度 99.9%）; 位置轴 "
        << (params.gate_position ? "参与门控" : "不参与门控") << "\n";
    log << "    自适应 R        : " << (params.adaptive_R ? "开启" : "关闭") << "\n\n";

    ArmorKalmanFilter kf(params);
    const std::size_t n_state = kf.state_dim();

    // 用前两帧的有限差分估计初速度。若首帧疑似 2π 跳变，则速度初值置零更安全。
    {
        Mat z0(4, 1, 0.0);
        z0(0, 0) = samples[0].x;
        z0(1, 0) = samples[0].y;
        z0(2, 0) = samples[0].z;
        z0(3, 0) = samples[0].yaw;

        // 速度初值取零。不要用前两帧差分去初始化速度：
        // 单帧差分把观测噪声与真实机动混在一起，本数据集首帧恰好跳了 0.16 m，
        // 差分会得到约 16 m/s 的伪速度，反而把滤波器带偏。
        // 匀速模型靠 P0 里适中的速度方差自行收敛速度，比外部硬塞初值稳健得多。
        kf.initialize(z0, Mat(n_state - 4, 1, 0.0));
    }

    // ---------------- 3. 前向滤波 ----------------
    // 第 0 帧用于初始化状态，因此这里从 i = 0 开始：step() 内部只在未初始化时建立初值。
    for (std::size_t i = 0; i < samples.size(); ++i) {
        Mat z(4, 1, 0.0);
        z(0, 0) = samples[i].x;
        z(1, 0) = samples[i].y;
        z(2, 0) = samples[i].z;
        z(3, 0) = samples[i].yaw;
        kf.step(z, params.dt * static_cast<double>(i));
    }

    // ---------------- 4. RTS 后向平滑 ----------------
    kf.smooth();

    // 调参模式：只吐对数似然，供参数扫描脚本使用
    if (opt.print_likelihood) {
        std::cout << std::setprecision(6) << kf.total_log_likelihood() << "\n";
        return 0;
    }

    const std::vector<FrameEstimate>& hist = kf.history();
    if (hist.size() != samples.size()) {
        std::cerr << "内部错误: 估计帧数与输入帧数不一致\n";
        return 1;
    }

    // ---------------- 5. 统计与指标 ----------------
    std::vector<double> raw_yaw(samples.size()), filt_yaw(samples.size()), smooth_yaw(samples.size());
    std::vector<double> raw_x(samples.size()), raw_y(samples.size()), raw_z(samples.size());
    std::vector<double> filt_x(samples.size()), filt_y(samples.size()), filt_z(samples.size());
    std::vector<double> smooth_x(samples.size()), smooth_y(samples.size()), smooth_z(samples.size());
    std::vector<double> vyaw_filt(samples.size());
    std::size_t wrap_hits = 0, gated_frames = 0, yaw_gated = 0;

    for (std::size_t i = 0; i < samples.size(); ++i) {
        raw_x[i] = samples[i].x;
        raw_y[i] = samples[i].y;
        raw_z[i] = samples[i].z;
        raw_yaw[i] = samples[i].yaw;
        filt_x[i] = hist[i].x_filt(0, 0);
        filt_y[i] = hist[i].x_filt(1, 0);
        filt_z[i] = hist[i].x_filt(2, 0);
        filt_yaw[i] = hist[i].x_filt(3, 0);
        smooth_x[i] = hist[i].x_smooth(0, 0);
        smooth_y[i] = hist[i].x_smooth(1, 0);
        smooth_z[i] = hist[i].x_smooth(2, 0);
        smooth_yaw[i] = hist[i].x_smooth(3, 0);
        vyaw_filt[i] = hist[i].x_filt(7, 0);
        if (hist[i].diag.yaw_wrapped) ++wrap_hits;
        if (hist[i].diag.gated) ++gated_frames;
        if (hist[i].diag.yaw_gated) ++yaw_gated;
    }

    // 角度统一解卷绕后再统计（否则 ±pi 处的分支差会被误当成巨大误差）
    const std::vector<double> ru = unwrap_series(raw_yaw);
    const std::vector<double> fu = unwrap_series(filt_yaw);
    const std::vector<double> su = unwrap_series(smooth_yaw);

    // 三点滑动平均作为无偏参考，粗略估计观测噪声量级。
    // 三点平均把白噪声方差压到 1/3，故残差乘以 sqrt(1.5) 还原单帧噪声尺度。
    const std::vector<double> ma_yaw = moving_average3(ru);
    std::vector<double> noise_center = residual_series(ru, ma_yaw);
    for (double& v : noise_center) {
        v *= std::sqrt(1.5);
    }

    const Stats s_raw_yaw = compute_stats(ru);
    const Stats s_filt_yaw = compute_stats(fu);
    const Stats s_smooth_yaw = compute_stats(su);
    const Stats s_noise_yaw = compute_stats(noise_center);
    const Stats s_vyaw = compute_stats(vyaw_filt);

    log << "[3] yaw 通道结果（角度已解卷绕后统计）\n";
    log << "                            " << std::setw(12) << "标准差" << std::setw(16)
        << "一阶差分绝对值" << std::setw(16) << "二阶差分绝对值" << std::setw(12) << "极差\n";
    auto row = [&](const std::string& name, const Stats& s) {
        log << "    " << std::left << std::setw(20) << name << std::right << std::fixed
            << std::setprecision(6) << std::setw(12) << s.sd << std::setw(16) << s.mean_abs_diff
            << std::setw(16) << s.mean_abs_diff2 << std::setw(12) << (s.max - s.min) << "\n";
    };
    row("yaw 原始观测", s_raw_yaw);
    row("yaw 卡尔曼滤波", s_filt_yaw);
    row("yaw RTS 平滑", s_smooth_yaw);
    row("yaw 噪声估计", s_noise_yaw);
    log << "\n";

    const double diff_ratio =
        s_raw_yaw.mean_abs_diff > 0 ? s_filt_yaw.mean_abs_diff / s_raw_yaw.mean_abs_diff : 0.0;
    const double d2_ratio = s_raw_yaw.mean_abs_diff2 > 0
                                ? s_filt_yaw.mean_abs_diff2 / s_raw_yaw.mean_abs_diff2
                                : 0.0;
    log << "    帧间抖动        : 一阶差分均值 " << fmt(s_raw_yaw.mean_abs_diff, 4) << " → "
        << fmt(s_filt_yaw.mean_abs_diff, 4) << " rad/帧   (降为 " << fmt(diff_ratio * 100.0, 2)
        << "%)\n";
    log << "    二阶差分均值    : " << fmt(s_raw_yaw.mean_abs_diff2, 5) << " → "
        << fmt(s_filt_yaw.mean_abs_diff2, 5) << "   (降为 " << fmt(d2_ratio * 100.0, 2) << "%)\n";
    log << "    2π 折叠修正     : " << wrap_hits << " 帧的新息被归一化（占 "
        << fmt(100.0 * static_cast<double>(wrap_hits) / static_cast<double>(samples.size()), 2)
        << "%）\n";
    if (params.yaw_mode == YawMode::WrapAndGate) {
        log << "    门控拒绝        : yaw 轴 " << yaw_gated << " 帧，整帧 " << gated_frames << " 帧\n";
    }
    log << "\n    角速度估计      : 均值 " << fmt(s_vyaw.mean, 4) << " rad/s, 标准差 "
        << fmt(s_vyaw.sd, 4) << " rad/s, 范围 [" << fmt(s_vyaw.min, 3) << ", " << fmt(s_vyaw.max, 3)
        << "]\n";
    log << "    （dt 的取值不影响滤波形状；它只决定速度输出的物理量纲。）\n\n";

    log << "[4] 位置通道（滤波前后对比）\n";
    const Stats rx = compute_stats(raw_x), fx = compute_stats(filt_x);
    const Stats ry = compute_stats(raw_y), fy = compute_stats(filt_y);
    const Stats rz = compute_stats(raw_z), fz = compute_stats(filt_z);
    log << "    轴    原始标准差   滤波后标准差   一阶差分(原始 → 滤波)\n";
    auto prow = [&](const char* nm, const Stats& a, const Stats& b) {
        log << "    " << std::left << std::setw(6) << nm << std::right << std::fixed
            << std::setprecision(5) << std::setw(12) << a.sd << std::setw(15) << b.sd
            << std::setw(13) << a.mean_abs_diff << " → " << b.mean_abs_diff << "\n";
    };
    prow("x", rx, fx);
    prow("y", ry, fy);
    prow("z", rz, fz);
    log << "\n";

    log << "[5] 新息似然（Q/R 调参依据，越大越好）\n";
    log << "    对数似然总和    : " << fmt(kf.total_log_likelihood(), 3) << "  (每帧 "
        << fmt(kf.total_log_likelihood() / static_cast<double>(samples.size()), 4) << ")\n";
    log << "    说明            : 高斯新息对数似然同时惩罚\"跟不上\"与\"过度平滑\"，\n";
    log << "                      是比单纯看标准差更可靠的 Q/R 选择准则。\n\n";

    // ---------------- 6. 输出 CSV ----------------
    std::size_t stride = 1;
    {
        std::ofstream out(opt.out_csv);
        if (!out) {
            std::cerr << "无法写入 " << opt.out_csv << "\n";
            return 1;
        }
        out << std::setprecision(10);
        out << "t,raw_x,raw_y,raw_z,raw_yaw,"
               "filt_x,filt_y,filt_z,filt_yaw,vx,vy,vz,vyaw,"
               "smooth_x,smooth_y,smooth_z,smooth_yaw,"
               "nis,yaw_nis,yaw_wrapped,gated,yaw_gated,meas_valid\n";
        const std::size_t n = samples.size();
        if (opt.dump_target_rows > 0 && n > opt.dump_target_rows) {
            stride = (n + opt.dump_target_rows - 1) / opt.dump_target_rows;
        }
        for (std::size_t i = 0; i < n; i += stride) {
            const FrameEstimate& fe = hist[i];
            out << params.dt * static_cast<double>(i) << ',' << samples[i].x << ',' << samples[i].y
                << ',' << samples[i].z << ',' << samples[i].yaw << ',' << fe.x_filt(0, 0) << ','
                << fe.x_filt(1, 0) << ',' << fe.x_filt(2, 0) << ',' << fe.x_filt(3, 0) << ','
                << fe.x_filt(4, 0) << ',' << fe.x_filt(5, 0) << ',' << fe.x_filt(6, 0) << ','
                << fe.x_filt(7, 0) << ',' << fe.x_smooth(0, 0) << ',' << fe.x_smooth(1, 0) << ','
                << fe.x_smooth(2, 0) << ',' << fe.x_smooth(3, 0) << ',' << fe.diag.nis << ','
                << fe.diag.yaw_nis << ',' << (fe.diag.yaw_wrapped ? 1 : 0) << ','
                << (fe.diag.gated ? 1 : 0) << ',' << (fe.diag.yaw_gated ? 1 : 0) << ','
                << (fe.diag.meas_valid ? 1 : 0) << "\n";
        }
        log << "[5] 输出\n";
        log << "    " << opt.out_csv << "   (" << ((n + stride - 1) / stride) << " 行, 抽取步长 "
            << stride << ")\n";
    }

    // ---------------- 7. 摘要写盘 ----------------
    if (!opt.summary_path.empty()) {
        std::ofstream sf(opt.summary_path);
        if (sf) {
            sf << log.str();
            log << "    " << opt.summary_path << "   (指标摘要)\n";
        }
    }

    if (opt.quiet) {
        std::cout << "yaw 一阶差分均值: " << fmt(s_raw_yaw.mean_abs_diff, 5) << " → "
                  << fmt(s_filt_yaw.mean_abs_diff, 5) << " rad/帧 (" << fmt(diff_ratio * 100.0, 2)
                  << "%); 2π 折叠修正 " << wrap_hits << " 帧; 结果写入 " << opt.out_csv << "\n";
    } else {
        std::cout << log.str();
    }
    return 0;
}
