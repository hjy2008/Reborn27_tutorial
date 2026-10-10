// kalman.hpp —— 装甲板位姿卡尔曼滤波器。
//
// 状态向量:  x = [ pos(3), yaw, vel(3), vyaw, (acc(3), ayaw)? ]^T
//
//   匀速模型 (CV, 8 维, 作业推荐):
//       x = [x, y, z, yaw, vx, vy, vz, vyaw]^T
//       F = [ I4  dt*I4 ]      H = [ I4  0 ]
//           [ 0     I4  ]
//
//   匀加速模型 (CA, 12 维, 可选):
//       x = [x, y, z, yaw, vx, vy, vz, vyaw, ax, ay, az, ayaw]^T
//       多出的加速度状态让滤波器能跟上机动，代价是更依赖调参。
//
// 三个本作业特有的关键点：
//   1) yaw 是角度，观测值与状态值可能相差 2π。新息必须做角度归一化，
//      否则一次分支跳变会产生约 6.28 rad 的虚假新息，把滤波器直接拉飞。
//   2) PnP 解算存在离群帧。提供逐轴卡方新息门控，按马氏距离剔除野值。
//   3) 加速度状态与 Q 的尺度差很多，Q 用"加速度噪声标准差"参数化，
//      并由命令行 / 似然扫描调参，避免量纲写错导致滤波器冻结或发散。
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mat.hpp"

namespace kalman {

// 状态维数随模型变化
std::size_t state_dim(bool use_ca);

// 把角度归一化到 (-pi, pi]
double wrap_angle(double a);

// 返回 a 与参考角 ref 之间的最短角差，结果落在 (-pi, pi]
double angle_diff(double a, double ref);

// yaw 处理模式
enum class YawMode {
    Plain,        // 不做任何角度处理（错误示范，用来在报告里做对照）
    Wrap,         // 只做角度新息归一化
    WrapAndGate,  // 角度归一化 + 卡方新息门控（推荐，默认）
};

// 运动模型
enum class MotionModel {
    CV,  // 匀速，8 维（作业推荐）
    CA,  // 匀加速，12 维
};

std::string to_string(YawMode m);
std::string to_string(MotionModel m);
bool parse_yaw_mode(const std::string& s, YawMode& out);
bool parse_motion_model(const std::string& s, MotionModel& out);

// 滤波器参数：Q 与 R 全部可配置
struct FilterParams {
    double dt = 0.01;  // 采样间隔 (s)，作业说明"每帧数据间隔相同"
    MotionModel model = MotionModel::CV;

    // ---- 过程噪声 Q：离散白噪声加速度模型 ----
    // 位置/yaw 子块为
    //   q * [ dt^4/4  dt^3/2 ]
    //       [ dt^3/2  dt^2   ]
    // 其中 q = sigma_a^2 为加速度功率谱密度。
    // CA 模型下额外给加速度通道一个小的抖动噪声 sigma_jerk，保证 P 始终可逆。
    // 默认值由"与真实数据同特征的合成数据 + 已知真值"扫描 RMSE 选出（见 README §5.2）：
    //   观测量测噪声 sigma_meas 由相邻帧差分的稳健尺度估计；
    //   过程噪声 sigma_a 用 tests/realistic 上的 RMSE 最优值。
    //   注意 sigma_a 不能小：数据里有一段真实的 3.62 Hz 振动，
    //   sigma_a 太小滤波器跟不上它，位置 RMSE 反而比直接用原始观测更差。
    double sigma_a_xy = 60.0;  // 水平加速度噪声标准差，m/s^2
    double sigma_a_z = 36.0;   // 垂直加速度噪声标准差，m/s^2
    double sigma_a_yaw = 2.0;  // 偏航角加速度噪声标准差，rad/s^2（yaw 基本静止，小值即可）
    double sigma_jerk = 200.0; // 仅 CA：加速度通道抖动噪声，m/s^3（rad/s^3）

    // ---- 观测噪声 R ----
    // R = diag(sigma_x^2, sigma_y^2, sigma_z^2, sigma_yaw^2)
    double sigma_meas_x = 0.10;    // m  （数据实测 ~0.099）
    double sigma_meas_y = 0.06;    // m  （稳健估计 0.010~0.078，取偏保守值）
    double sigma_meas_z = 0.02;    // m  （数据实测 ~0.015）
    double sigma_meas_yaw = 0.14;  // rad（两个 yaw 簇内的散布约 ±0.15）

    // ---- 初始协方差 P0 ----
    // 注意 p0_vel 不要给太大：初始速度方差远大于位置方差时，
    // 首帧观测会通过 P 的耦合把一个巨大的错误速度灌进状态。
    double p0_pos = 0.25;   // 初始位置方差
    double p0_yaw = 0.25;   // 初始 yaw 方差
    double p0_vel = 4.0;    // 初始速度方差 (m/s)^2
    double p0_acc = 100.0;  // 仅 CA：初始加速度方差

    // ---- 新息门控 ----
    YawMode yaw_mode = YawMode::WrapAndGate;
    // 门控逐轴进行：被判为野值的那一维跳过更新，其余维度照常吸收观测。
    // 单轴卡方门限（1 自由度）：99.9% 分位 = 10.828
    double gate_chi2_axis = 10.828;
    // 位置三轴是否也参与门控。默认关闭，原因见 README：
    // 位置观测噪声是重尾的，硬门控会拒掉大量真实样本，使 P 收缩、
    // 门限随之收紧，形成"越来越拒"的正反馈并导致滤波器饥饿发散。
    // yaw 通道的野值是 2π 分支跳变而非重尾噪声，门控在那里是必要且安全的。
    bool gate_position = false;

    // ---- 自适应 R ----
    // 用滑动窗口内的新息方差在线修正 R，符号上等价于"观测变差时更信模型"。
    // 这是启发式做法，只在窗口很短、缩放被夹住时才稳定，默认关闭。
    bool adaptive_R = false;
    std::size_t adaptive_window = 30;
    double adaptive_forget = 0.05;
};

// 单帧滤波诊断信息
struct StepDiag {
    bool meas_valid = false;  // 该帧观测是否合法（非 NaN/Inf）
    bool gated = false;       // 是否被门控整体拒绝
    bool yaw_gated = false;   // yaw 轴是否被单独拒绝
    bool yaw_wrapped = false; // 该帧 yaw 新息是否发生 2π 折叠
    double nis = 0.0;         // 归一化新息平方（4 维之和，用预测协方差对角近似）
    double yaw_nis = 0.0;     // yaw 轴归一化新息平方
    double log_likelihood = 0.0;  // 该帧新息的对数似然贡献（用于调参）
    std::size_t rejected_axes = 0;  // 被门控拒绝的观测维度数
};

// 每一帧的估计结果
struct FrameEstimate {
    double t = 0.0;            // 时间戳 (s)
    Mat x_filt;                // 前向滤波后验状态 (n x 1)
    Mat x_smooth;              // RTS 平滑状态 (n x 1)
    Mat P_filt;                // 后验协方差 (n x n)
    Mat P_smooth;              // 平滑协方差 (n x n)
    StepDiag diag;
};

class ArmorKalmanFilter {
public:
    explicit ArmorKalmanFilter(const FilterParams& params);

    // 用首帧观测初始化状态；v0 为 4x1（[vx,vy,vz,vyaw]，CA 时再拼 [ax,ay,az,ayaw]），
    // 留空则速度/加速度初值置零。
    void initialize(const Mat& z0, const Mat& v0 = Mat());

    // 预测 + 更新一步。z 为 4x1 观测。首次调用等价于 initialize(z)。
    void step(const Mat& z, double t);

    // 前向滤波全部跑完后调用，做一次 RTS 后向平滑
    void smooth();

    const std::vector<FrameEstimate>& history() const { return history_; }
    const FilterParams& params() const { return p_; }
    std::size_t state_dim() const { return n_; }

    // 全部帧的新息对数似然之和（越大越好，用于 Q/R 调参）
    double total_log_likelihood() const { return total_log_likelihood_; }

    // 当前生效的观测噪声标准差（自适应 R 会改变它）
    const std::vector<double>& effective_sigma() const { return effective_sigma_; }

private:
    Mat build_F() const;
    Mat build_Q() const;
    Mat build_R() const;
    void adapt_R(const Mat& innovation);

    FilterParams p_;
    std::size_t n_ = 8;  // 状态维数：CV=8, CA=12
    Mat x_;  // 当前后验状态
    Mat P_;  // 当前后验协方差
    Mat x_pred_, P_pred_;  // 最近一次预测，供平滑与诊断使用
    std::vector<FrameEstimate> history_;
    std::vector<Mat> x_pred_hist_, P_pred_hist_;
    std::vector<double> effective_sigma_;
    std::vector<std::vector<double>> innov_window_;  // 自适应 R 的新息窗口
    double total_log_likelihood_ = 0.0;
    bool initialized_ = false;
};

}  // namespace kalman
