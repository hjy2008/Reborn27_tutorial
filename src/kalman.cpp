#include "kalman.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kalman {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr std::size_t kMeasDim = 4;  // [x, y, z, yaw]，不随模型改变
constexpr std::size_t kPosDim = 4;   // 位置类状态维数：x, y, z, yaw
}  // namespace

std::size_t state_dim(bool use_ca) { return use_ca ? 12 : 8; }

double wrap_angle(double a) {
    if (!std::isfinite(a)) {
        return a;
    }
    a = std::fmod(a + kPi, kTwoPi);
    if (a < 0.0) {
        a += kTwoPi;
    }
    return a - kPi;
}

double angle_diff(double a, double ref) { return wrap_angle(a - ref); }

std::string to_string(YawMode m) {
    switch (m) {
        case YawMode::Plain:
            return "plain";
        case YawMode::Wrap:
            return "wrap";
        case YawMode::WrapAndGate:
            return "wrap+gate";
    }
    return "unknown";
}

std::string to_string(MotionModel m) {
    switch (m) {
        case MotionModel::CV:
            return "CV(匀速, 8维)";
        case MotionModel::CA:
            return "CA(匀加速, 12维)";
    }
    return "unknown";
}

bool parse_yaw_mode(const std::string& s, YawMode& out) {
    if (s == "plain") {
        out = YawMode::Plain;
        return true;
    }
    if (s == "wrap") {
        out = YawMode::Wrap;
        return true;
    }
    if (s == "wrap+gate" || s == "wrap-gate" || s == "gate") {
        out = YawMode::WrapAndGate;
        return true;
    }
    return false;
}

bool parse_motion_model(const std::string& s, MotionModel& out) {
    if (s == "cv" || s == "CV") {
        out = MotionModel::CV;
        return true;
    }
    if (s == "ca" || s == "CA") {
        out = MotionModel::CA;
        return true;
    }
    return false;
}

ArmorKalmanFilter::ArmorKalmanFilter(const FilterParams& params) : p_(params) {
    n_ = (p_.model == MotionModel::CA) ? 12 : 8;
    effective_sigma_ = {p_.sigma_meas_x, p_.sigma_meas_y, p_.sigma_meas_z, p_.sigma_meas_yaw};
}

Mat ArmorKalmanFilter::build_F() const {
    const double dt = p_.dt;
    Mat F = Mat::identity(n_);
    for (std::size_t i = 0; i < kPosDim; ++i) {
        F(i, i + kPosDim) = dt;  // pos += v * dt
    }
    if (n_ == 12) {
        for (std::size_t i = 0; i < kPosDim; ++i) {
            F(i, i + 2 * kPosDim) = 0.5 * dt * dt;  // pos += 0.5 a dt^2
            F(i + kPosDim, i + 2 * kPosDim) = dt;   // v   += a dt
        }
    }
    return F;
}

Mat ArmorKalmanFilter::build_Q() const {
    // 离散白噪声加速度模型：每个通道用同一个 q = sigma_a^2 把 [pos, vel] 耦合起来，
    //   Q_pp = q dt^4/4,  Q_pv = q dt^3/2,  Q_vv = q dt^2
    // CA 模型改用标准的白噪声加加速度(jerk)模型，三阶耦合一次给全。
    Mat Q(n_, n_, 0.0);
    const double dt = p_.dt;
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;
    const double dt5 = dt4 * dt;
    const double dt6 = dt3 * dt3;

    const double sigma[kPosDim] = {p_.sigma_a_xy, p_.sigma_a_xy, p_.sigma_a_z, p_.sigma_a_yaw};
    for (std::size_t i = 0; i < kPosDim; ++i) {
        const double q = sigma[i] * sigma[i];
        if (n_ == 8) {
            Q(i, i) = q * dt4 / 4.0;
            Q(i, i + 4) = q * dt3 / 2.0;
            Q(i + 4, i) = q * dt3 / 2.0;
            Q(i + 4, i + 4) = q * dt2;
        } else {
            // 白噪声 jerk：jerk PSD = (sigma_a / dt)^2 会过于激进，
            // 这里把 sigma_a 直接当作 jerk 尺度，用 dt 构造标准的三阶模型。
            Q(i, i) = q * dt6 / 36.0;
            Q(i, i + 4) = q * dt5 / 12.0;
            Q(i, i + 8) = q * dt4 / 6.0;
            Q(i + 4, i) = q * dt5 / 12.0;
            Q(i + 4, i + 4) = q * dt4 / 4.0;
            Q(i + 4, i + 8) = q * dt3 / 2.0;
            Q(i + 8, i) = q * dt4 / 6.0;
            Q(i + 8, i + 4) = q * dt3 / 2.0;
            // 加速度通道加一点独立抖动，避免 P 在长时间无机动时数值退化
            Q(i + 8, i + 8) = q * dt2 + p_.sigma_jerk * p_.sigma_jerk * dt2;
        }
    }
    return Q;
}

Mat ArmorKalmanFilter::build_R() const {
    Mat R(kMeasDim, kMeasDim, 0.0);
    for (std::size_t i = 0; i < kMeasDim; ++i) {
        R(i, i) = effective_sigma_[i] * effective_sigma_[i];
    }
    return R;
}

void ArmorKalmanFilter::initialize(const Mat& z0, const Mat& v0) {
    if (z0.rows() != kMeasDim || z0.cols() != 1) {
        throw std::runtime_error("initialize 需要 4x1 观测");
    }
    x_ = Mat(n_, 1, 0.0);
    for (std::size_t i = 0; i < kMeasDim; ++i) {
        x_(i, 0) = (i == 3) ? wrap_angle(z0(i, 0)) : z0(i, 0);  // yaw 落位到 (-pi, pi]
    }
    if (!v0.empty()) {
        if (v0.rows() != n_ - kPosDim || v0.cols() != 1) {
            throw std::runtime_error("initialize 的速度初值维数与模型不符");
        }
        for (std::size_t i = 0; i + kPosDim < n_; ++i) {
            x_(i + kPosDim, 0) = v0(i, 0);
        }
    }

    P_ = Mat(n_, n_, 0.0);
    P_(0, 0) = P_(1, 1) = p_.p0_pos;
    P_(2, 2) = p_.p0_pos;
    P_(3, 3) = p_.p0_yaw;
    for (std::size_t i = kPosDim; i < kPosDim * 2; ++i) {
        P_(i, i) = p_.p0_vel;
    }
    for (std::size_t i = kPosDim * 2; i < n_; ++i) {
        P_(i, i) = p_.p0_acc;
    }
    initialized_ = true;
}

void ArmorKalmanFilter::adapt_R(const Mat& innovation) {
    // 经验性自适应：用滑动窗口内的新息方差当作观测噪声方差。
    // 严格的自适应 R 需要从新息协方差中减去预测协方差的贡献，这里做了简化，
    // 因此窗口取短、并把结果夹在标称值附近，只用于抑制观测质量的慢漂移。
    std::vector<double> v(kMeasDim, 0.0);
    for (std::size_t i = 0; i < kMeasDim; ++i) {
        v[i] = innovation(i, 0);
    }
    innov_window_.push_back(v);
    if (innov_window_.size() > p_.adaptive_window) {
        innov_window_.erase(innov_window_.begin());
    }
    if (innov_window_.size() < 5) {
        return;
    }

    const double lambda = p_.adaptive_forget;
    for (std::size_t i = 0; i < kMeasDim; ++i) {
        double mean = 0.0;
        for (const auto& w : innov_window_) {
            mean += w[i];
        }
        mean /= static_cast<double>(innov_window_.size());
        double var = 0.0;
        for (const auto& w : innov_window_) {
            const double d = w[i] - mean;
            var += d * d;
        }
        var /= static_cast<double>(innov_window_.size());

        const double nominal = std::pow(effective_sigma_[i], 2);
        const double lo = nominal * std::max(0.05, 1.0 - lambda);
        const double hi = nominal / std::max(0.05, 1.0 - lambda);
        effective_sigma_[i] = std::sqrt(std::min(std::max(var, lo), hi));
    }
}

void ArmorKalmanFilter::step(const Mat& z, double t) {
    // 第 0 帧没有历史可预测，直接用观测建立初始状态。
    // 注意这里不写入 history_，保证 history_ 与输入帧一一对应。
    if (!initialized_) {
        initialize(z);
        return;
    }

    const Mat F = build_F();
    const Mat Q = build_Q();

    // ---------- 1. 预测 ----------
    x_pred_ = F * x_;
    P_pred_ = (F * P_ * F.transpose() + Q).symmetrized();

    // ---------- 2. 构造新息（含角度归一化） ----------
    StepDiag diag;
    bool meas_valid = true;
    for (std::size_t i = 0; i < kMeasDim; ++i) {
        if (!std::isfinite(z(i, 0))) {
            meas_valid = false;
        }
    }
    diag.meas_valid = meas_valid;

    Mat innovation(kMeasDim, 1, 0.0);
    if (meas_valid) {
        for (std::size_t i = 0; i < kMeasDim; ++i) {
            const double pred = x_pred_(i, 0);
            const double raw = z(i, 0) - pred;
            if (i == 3 && p_.yaw_mode != YawMode::Plain) {
                const double wrapped = wrap_angle(raw);
                if (std::abs(wrapped - raw) > 1e-9) {
                    diag.yaw_wrapped = true;  // 该帧发生了 2π 折叠
                }
                innovation(i, 0) = wrapped;
            } else {
                innovation(i, 0) = raw;
            }
        }
    }

    // ---------- 3. 新息统计量、门控与似然 ----------
    const Mat R = build_R();
    bool skip_all = false;
    std::vector<bool> axis_accept(kMeasDim, true);
    if (meas_valid) {
        std::vector<double> axis_nis(kMeasDim, 0.0);
        double nis_total = 0.0;
        for (std::size_t i = 0; i < kMeasDim; ++i) {
            const double s = P_pred_(i, i) + R(i, i);
            axis_nis[i] = (innovation(i, 0) * innovation(i, 0)) / std::max(s, 1e-12);
            nis_total += axis_nis[i];
            // 逐轴高斯对数似然（调参用；门控不参与，避免似然被截断而失真）
            diag.log_likelihood +=
                -0.5 * (std::log(kTwoPi * std::max(s, 1e-12)) + axis_nis[i]);
        }
        diag.nis = nis_total;
        diag.yaw_nis = axis_nis[3];
        total_log_likelihood_ += diag.log_likelihood;

        if (p_.yaw_mode == YawMode::WrapAndGate) {
            for (std::size_t i = 0; i < kMeasDim; ++i) {
                const bool is_yaw_axis = (i == 3);
                if (!is_yaw_axis && !p_.gate_position) {
                    continue;  // 位置轴默认不门控，避免饥饿发散
                }
                if (axis_nis[i] > p_.gate_chi2_axis) {
                    axis_accept[i] = false;
                    ++diag.rejected_axes;
                }
            }
            if (diag.rejected_axes == kMeasDim) {
                skip_all = true;
                diag.gated = true;
            }
            if (!axis_accept[3]) {
                diag.yaw_gated = true;
            }
        }
    } else {
        skip_all = true;
    }

    // ---------- 4. 逐轴标量更新（Joseph 形式，保证 P 正定对称） ----------
    x_ = x_pred_;
    P_ = P_pred_;
    if (!skip_all) {
        for (std::size_t i = 0; i < kMeasDim; ++i) {
            if (!axis_accept[i]) {
                continue;
            }
            // H_i 只在第 i 维取 1，因此 S = P(i,i) + R(i,i)，K = P 的第 i 列 / S
            const double S = P_(i, i) + R(i, i);
            if (S < 1e-12) {
                continue;
            }
            Mat K(n_, 1, 0.0);
            for (std::size_t r = 0; r < n_; ++r) {
                K(r, 0) = P_(r, i) / S;
            }
            const double nu = innovation(i, 0);
            x_ = x_ + K * nu;

            // P = (I - K H) P (I - K H)^T + K R K^T
            Mat IKH = Mat::identity(n_);
            for (std::size_t r = 0; r < n_; ++r) {
                IKH(r, i) -= K(r, 0);
            }
            Mat KRKt = (K * K.transpose()) * R(i, i);
            P_ = (IKH * P_ * IKH.transpose() + KRKt).symmetrized();
        }

        // yaw 状态贴回 (-pi, pi]
        if (p_.yaw_mode != YawMode::Plain) {
            x_(3, 0) = wrap_angle(x_(3, 0));
        }

        if (p_.adaptive_R) {
            adapt_R(innovation);
        }
    }

    // ---------- 5. 记录 ----------
    x_pred_hist_.push_back(x_pred_);
    P_pred_hist_.push_back(P_pred_);

    FrameEstimate fe;
    fe.t = t;
    fe.x_filt = x_;
    fe.P_filt = P_;
    fe.diag = diag;
    history_.push_back(fe);
}

void ArmorKalmanFilter::smooth() {
    const std::size_t n = history_.size();
    if (n < 2) {
        return;
    }

    // 末帧平滑值等于滤波值
    history_[n - 1].x_smooth = history_[n - 1].x_filt;
    history_[n - 1].P_smooth = history_[n - 1].P_filt;

    const Mat F = build_F();

    for (std::size_t k = n - 1; k-- > 0;) {
        const Mat& P_pred_next = P_pred_hist_[k + 1];
        // RTS 增益 G = P_k F^T (P_{k+1|k})^{-1}
        Mat G = (history_[k].P_filt * F.transpose()) * P_pred_next.inverse_spd();

        // 角度状态在相减前必须归一化，否则 2π 误差会污染平滑结果
        const Mat& x_pred_next = x_pred_hist_[k + 1];
        Mat dx = history_[k + 1].x_smooth - x_pred_next;
        dx(3, 0) = angle_diff(history_[k + 1].x_smooth(3, 0), x_pred_next(3, 0));

        history_[k].x_smooth = history_[k].x_filt + G * dx;
        history_[k].P_smooth =
            (history_[k].P_filt + G * (history_[k + 1].P_smooth - P_pred_next) * G.transpose())
                .symmetrized();
        if (p_.yaw_mode != YawMode::Plain) {
            history_[k].x_smooth(3, 0) = wrap_angle(history_[k].x_smooth(3, 0));
        }
    }
}

}  // namespace kalman
