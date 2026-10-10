// 作业七：单方向空气阻力模型求解炮台出射角（C++ 版，与 hw7_ballistic.m 完全等价）
// 模型：竖直方向只受重力 z'' = -g；水平方向阻力 f = k*v_s^2（沿炮管方向反向）
//   v_s(t) = v0*cos(theta)/(1+k*v0*cos(theta)*t)
//   x(t)   = ln(1+k*v0*cos(theta)*t)/k                   
// 编译：g++ -O2 -o hw7 hw7_ballistic.cpp && ./hw7
#include <cstdio>
#include <cmath>
#include <initializer_list>

static const double v0 = 17.0;      // 初速度 m/s
static const double k  = 0.0089;    // 阻力系数 1/m
static const double g  = 9.8;       // 重力加速度
static const double PI = 3.14159265358979323846;

// 由式(1) 反解飞行时间：x = ln(1+k*v0*cos(theta)*t)/k  ->  t
// 注意：x(t)=(1/k)ln(1+k v0 cos(theta) t)，ln 内是 k*v0*cos(theta)*t，
//       不含 c 的指数项，故 t = (e^{k*x}-1)/(k*v0*cos(theta))
static double flightTime(double theta, double x) {
    double c = std::cos(theta);
    double L = k * x;                              // = k*s
    if (L > 40) return INFINITY;                   // 防溢出（theta -> 90deg 时指数发散）
    return std::expm1(L) / (k * v0 * c);
}

// 给定出射角，返回飞过水平距离 s 时的高度
static double zLand(double theta, double s) {
    double c = std::cos(theta);
    if (c <= 1e-12) return INFINITY;
    double t = flightTime(theta, s);
    return v0 * std::sin(theta) * t - 0.5 * g * t * t;
}

int main() {
    const double x = 3.0, y = 4.0, z0 = 0.25;
    const double s = std::sqrt(x * x + y * y);        // 水平距离
    const double yaw = std::atan2(y, x);

    printf("水平距离 s = %.6f m, 方位角 yaw = %.6f deg\n\n", s, yaw * 180 / PI);

    auto F = [&](double th) { return zLand(th, s) - z0; };

    // ---------- 1. 解的存在性与唯一性 ----------
    printf("--- 存在性 / 多解判断 ---\n");
    printf("  F(0 deg)   = %+10.5f m   (<0 打低)\n", F(0.0));
    printf("  F(80 deg)  = %+10.5f m   (>0 打高)\n", F(80 * PI / 180));
    for (double d : {0.0, 30.0, 60.0, 80.0})
        printf("  dZ/dtheta @ %4.0f deg = %8.4f m/rad\n", d,
               (zLand(d * PI / 180 + 1e-7, s) - zLand(d * PI / 180 - 1e-7, s)) / 2e-7);
    printf("  (在 [0,80] 内恒 > 0 -> 落点高度随出射角单调上升 -> 唯一解)\n");
    printf("  theta -> 90deg 时 t=(e^{ks}-1)/(k v0 cos) 随 1/cos 发散，落点 -> -inf，\n");
    printf("  89~90deg 是角秒量级的病态区，求解区间取 [0,80deg] 即可。\n\n");

    // ---------- 2. 讲义迭代法 ----------
    printf("--- 迭代补偿过程 ---\n");
    printf(" %2s %12s %12s %12s %12s\n", "i", "pitch(deg)", "z_real(m)", "dz(m)", "z_temp(m)");
    double zt = z0;
    for (int i = 1; i <= 10; ++i) {
        double th = std::atan2(zt, s);
        double zr = zLand(th, s);
        double dz = z0 - zr;
        printf(" %2d %12.6f %12.7f %12.7f %12.7f\n", i, th * 180 / PI, zr, dz, zt + dz);
        zt += dz;
    }
    double thIter = std::atan2(zt, s);

    printf("\n--- 数值解 ---\n");
    printf(" 迭代法 theta = %.6f deg = %.8f rad\n", thIter * 180 / PI, thIter);
    return 0;
}
