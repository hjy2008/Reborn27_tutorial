#!/usr/bin/env bash
#
# 作业六 · 卡尔曼滤波（装甲板位姿）—— 一键运行
#
# 做三件事：编译 → 滤波 → 出图。
# 完成后看 figs/fig5_yaw_ekf_style.png 和 summary.txt。
#
# 用法：
#   ./run.sh                 默认参数
#   ./run.sh --sigma-a-yaw 5 透传任意滤波参数（见 ./l6 --help）
#
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

CXX=${CXX:-g++}
SRC="main.cpp src/kalman.cpp src/data.cpp"
BIN=./l6
ARGS=("$@")   # 未传参数时为空数组，l6 会走默认值

# ---------- 1. 环境检查 ----------
command -v "$CXX" >/dev/null 2>&1 || { echo "错误: 找不到编译器 $CXX" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "错误: 找不到 python3（出图需要）" >&2; exit 1; }
python3 -c 'import matplotlib' 2>/dev/null || {
    echo "错误: python3 缺少 matplotlib，无法出图。可执行: pip3 install matplotlib" >&2; exit 1; }
[ -f armor_data.csv ] || { echo "错误: 找不到 armor_data.csv" >&2; exit 1; }
[ -f NotoSansCJKsc-Regular.otf ] || {
    echo "错误: 找不到 NotoSansCJKsc-Regular.otf（中文绘图需要）" >&2; exit 1; }

# ---------- 2. 编译 ----------
echo "[1/3] 编译"
"$CXX" -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude $SRC -o "$BIN"

# ---------- 3. 滤波 ----------
echo "[2/3] 卡尔曼滤波"
if [ ${#ARGS[@]} -eq 0 ]; then
    "$BIN" --summary summary.txt --out filtered.csv
else
    "$BIN" --summary summary.txt --out filtered.csv "${ARGS[@]}"
fi

# ---------- 4. 出图 ----------
echo
echo "[3/3] 生成图表"
# matplotlib 默认往 ~/.config 写缓存，某些环境不可写；这里固定到工作目录内
MPLCONFIGDIR="$(pwd)/.mplcache" python3 tools/plot.py

echo
echo "完成。"
echo "  图表    : Result.png"
echo "  逐帧数据: filtered.csv"
echo "  指标摘要: summary.txt"
