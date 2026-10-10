#!/usr/bin/env python3
"""生成 yaw 滤波前后对比图（armor_yaw vs target_yaw）。

只做这一张图：
  · 蓝线 armor_yaw  = 原始 PnP 观测（在 ±π 处会跳变，表现为贯穿全图的竖线）
  · 橙线 target_yaw = 卡尔曼滤波输出

中文用工作目录下的 NotoSansCJKsc-Regular.otf 渲染。脚本自带缺字检查：
matplotlib 遇到字体里没有的汉字会静默回退成方框，这里显式查一遍并报出来。
"""
import csv
import math
import os
import sys
import warnings

# 本机装了多份 matplotlib（系统包 + pip 包），Axes3D 导入会失败并打警告。
# 本脚本只用 2D，且警告在 import matplotlib 时就会抛出，所以必须放在导入之前。
warnings.filterwarnings('ignore', message='Unable to import Axes3D')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = os.path.join(ROOT, 'NotoSansCJKsc-Regular.otf')
SRC = os.path.join(ROOT, 'filtered.csv')
OUT = os.path.join(ROOT, 'Result.png')

# 图里用到的全部汉字，用于缺字自检
CJK_USED = '装甲板偏航角滤波结果帧序号弧度原始观测输出'


def setup_font():
    """注册 NotoSansCJKsc-Regular.otf 并设为默认字体。"""
    if not os.path.exists(FONT):
        sys.exit(f'找不到字体文件: {FONT}')
    font_manager.fontManager.addfont(FONT)
    name = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [name]
    plt.rcParams['axes.unicode_minus'] = False  # 负号用 ASCII，避免变成方框
    return name


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def missing_glyphs(font_name, text):
    """返回字体里缺失的字符列表（matplotlib 会静默回退，所以自己查）。"""
    from matplotlib.ft2font import FT2Font
    path = font_manager.findfont(font_manager.FontProperties(family=font_name))
    face = FT2Font(path)
    return [c for c in set(text) if face.get_char_index(ord(c)) == 0]


def main():
    font_name = setup_font()
    if not os.path.exists(SRC):
        sys.exit(f'找不到 {SRC}，请先运行 ./run.sh 生成')

    rows = list(csv.DictReader(open(SRC, encoding='utf-8')))
    if not rows:
        sys.exit(f'{SRC} 里没有数据')

    x = list(range(len(rows)))
    armor = [float(r['raw_yaw']) for r in rows]
    target = [float(r['filt_yaw']) for r in rows]

    fig, ax = plt.subplots(figsize=(7.21, 4.48))
    ax.plot(x, armor, '-', lw=1.0, color='#1f77b4', label='原始观测 armor_yaw')
    ax.plot(x, target, '-', lw=1.5, color='#ff7f0e', label='滤波输出 target_yaw')

    ax.set_title('装甲板偏航角滤波结果')
    ax.set_xlabel('帧序号')
    ax.set_ylabel('偏航角 / 弧度')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, len(rows) - 1)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)

    bad = missing_glyphs(font_name, CJK_USED)
    if bad:
        print(f'  ⚠ 字体缺字（会显示成方框）: {"".join(bad)}', file=sys.stderr)
        return 1

    step = [abs(wrap(target[i + 1] - target[i])) for i in range(len(target) - 1)]
    print(f'{os.path.relpath(OUT, ROOT)}')
    print(f'  字体            : {os.path.basename(FONT)}  →  {font_name}')
    print(f'  帧数            : {len(rows)}')
    print(f'  target 一阶差分 : {sum(step) / len(step):.5f} 弧度/帧')
    print(f'  target 范围     : [{min(target):+.4f}, {max(target):+.4f}] 弧度')
    return 0


if __name__ == '__main__':
    sys.exit(main())
