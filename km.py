import matplotlib
matplotlib.use('Agg')  # Termux 无图形界面，必须使用非交互后端
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

def evaluate(x, z, true, label=""):
    x, z, true = np.array(x), np.array(z), np.array(true)
    rmse_z    = np.sqrt(np.mean((z - true) ** 2))
    rmse_x    = np.sqrt(np.mean((x - true) ** 2))
    mae_z     = np.mean(np.abs(z - true))
    mae_x     = np.mean(np.abs(x - true))
    std_z     = np.std(z)
    std_x     = np.std(x)

    # print(f"===== 滤波评价 {label} =====")
    # print(f"RMSE : 测量={rmse_z:.2f}  滤波={rmse_x:.2f}  "
    #       f"改善={100*(rmse_z-rmse_x)/rmse_z:.1f}%")
    # print(f"MAE  : 测量={mae_z:.2f}  滤波={mae_x:.2f}  "
    #       f"改善={100*(mae_z-mae_x)/mae_z:.1f}%")
    # print(f"平滑: 测量std={std_z:.2f}  滤波std={std_x:.2f}  "
    #       f"平滑度提升={100*(std_z-std_x)/std_z:.1f}%")

    # 简易打分（综合误差与平滑度）
    score = (rmse_x / rmse_z) + (std_x / std_z)
    # print(f"综合得分 score={score:.3f}  (越小越好)\n")
    return score

# ==================== 1. 设置本地中文字体（绝对路径） ====================
FONT_PATH = './NotoSansCJKsc-Regular.otf'
# 如果路径不对，尝试使用：
# FONT_PATH = '/data/data/com.termux/files/usr/share/fonts/wqy-zenhei.ttc'

try:
    font_manager.fontManager.addfont(FONT_PATH)
    font_prop = font_manager.FontProperties(fname=FONT_PATH)
    FONT_NAME = font_prop.get_name()
    print(f'成功加载字体: {FONT_NAME}')
except Exception as e:
    print(f'字体加载失败: {e}')
    FONT_NAME = 'DejaVu Sans'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [FONT_NAME, 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 2. 核心：把 C++ 逻辑 + 绘图封装成函数 ====================
def km_filter_and_plot(a=0.6, b=0.4, t=5.0,
                       save_path=None, show=False, draw=True):
    """
    Kalman 滤波（简化的 α-β 滤波）并绘制结果。

    参数:
        a        : 位置修正系数
        b        : 速度修正系数
        t        : 采样周期
        save_path: 保存图片路径，若为 None 则自动生成 'out_a{a}_b{b}.jpg'
        show     : 是否显示图像（Termux 下无图形界面，一般保持 False）
    """

    z = [30171, 30353, 30759, 30799, 31018, 31278, 31276, 31379, 31748, 32175]

    x = [0.0] * 10       # 估计值
    v = [0.0] * 10       # 速度估计值
    x_pred = [0.0] * 10  # 预测值
    v_pred = [0.0] * 10  # 预测速度

    x[0] = z[0]
    v[0] = 40

    for i in range(1, 10):
        x_pred_ = x[i - 1] + v[i - 1] * t
        v_pred_ = v[i - 1]

        x_pred[i] = x_pred_
        v_pred[i] = v_pred_

        x[i] = x_pred_ + a * (z[i] - x_pred_)
        v[i] = v_pred_ + b * (z[i] - x_pred_) / t

    # ---------- 2.2 生成 out.csv（保持与 C++ 输出一致） ----------
    with open('out.csv', 'w', encoding='utf-8-sig') as f:
        f.write('n,' + ','.join(str(i) for i in range(1, 11)) + '\n')
        f.write('z,' + ','.join(f'{val:.2f}' for val in z) + '\n')
        real = list(range(30200, 32001, 200))
        f.write('真实值,' + ','.join(str(val) for val in real) + '\n')
        f.write('x,' + ','.join(f'{val:.2f}' for val in x) + '\n')
        f.write('v,' + ','.join(f'{val:.2f}' for val in v) + '\n')
        f.write('x_pred,' + ','.join(f'{val:.2f}' for val in x_pred) + '\n')
        f.write('v_pred,' + ','.join(f'{val:.2f}' for val in v_pred) + '\n')

    # ---------- 2.3 读取 out.csv ----------
    data = {}
    with open('out.csv', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip().rstrip(',')
            if not line:
                continue
            parts = line.split(',')
            key = parts[0].strip()
            values = [float(p) for p in parts[1:] if p.strip() != '']
            data[key] = values

    n = data['n']

    # ---------- 2.4 绘图 ----------
    real = list(range(30200, 32001, 200))
    score = evaluate(x, z, real, label=f"a={a}, b={b}")
    
    if draw:
        fig, ax = plt.subplots(figsize=(10, 6))
    
        ax.plot(n, data['真实值'], color='green', marker='d',
                linestyle='-', linewidth=2, markersize=7, label='真实值')
        ax.plot(n, data['z'], color='blue', marker='s',
                linestyle='-', linewidth=2, markersize=6, label='测量值 (z)')
        ax.plot(n, data['x'], color='red', marker='o',
                linestyle='-', linewidth=2, markersize=6, label='估计值 (x)')
        ax.plot(n, data['x_pred'], color='gray', marker='^',
                linestyle='-', linewidth=2, markersize=7, label='预测值 (x_pred)')
    
        # 标题带上 a、b 的值
        ax.set_title(f'距离与时间的关系  (a={a}, b={b})',
                     color='red', fontweight='bold', fontsize=14, pad=15)
    
        # 在图中再标注 a、b 的数值（左上角文字），保证一眼能看到
        ax.text(0.99, 0.02, f'a = {a}\nb = {b}',
                transform=ax.transAxes,
                fontsize=11, color='darkred', fontweight='bold',
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='#fffbe6', edgecolor='red', alpha=0.9))
    
        ax.set_xlabel('时间 (s)', color='red', fontsize=12)
        ax.set_ylabel('距离 (m)', color='red', fontsize=12)
    
        ax.legend(loc='upper left', frameon=True, edgecolor='gray', fontsize=10)
        ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', colors='black', labelsize=10)
    
        # ---------- 2.5 Y 轴不从 0 开始，过滤掉 0 值 ----------
        all_y_values = data['真实值'] + data['z'] + data['x'] + data['x_pred']
        valid_y_values = [val for val in all_y_values if val > 0]
    
        y_min, y_max = min(valid_y_values), max(valid_y_values)
        y_margin = (y_max - y_min) * 0.05
        ax.set_ylim(bottom=y_min - y_margin, top=y_max + y_margin)
    
        ax.set_xlim(left=1, right=10)
    
        # ---------- 2.6 保存 ----------
        if save_path is None:
            save_path = f'out_a{a}_b{b}.jpg'
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'已保存为 {save_path}  (a={a}, b={b})')
    
        if show:
            plt.show()
        plt.close(fig)

    return x, v, x_pred, v_pred, score  # 顺手把结果返回，方便后续分析


# ==================== 3. 调用示例 ====================
if __name__ == '__main__':
    results = []
    for i in range(1, 100):
        for j in range(1, 100):
            _, _, _, _, s = km_filter_and_plot(a=i/100, b=j/100, draw=False)
            results.append([i/100, j/100, s])
    
    best = min(results, key=lambda r: r[2])
    print(f"最优参数: a={best[0]}, b={best[1]}, score={best[2]:.4f}")
    
    km_filter_and_plot(best[0], best[1])
    
    # ---- 2. 按 score 排序打印 Top 10 ----
    print("\nTop 10 最优参数：")
    for rank, (a, b, s) in enumerate(sorted(results, key=lambda r: r[2])[:10], 1):
        print(f"  {rank:>2}. a={a}, b={b}, score={s:.4f}")
    
    
                
