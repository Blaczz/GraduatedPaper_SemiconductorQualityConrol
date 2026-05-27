import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 中文字体配置
# ==============================================================================
chinese_font_prop = None

font_paths = [
    (r'C:\Windows\Fonts\msyh.ttc', 'Microsoft YaHei'),
    (r'C:\Windows\Fonts\simhei.ttf', 'SimHei'),
    (r'C:\Windows\Fonts\simsun.ttc', 'SimSun'),
    (r'C:\Windows\Fonts\kaiti.ttf', 'KaiTi'),
    (r'C:\Windows\Fonts\fs.ttf', 'FangSong'),
]

for font_path, font_name in font_paths:
    if os.path.exists(font_path):
        try:
            chinese_font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [font_name, 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"[字体] 成功加载: {font_name}")
            break
        except Exception:
            continue

if chinese_font_prop is None:
    print("[警告] 未找到合适的中文字体")

sns.set_style("whitegrid")

# ==============================================================================
# 2. 参数定义
# ==============================================================================
NUMERIC_PARAMS = [
    'Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
    'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
    'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count'
]

PARAM_CN = {
    'Chamber_Temperature': '腔体温度',
    'Gas_Flow_Rate': '气体流量',
    'RF_Power': '射频功率',
    'Etch_Depth': '刻蚀深度',
    'Rotation_Speed': '旋转速度',
    'Vacuum_Pressure': '真空压力',
    'Stage_Alignment_Error': '晶圆台对准误差',
    'Vibration_Level': '振动级别',
    'UV_Exposure_Intensity': 'UV曝光强度',
    'Particle_Count': '颗粒数'
}

# ==============================================================================
# 3. 数据加载
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'semiconductor_quality_control.csv')
df = pd.read_csv(data_path)
figures_dir = os.path.join(script_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

print("=" * 60)
print("  参数分布直方图（与正态分布对比）")
print("=" * 60)
print(f"样本总量: {len(df)}")

# ==============================================================================
# 4. 正态性检验（Shapiro-Wilk）
# ==============================================================================
print("\n【正态性检验结果（Shapiro-Wilk）】")
print("-" * 60)

normality_results = []
for param in NUMERIC_PARAMS:
    sample_data = df[param].values[:5000]
    w_stat, p_value = stats.shapiro(sample_data)
    is_normal = "是" if p_value > 0.05 else "否"

    normality_results.append({
        '参数': PARAM_CN[param],
        '偏度': df[param].skew(),
        '峰度': df[param].kurtosis(),
        'W统计量': w_stat,
        'p值': p_value,
        '正态分布': is_normal
    })

    print(f"  {PARAM_CN[param]:<12s}: 偏度={df[param].skew():7.3f}, "
          f"峰度={df[param].kurtosis():7.3f}, p={p_value:.4f} -> {is_normal}")

normality_df = pd.DataFrame(normality_results)
normal_count = len(normality_df[normality_df['正态分布'] == '是'])

# ==============================================================================
# 5. 绘制直方图（2×5子图布局）
# ==============================================================================
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for i, param in enumerate(NUMERIC_PARAMS):
    ax = axes[i]
    data = df[param]
    param_cn = PARAM_CN[param]

    # 直方图 + KDE
    sns.histplot(data, kde=True, ax=ax, color='steelblue', alpha=0.6, edgecolor='white')

    # 正态分布拟合曲线
    x_range = np.linspace(data.min(), data.max(), 100)
    bin_width = (data.max() - data.min()) / 20
    normal_curve = stats.norm.pdf(x_range, data.mean(), data.std())
    ax.plot(x_range, normal_curve * len(data) * bin_width,
            'r--', linewidth=2, label='正态拟合')

    # 标题 — 子图编号 (a)(b)...(j)
    if chinese_font_prop:
        ax.set_title(f'({chr(97 + i)}) {param_cn}',
                     fontweight='bold', fontsize=11, fontproperties=chinese_font_prop)
        ax.set_xlabel('数值', fontproperties=chinese_font_prop)
        ax.set_ylabel('频数', fontproperties=chinese_font_prop)
        ax.legend(prop=chinese_font_prop, fontsize=8)
    else:
        ax.set_title(f'({chr(97 + i)}) {param_cn}', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)

    # 偏度/峰度标注
    skew = data.skew()
    kurt = data.kurtosis()
    text_kw = dict(transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    if chinese_font_prop:
        ax.text(0.97, 0.95, f'偏度:{skew:.2f}\n峰度:{kurt:.2f}',
                fontproperties=chinese_font_prop, **text_kw)
    else:
        ax.text(0.97, 0.95, f'Skew:{skew:.2f}\nKurt:{kurt:.2f}', **text_kw)

    ax.grid(True, alpha=0.3)


plt.tight_layout(rect=[0, 0.02, 1, 0.95])

# 保存
output_path = os.path.join(figures_dir, '参数分布直方图.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[保存成功] {output_path}")

# ==============================================================================
# 6. 统计摘要
# ==============================================================================
print("\n" + "-" * 60)
print(f"正态性检验汇总：{normal_count}/10 个参数通过Shapiro-Wilk检验（p > 0.05）")
print("-" * 60)
print("工业工程解读：")
print("  偏度(Skewness) ≈ 0 → 对称分布；峰度(Kurtosis) ≈ 0 → 与正态分布相似")
print("  多数参数近似正态，支持SPC X-MR控制图；偏离参数需考虑数据变换")
