import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
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
print("  图表2：参数分布箱线图（按缺陷状态分组）")
print("=" * 60)
print(f"样本总量: {len(df)}")

# ==============================================================================
# 4. 绘制箱线图（2×5子图布局）
# ==============================================================================
fig, axes = plt.subplots(2, 5, figsize=(20, 12))
axes = axes.flatten()

for i, param in enumerate(NUMERIC_PARAMS):
    ax = axes[i]
    param_cn = PARAM_CN[param]

    # 绘制箱线图
    bp = ax.boxplot(
        [df[df['Defect'] == 0][param], df[df['Defect'] == 1][param]],
        patch_artist=True,
        labels=['正常', '缺陷'],
        medianprops=dict(color='red', linewidth=2),
        boxprops=dict(facecolor='lightblue', alpha=0.7),
        whiskerprops=dict(color='gray'),
        capprops=dict(color='gray'),
        flierprops=dict(marker='o', markerfacecolor='orange', markersize=3, alpha=0.5)
    )

    # 设置x轴标签字体
    if chinese_font_prop:
        ax.set_xticklabels(['正常', '缺陷'], fontproperties=chinese_font_prop)

    # 设置标题和标签 — 子图编号 (a)(b)...(j)
    if chinese_font_prop:
        ax.set_title(f'({chr(97 + i)}) {param_cn}',
                     fontweight='bold', fontsize=10, fontproperties=chinese_font_prop)
        ax.set_ylabel('数值', fontproperties=chinese_font_prop)
        ax.set_xlabel('批次类型', fontproperties=chinese_font_prop)
    else:
        ax.set_title(f'({chr(97 + i)}) {param_cn}', fontweight='bold', fontsize=10)
        ax.set_ylabel('Value')

    ax.grid(True, alpha=0.3)

# 调整子图间距
fig.subplots_adjust(hspace=0.3, wspace=0.4)

# 保存图表
output_path = os.path.join(figures_dir, '参数分布箱线图.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[保存成功] {output_path}")

