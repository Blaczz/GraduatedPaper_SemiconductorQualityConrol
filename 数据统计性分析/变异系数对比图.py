import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
# 3. 数据加载与计算
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'semiconductor_quality_control.csv')
df = pd.read_csv(data_path)
figures_dir = os.path.join(script_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# 计算各参数变异系数
cv_data = []
for param in NUMERIC_PARAMS:
    data = df[param]
    cv = (data.std() / data.mean()) * 100
    cv_data.append({
        '参数': PARAM_CN[param],
        '变异系数(%)': cv,
        '均值': data.mean(),
        '标准差': data.std()
    })

cv_df = pd.DataFrame(cv_data).sort_values('变异系数(%)', ascending=True)

print("=" * 60)
print("  工艺参数变异系数(CV)对比分析")
print("=" * 60)
print(f"样本总量: {len(df)}")
print(f"\n{'参数':<16s} {'CV(%)':>8s}  {'分级':>10s}")
print("-" * 40)
for _, r in cv_df.iterrows():
    grade = '低变异' if r['变异系数(%)'] < 5 else '中等变异' if r['变异系数(%)'] < 10 else '高变异'
    print(f"{r['参数']:<16s} {r['变异系数(%)']:>8.2f}  {grade:>10s}")

# ==============================================================================
# 4. 绘制变异系数对比图
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#2ecc71' if cv < 5 else '#f39c12' if cv < 10 else '#e74c3c'
          for cv in cv_df['变异系数(%)']]

bars = ax.barh(range(len(cv_df)), cv_df['变异系数(%)'],
               color=colors, alpha=0.7, height=0.7)

# y轴标签
if chinese_font_prop:
    ax.set_yticks(range(len(cv_df)))
    ax.set_yticklabels(cv_df['参数'].values, fontproperties=chinese_font_prop, fontsize=10)
    ax.set_xlabel('变异系数 CV (%)', fontsize=12, fontweight='bold', fontproperties=chinese_font_prop)

else:
    ax.set_yticks(range(len(cv_df)))
    ax.set_yticklabels(cv_df['参数'].values, fontsize=10)
    ax.set_xlabel('CV (%)', fontsize=12, fontweight='bold')
    ax.set_title('CV Comparison', fontsize=14, fontweight='bold', pad=15)

# 数值标签
for i, (_, row) in enumerate(cv_df.iterrows()):
    ax.text(row['变异系数(%)'] + 0.2, i, f"{row['变异系数(%)']:.1f}%",
            va='center', fontsize=9)

# 参考线
ax.axvline(x=5, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='低变异 (<5%)')
ax.axvline(x=10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='中等变异 (5-10%)')

if chinese_font_prop:
    ax.legend(loc='lower right', prop=chinese_font_prop)
else:
    ax.legend(loc='lower right')

ax.grid(True, axis='x', alpha=0.3)
ax.set_xlim(0, cv_df['变异系数(%)'].max() * 1.2)

plt.tight_layout()

output_path = os.path.join(figures_dir, '变异系数对比图.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[保存成功] {output_path}")

# ==============================================================================
# 5. 统计摘要
# ==============================================================================
cv_high = cv_df[cv_df['变异系数(%)'] >= 10]
print(f"\n高变异参数（CV ≥ 10%）：{len(cv_high)} 个")
for _, r in cv_high.iterrows():
    print(f"  · {r['参数']}: CV = {r['变异系数(%)']:.2f}%")
print(f"\n工业工程解读：")
print(f"  平均变异系数: {cv_df['变异系数(%)'].mean():.2f}%")
print(f"  最大变异系数: {cv_df['变异系数(%)'].max():.2f}% ({cv_df.iloc[-1]['参数']})")
print("  高变异参数需优先纳入SPC控制范围，并排查变异来源")
