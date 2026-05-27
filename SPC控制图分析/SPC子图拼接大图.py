"""
SPC控制图子图拼接 — 10个工艺参数X图拼为大图 + MR图拼为大图
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'semiconductor_quality_control.csv')
figures_dir = os.path.join(script_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

df = pd.read_csv(data_path)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)
print(f"数据加载: {df.shape[0]}行")

PARAMS = [
    'Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
    'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
    'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count'
]

PARAM_CN_SHORT = {
    'Chamber_Temperature':   '腔体温度',
    'RF_Power':              '射频功率',
    'Gas_Flow_Rate':         '气体流量',
    'Etch_Depth':            '刻蚀深度',
    'Particle_Count':        '颗粒数',
    'Rotation_Speed':        '旋转速度',
    'Vacuum_Pressure':       '真空压力',
    'Stage_Alignment_Error': '晶圆台对准误差',
    'Vibration_Level':       '振动级别',
    'UV_Exposure_Intensity': 'UV曝光强度'
}

PARAM_UNIT = {
    'Chamber_Temperature':   '℃',
    'RF_Power':              'W',
    'Gas_Flow_Rate':         'sccm',
    'Etch_Depth':            'nm',
    'Particle_Count':        '个/cm^3',
    'Rotation_Speed':        'rpm',
    'Vacuum_Pressure':       'Pa',
    'Stage_Alignment_Error': 'μm',
    'Vibration_Level':       'g',
    'UV_Exposure_Intensity': 'mW/cm^2'
}

# 提前计算所有参数的控制限
spc_data = {}
for param in PARAMS:
    values = df[param].values
    n = len(values)
    mr = np.abs(np.diff(values))
    mr_cl = np.mean(mr)
    x_cl = np.mean(values)
    x_ucl = x_cl + 2.66 * mr_cl
    x_lcl = x_cl - 2.66 * mr_cl
    mr_ucl = 3.267 * mr_cl
    outliers_x = (values > x_ucl) | (values < x_lcl)
    outliers_mr = mr > mr_ucl
    spc_data[param] = {
        'values': values, 'mr': mr, 'n': n,
        'x_cl': x_cl, 'x_ucl': x_ucl, 'x_lcl': x_lcl,
        'mr_cl': mr_cl, 'mr_ucl': mr_ucl,
        'outliers_x': outliers_x, 'outliers_mr': outliers_mr
    }

# ==================== 图1：10个参数的X图子图拼接 ====================
fig, axes = plt.subplots(5, 2, figsize=(22, 26))
axes = axes.flatten()

for idx, param in enumerate(PARAMS):
    ax = axes[idx]
    sd = spc_data[param]
    timestamps = df['Timestamp']

    ax.plot(timestamps, sd['values'], 'o-', color='#3498db',
            markersize=2.0, linewidth=0.8, alpha=0.6)

    ax.axhline(y=sd['x_ucl'], color='red', ls='--', lw=1.5,
               label=f'UCL={sd["x_ucl"]:.2f}')
    ax.axhline(y=sd['x_cl'], color='green', ls='-', lw=1.5,
               label=f'CL={sd["x_cl"]:.2f}')
    ax.axhline(y=sd['x_lcl'], color='red', ls='--', lw=1.5,
               label=f'LCL={sd["x_lcl"]:.2f}')

    ax.scatter(timestamps[sd['outliers_x']], sd['values'][sd['outliers_x']],
               color='red', s=25, zorder=5, label='异常点')

    unit = PARAM_UNIT[param]
    ax.set_title(f'({chr(97+idx)}) {PARAM_CN_SHORT[param]} ({unit})',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('观测值', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

fig.suptitle('X图（单值控制图）— 10个工艺参数汇总',
             fontsize=18, fontweight='bold', y=0.998)
plt.tight_layout(rect=[0, 0, 1, 0.995])
x_save_path = os.path.join(figures_dir, 'SPC_X图_10参数汇总大图.png')
plt.savefig(x_save_path, dpi=400, bbox_inches='tight')
plt.close()
print(f"X图大图已保存: {x_save_path}")

# ==================== 图2：10个参数的MR图子图拼接 ====================
fig, axes = plt.subplots(5, 2, figsize=(22, 26))
axes = axes.flatten()

for idx, param in enumerate(PARAMS):
    ax = axes[idx]
    sd = spc_data[param]
    timestamps = df['Timestamp']

    ax.plot(timestamps[1:], sd['mr'], 'o-', color='#e67e22',
            markersize=2.0, linewidth=0.8, alpha=0.6)

    ax.axhline(y=sd['mr_ucl'], color='red', ls='--', lw=1.5,
               label=f'UCL={sd["mr_ucl"]:.2f}')
    ax.axhline(y=sd['mr_cl'], color='green', ls='-', lw=1.5,
               label=f'CL={sd["mr_cl"]:.2f}')
    ax.axhline(y=0, color='red', ls='--', lw=1.5, label='LCL=0')

    ax.scatter(timestamps[1:][sd['outliers_mr']], sd['mr'][sd['outliers_mr']],
               color='red', s=25, zorder=5, label='异常点')

    unit = PARAM_UNIT[param]
    ax.set_title(f'({chr(97+idx)}) {PARAM_CN_SHORT[param]} ({unit})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('时间', fontsize=10)
    ax.set_ylabel('移动极差', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

fig.suptitle('MR图（移动极差控制图）— 10个工艺参数汇总',
             fontsize=18, fontweight='bold', y=0.998)
plt.tight_layout(rect=[0, 0, 1, 0.995])
mr_save_path = os.path.join(figures_dir, 'SPC_MR图_10参数汇总大图.png')
plt.savefig(mr_save_path, dpi=400, bbox_inches='tight')
plt.close()
print(f"MR图大图已保存: {mr_save_path}")

print("\n完成！生成2张大图：")
print(f"  1. X图汇总大图: {x_save_path}")
print(f"  2. MR图汇总大图: {mr_save_path}")
