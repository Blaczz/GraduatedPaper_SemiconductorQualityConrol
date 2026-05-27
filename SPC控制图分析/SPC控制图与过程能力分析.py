import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

PARAM_CN = {
    'Chamber_Temperature':   '腔体温度 (℃)',
    'RF_Power':              '射频功率 (W)',
    'Gas_Flow_Rate':         '气体流量 (sccm)',
    'Etch_Depth':            '刻蚀深度 (nm)',
    'Particle_Count':        '颗粒数 (个/cm³)',
    'Rotation_Speed':        '旋转速度 (rpm)',
    'Vacuum_Pressure':       '真空压力 (Pa)',
    'Stage_Alignment_Error': '晶圆台对准误差 (μm)',
    'Vibration_Level':       '振动级别 (g)',
    'UV_Exposure_Intensity': 'UV曝光强度 (mW/cm^2)'
}

PARAM_CN_SHORT = {k: v.split(' (')[0] for k, v in PARAM_CN.items()}

def plot_spc_chart(data, param, save_path):
    """X-MR控制图，基于AIAG SPC标准"""
    values = data[param].values
    n = len(values)

    # 移动极差
    mr = np.abs(np.diff(values))
    mr_cl = np.mean(mr)

    # X图控制限（基于MR，d2=1.128，n=2）
    x_cl = np.mean(values)
    x_ucl = x_cl + 2.66 * mr_cl
    x_lcl = x_cl - 2.66 * mr_cl

    # MR图控制限
    mr_ucl = 3.267 * mr_cl

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # X图
    ax1.plot(data['Timestamp'], values, 'o-', color='#3498db',
             markersize=4, linewidth=1, alpha=0.7, label='观测值')
    ax1.axhline(y=x_ucl, color='red', ls='--', lw=2, label=f'UCL={x_ucl:.3f}')
    ax1.axhline(y=x_cl, color='green', ls='-', lw=2, label=f'CL={x_cl:.3f}')
    ax1.axhline(y=x_lcl, color='red', ls='--', lw=2, label=f'LCL={x_lcl:.3f}')

    outliers_x = (values > x_ucl) | (values < x_lcl)
    ax1.scatter(data['Timestamp'][outliers_x], values[outliers_x],
                color='red', s=100, zorder=5, label='异常点')
    ax1.set_ylabel('观测值', fontsize=11)
    ax1.set_title(f'X图（单值控制图）— {PARAM_CN.get(param, param)}',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # MR图
    ax2.plot(data['Timestamp'][1:], mr, 'o-', color='#e67e22',
             markersize=4, linewidth=1, alpha=0.7, label='移动极差')
    ax2.axhline(y=mr_ucl, color='red', ls='--', lw=2, label=f'UCL={mr_ucl:.3f}')
    ax2.axhline(y=mr_cl, color='green', ls='-', lw=2, label=f'CL={mr_cl:.3f}')
    ax2.axhline(y=0, color='red', ls='--', lw=2, label='LCL=0')

    outliers_mr = mr > mr_ucl
    ax2.scatter(data['Timestamp'][1:][outliers_mr], mr[outliers_mr],
                color='red', s=100, zorder=5, label='异常点')
    ax2.set_xlabel('时间', fontsize=11)
    ax2.set_ylabel('移动极差', fontsize=11)
    ax2.set_title(f'MR图（移动极差控制图）— {PARAM_CN.get(param, param)}',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)

    fig.suptitle(f'SPC控制图分析 — {PARAM_CN.get(param, param)}',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'parameter': param,
        'n': n,
        'mean': x_cl, 'std': np.std(values),
        'x_ucl': x_ucl, 'x_lcl': x_lcl,
        'mr_cl': mr_cl, 'mr_ucl': mr_ucl,
        'outliers_x': int(np.sum(outliers_x)),
        'outliers_mr': int(np.sum(outliers_mr)),
        'outlier_rate_x': np.sum(outliers_x) / n * 100,
        'outlier_rate_mr': np.sum(outliers_mr) / (n - 1) * 100
    }


def calc_capability(data, param):
    """计算过程能力指数Cp/Cpk/Cpm，规格限μ±3σ"""
    values = data[param].values
    mean = np.mean(values)
    std = np.std(values, ddof=1)

    usl = mean + 3 * std
    lsl = mean - 3 * std

    if std > 0:
        cp = (usl - lsl) / (6 * std)
        cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
        target = (usl + lsl) / 2
        cpm = (usl - lsl) / (6 * np.sqrt(np.mean((values - target) ** 2)))
        defect_rate = (1 - stats.norm.cdf(usl, mean, std)) + stats.norm.cdf(lsl, mean, std)
        ppm = defect_rate * 1e6
    else:
        cp = cpk = cpm = defect_rate = ppm = np.nan

    if cpk >= 1.67:
        level, color = '优秀', 'green'
    elif cpk >= 1.33:
        level, color = '良好', '#27ae60'
    elif cpk >= 1.00:
        level, color = '尚可', '#f39c12'
    elif cpk >= 0.67:
        level, color = '不足', '#e67e22'
    else:
        level, color = '严重不足', 'red'

    return {
        'parameter': param, 'n': len(values),
        'mean': mean, 'std': std, 'usl': usl, 'lsl': lsl,
        'cp': cp, 'cpk': cpk, 'cpm': cpm,
        'defect_rate': defect_rate, 'ppm': ppm,
        'level': level, 'color': color
    }


print("\n>>> SPC控制图与过程能力分析（10个工艺参数）")

spc_results = []
cap_results = []

for i, param in enumerate(PARAMS):
    print(f"  [{i+1}/10] {PARAM_CN_SHORT[param]} ...")

    save_path = os.path.join(figures_dir, f'SPC_{param}_控制图.png')
    spc = plot_spc_chart(df, param, save_path)
    spc_results.append(spc)

    cap = calc_capability(df, param)
    cap_results.append(cap)

    print(f"        X图异常点={spc['outliers_x']}, "
          f"MR图异常点={spc['outliers_mr']}, "
          f"Cpk={cap['cpk']:.3f} ({cap['level']})")

print("\n>>> 生成汇总图表")

fig, ax = plt.subplots(figsize=(14, 8))
params_cn = [PARAM_CN_SHORT[r['parameter']] for r in cap_results]
cp_vals = [r['cp'] for r in cap_results]
cpk_vals = [r['cpk'] for r in cap_results]
colors = [r['color'] for r in cap_results]

x = np.arange(len(params_cn))
w = 0.35

ax.bar(x - w/2, cp_vals, w, label='Cp（潜在能力）',
       color='skyblue', alpha=0.7, edgecolor='black')
ax.bar(x + w/2, cpk_vals, w, label='Cpk（实际能力）',
       color=colors, alpha=0.7, edgecolor='black')

for hline, clr, lbl in [(1.67, 'green', '优秀 ≥1.67'),
                         (1.33, '#27ae60', '良好 ≥1.33'),
                         (1.00, '#f39c12', '尚可 ≥1.00'),
                         (0.67, '#e67e22', '不足 ≥0.67')]:
    ax.axhline(y=hline, color=clr, ls='--', lw=2, label=lbl)

ax.set_xticks(x)
ax.set_xticklabels(params_cn, rotation=45, ha='right')
ax.set_xlabel('工艺参数', fontsize=12, fontweight='bold')
ax.set_ylabel('过程能力指数', fontsize=12, fontweight='bold')
ax.set_title('工艺参数过程能力对比（规格限 μ ± 3σ）',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '05_过程能力对比图.png'), dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(params_cn))
ax.bar(x - w/2, [r['outliers_x'] for r in spc_results], w,
       label='X图异常点', color='#3498db', alpha=0.7, edgecolor='black')
ax.bar(x + w/2, [r['outliers_mr'] for r in spc_results], w,
       label='MR图异常点', color='#e74c3c', alpha=0.7, edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(params_cn, rotation=45, ha='right')
ax.set_xlabel('工艺参数', fontsize=12, fontweight='bold')
ax.set_ylabel('异常点数量', fontsize=12, fontweight='bold')
ax.set_title('SPC控制图异常点统计（Nelson规则：超出UCL/LCL）',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '06_SPC异常点统计图.png'), dpi=300, bbox_inches='tight')
plt.close()

# ===== 表3.4 SPC分析结果汇总表 =====
spc_df = pd.DataFrame(spc_results)
spc_df = spc_df[['parameter', 'outliers_x', 'outlier_rate_x',
                 'outliers_mr', 'outlier_rate_mr',
                 'x_ucl', 'x_lcl', 'mean', 'std']]
spc_df.columns = ['工艺参数', 'X图异常点', 'X图异常率',
                  'MR图异常点', 'MR图异常率',
                  'USL', 'LSL', 'μ', 'σ']
spc_df['工艺参数'] = spc_df['工艺参数'].map(PARAM_CN_SHORT)
spc_df['X图异常率'] = spc_df['X图异常率'].apply(lambda x: f'{x:.2f}%')
spc_df['MR图异常率'] = spc_df['MR图异常率'].apply(lambda x: f'{x:.2f}%')
spc_df[['USL', 'LSL', 'μ', 'σ']] = spc_df[['USL', 'LSL', 'μ', 'σ']].round(3)

excel_path = os.path.join(figures_dir, '表3.4_SPC分析结果汇总表.xlsx')
spc_df.to_excel(excel_path, index=False, sheet_name='SPC分析结果汇总')
print(f"\n表3.4已保存: {excel_path}")

print("\n" + "=" * 70)
print("SPC分析与过程能力评估摘要")
print("=" * 70)

cap_df = pd.DataFrame(cap_results)
cap_df = cap_df[['parameter', 'cp', 'cpk', 'cpm', 'level', 'ppm']]
cap_df.columns = ['参数', 'Cp', 'Cpk', 'Cpm', '评级', '预期PPM']

for _, r in cap_df.iterrows():
    print(f"  {PARAM_CN_SHORT[r['参数']]:<10} Cp={r['Cp']:.3f}  Cpk={r['Cpk']:.3f}  "
          f"Cpm={r['Cpm']:.3f}  {r['评级']}  PPM={r['预期PPM']:.0f}")

poor = cap_df[cap_df['Cpk'] < 1.33]
if len(poor) > 0:
    print(f"\n  Cpk < 1.33（需改进）：{len(poor)} 个参数")

print(f"\n输出文件：12张图表 + 1个Excel")
print(f"图表目录：{figures_dir}")
print("=" * 70)
