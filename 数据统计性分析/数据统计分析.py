import pandas as pd
import numpy as np
import os

# ==============================================================================
# 1. 数据加载
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'semiconductor_quality_control.csv')

df = pd.read_csv(data_path)

print("=" * 70)
print("半导体晶圆制造数据 · 描述性统计分析")
print("=" * 70)
print(f"样本总量: {len(df)}")
print(f"时间跨度: {df['Timestamp'].min()} ~ {df['Timestamp'].max()}")

# ==============================================================================
# 2. 参数定义
# ==============================================================================
PARAMS = [
    'Chamber_Temperature',    # 腔体温度
    'Gas_Flow_Rate',          # 气体流量
    'RF_Power',               # 射频功率
    'Etch_Depth',             # 刻蚀深度
    'Rotation_Speed',         # 旋转速度
    'Vacuum_Pressure',        # 真空压力
    'Stage_Alignment_Error',  # 晶圆台对准误差
    'Vibration_Level',        # 振动级别
    'UV_Exposure_Intensity',  # UV曝光强度
    'Particle_Count'          # 颗粒数
]

PARAM_CN = {
    'Chamber_Temperature':   '腔体温度',
    'Gas_Flow_Rate':         '气体流量',
    'RF_Power':              '射频功率',
    'Etch_Depth':            '刻蚀深度',
    'Rotation_Speed':        '旋转速度',
    'Vacuum_Pressure':       '真空压力',
    'Stage_Alignment_Error': '晶圆台对准误差',
    'Vibration_Level':       '振动级别',
    'UV_Exposure_Intensity': 'UV曝光强度',
    'Particle_Count':        '颗粒数'
}

# ==============================================================================
# 3. 计算统计量
# ==============================================================================
stats_list = []
for p in PARAMS:
    s = df[p]
    m = s.mean()
    std = s.std()
    stats_list.append({
        '参数':        PARAM_CN[p],
        '均值':        round(m, 6),
        '标准差':      round(std, 6),
        '变异系数(%)': round((std / m) * 100 if m != 0 else 0, 2),
        '最小值':      round(s.min(), 6),
        '最大值':      round(s.max(), 6),
        '极差':        round(s.max() - s.min(), 6),
        '中位数':      round(s.median(), 6)
    })

stats_df = pd.DataFrame(stats_list)

# ==============================================================================
# 4. 控制台输出
# ==============================================================================
print("\n" + "-" * 95)
header = (f"{'参数':<16} {'均值':>12} {'标准差':>12} {'CV(%)':>8} "
          f"{'最小值':>12} {'最大值':>12} {'极差':>12} {'中位数':>12}")
print(header)
print("-" * 95)

for _, r in stats_df.iterrows():
    print(f"{r['参数']:<16} {r['均值']:>12.6f} {r['标准差']:>12.6f} "
          f"{r['变异系数(%)']:>8.2f} {r['最小值']:>12.6f} {r['最大值']:>12.6f} "
          f"{r['极差']:>12.6f} {r['中位数']:>12.6f}")

# ==============================================================================
# 5. 变异系数分级摘要
# ==============================================================================
cv_high = stats_df[stats_df['变异系数(%)'] >= 10]
cv_med  = stats_df[(stats_df['变异系数(%)'] >= 5) & (stats_df['变异系数(%)'] < 10)]
cv_low  = stats_df[stats_df['变异系数(%)'] < 5]

print(f"\n变异系数(CV)分级：")
print(f"  低变异 (<5%)        {len(cv_low):>2} 个 -> 工艺稳定")
print(f"  中等变异 (5%~10%)   {len(cv_med):>2} 个 -> 需关注")
print(f"  高变异 (>=10%)      {len(cv_high):>2} 个 -> 存在异常风险")

if len(cv_high) > 0:
    print(f"\n  高变异参数明细：")
    for _, r in cv_high.sort_values('变异系数(%)', ascending=False).iterrows():
        print(f"    · {r['参数']}: CV = {r['变异系数(%)']:.2f}%")

# ==============================================================================
# 6. 保存 Excel
# ==============================================================================
excel_path = os.path.join(script_dir, '十项指标描述性统计.xlsx')

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    stats_df.to_excel(writer, sheet_name='描述性统计', index=False)

    # 变异系数分级附表
    cv_analysis = stats_df[['参数', '变异系数(%)']].copy()
    cv_analysis['变异等级'] = cv_analysis['变异系数(%)'].apply(
        lambda x: '高变异' if x >= 10 else '中等变异' if x >= 5 else '低变异'
    )
    cv_analysis = cv_analysis.sort_values('变异系数(%)', ascending=False)
    cv_analysis.to_excel(writer, sheet_name='变异系数分级', index=False)

print(f"\n已保存: {excel_path}")
print("=" * 70)
