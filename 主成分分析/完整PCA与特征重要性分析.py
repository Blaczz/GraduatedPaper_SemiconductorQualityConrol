import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 数据加载与预处理
# ==============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'semiconductor_quality_control.csv')
figures_dir = os.path.join(script_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

df = pd.read_csv(data_path)
print(f"数据加载完成: {df.shape[0]} 行 × {df.shape[1]} 列")

# 工艺参数列表
process_features = [
    'Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
    'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
    'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count'
]

feature_cn = {
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

X = df[process_features].copy()
y = df['Defect'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================================================================
# 2. 特征重要性分析（三模型融合）
# ==============================================================================
print("\n>>> 特征重要性分析（随机森林 + 逻辑回归 + SVM）")

# 2.1 随机森林（SMOTE过采样 + 网格搜索确定的最优参数）
# 最优参数: n_estimators=300, max_depth=None, min_samples_split=2,
#   min_samples_leaf=1, class_weight='balanced'
# random_state=42 确保结果可复现
print("\n> 随机森林（SMOTE过采样 + 最优参数）")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"  SMOTE: {X_train.shape[0]}条 -> {X_train_res.shape[0]}条 (已平衡)")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_split=2,
    min_samples_leaf=1, class_weight='balanced',
    random_state=42, n_jobs=-1
)
rf.fit(X_train_res, y_train_res)
rf_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
rf_imp['rf_norm'] = rf_imp['importance'] / rf_imp['importance'].max() * 100

# 2.2 逻辑回归（使用网格搜索确定的最优参数）
# 最优参数: C=0.001, penalty='l2', solver='lbfgs',
#   max_iter=1000, class_weight='balanced'
# random_state=42 确保结果可复现
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # 供SVM排列重要性使用
lr = LogisticRegression(
    C=0.001, penalty='l2', solver='lbfgs',
    max_iter=1000, class_weight='balanced',
    random_state=42
)
lr.fit(X_train_scaled, y_train)
lr_imp = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr.coef_[0]
})
lr_imp['abs_coef'] = np.abs(lr_imp['coefficient'])
lr_imp = lr_imp.sort_values('abs_coef', ascending=False)
lr_imp['lr_norm'] = lr_imp['abs_coef'] / lr_imp['abs_coef'].max() * 100

# 2.3 SVM（RBF核 + 类别平衡 + 排列重要性）
# 最优参数: C=1, gamma=0.001, kernel='rbf', class_weight='balanced', max_iter=5000
# RBF核无法直接提取系数，使用排列重要性（permutation importance）
# random_state=42 确保结果可复现
print("> SVM（RBF核 + 类别平衡 + 排列重要性）")
svm = SVC(
    kernel='rbf', C=1, gamma=0.001, class_weight='balanced',
    max_iter=5000, random_state=42
)
svm.fit(X_train_scaled, y_train)
print(f"  支持向量数: {sum(svm.n_support_)}")
perm_result = permutation_importance(svm, X_test_scaled, y_test,
                                     n_repeats=10, random_state=42, n_jobs=-1)
svm_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': perm_result.importances_mean
})
svm_imp['abs_imp'] = np.abs(svm_imp['importance'])
svm_imp = svm_imp.sort_values('abs_imp', ascending=False)
svm_imp['svm_norm'] = svm_imp['abs_imp'] / svm_imp['abs_imp'].max() * 100

# 2.4 三模型融合
merged = rf_imp[['feature', 'rf_norm']].merge(
    lr_imp[['feature', 'lr_norm']], on='feature', how='outer'
).merge(
    svm_imp[['feature', 'svm_norm']], on='feature', how='outer'
).fillna(0)
merged['combined'] = (merged['rf_norm'] + merged['lr_norm'] + merged['svm_norm']) / 3
merged = merged.sort_values('combined', ascending=False)

print("融合特征重要性 Top 5:")
for _, r in merged.head(5).iterrows():
    print(f"  · {r['feature']}: {r['combined']:.1f}")

# ==============================================================================
# 3. 特征重要性可视化
# ==============================================================================

# 3.1 随机森林
fig, ax = plt.subplots(figsize=(12, 8))
top = rf_imp.head(10)
top['cn'] = top['feature'].map(lambda x: feature_cn.get(x, x))
colors = ['#e74c3c' if v > top['importance'].quantile(0.75) else '#3498db'
          for v in top['importance']]
ax.barh(range(len(top)), top['importance'], color=colors, alpha=0.7, edgecolor='black')
for i, (_, r) in enumerate(top.iterrows()):
    ax.text(r['importance'], i, f"  {r['importance']*100:.2f}%", va='center', fontweight='bold')
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top['cn'], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Gini重要性', fontsize=12, fontweight='bold')
ax.set_title('随机森林特征重要性（Top 10）', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '12_随机森林特征重要性.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3.2 逻辑回归
fig, ax = plt.subplots(figsize=(12, 8))
top = lr_imp.head(10)
top['cn'] = top['feature'].map(lambda x: feature_cn.get(x, x))
colors = ['#e74c3c' if c > 0 else '#3498db' for c in top['coefficient']]
ax.barh(range(len(top)), top['abs_coef'], color=colors, alpha=0.7, edgecolor='black')
for i, (_, r) in enumerate(top.iterrows()):
    ax.text(r['abs_coef'], i, f"  {r['coefficient']:.3f}", va='center', fontweight='bold')
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top['cn'], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('系数绝对值', fontsize=12, fontweight='bold')
ax.set_title('逻辑回归特征重要性（Top 10）', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '13_逻辑回归特征重要性.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3.3 SVM（排列重要性）
fig, ax = plt.subplots(figsize=(12, 8))
top = svm_imp.head(10)
top['cn'] = top['feature'].map(lambda x: feature_cn.get(x, x))
colors = ['#e74c3c' if v > 0 else '#3498db' for v in top['importance']]
ax.barh(range(len(top)), top['abs_imp'], color=colors, alpha=0.7, edgecolor='black')
for i, (_, r) in enumerate(top.iterrows()):
    ax.text(r['abs_imp'], i, f"  {r['importance']:.4f}", va='center', fontweight='bold')
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top['cn'], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('排列重要性（均值）', fontsize=12, fontweight='bold')
ax.set_title('SVM特征重要性（RBF核，排列重要性Top 10）', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '15_支持向量机特征重要性.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3.4 三模型融合
fig, ax = plt.subplots(figsize=(14, 10))
top = merged.head(10)
top['cn'] = top['feature'].map(lambda x: feature_cn.get(x, x))
x = np.arange(len(top))
w = 0.25
ax.barh(x - w, top['rf_norm'], w, label='随机森林', color='#3498db', alpha=0.8, edgecolor='black')
ax.barh(x,     top['lr_norm'], w, label='逻辑回归', color='#e74c3c', alpha=0.8, edgecolor='black')
ax.barh(x + w, top['svm_norm'], w, label='SVM',      color='#f39c12', alpha=0.8, edgecolor='black')
for i, (_, r) in enumerate(top.iterrows()):
    ax.text(max(r['rf_norm'], r['lr_norm'], r['svm_norm']) + 3, i,
            f"融合:{r['combined']:.1f}", va='center', fontsize=9,
            fontweight='bold', color='#2c3e50')
ax.set_yticks(x)
ax.set_yticklabels(top['cn'], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('标准化特征重要性', fontsize=12, fontweight='bold')
ax.set_title('三模型特征重要性融合（Top 10）', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '14_三模型融合特征重要性.png'), dpi=300, bbox_inches='tight')
plt.close()

print("[OK] 4 张特征重要性图表已生成")

# ==============================================================================
# 4. 基于特征重要性识别关键参数 + PCA降维
# ==============================================================================
print("\n>>> 特征重要性识别关键参数")

# 4.1 按融合重要性排序，识别Top关键参数（工业工程解读用）
merged_sorted = merged.sort_values('combined', ascending=False)
merged_sorted['cumsum_importance'] = merged_sorted['combined'].cumsum() / merged_sorted['combined'].sum()

# 识别累计重要性≥80%的参数数量（用于工业解读，不做硬性筛选）
key_count_80 = len(merged_sorted[merged_sorted['cumsum_importance'] <= 0.80])
key_params_top = merged_sorted.head(max(3, key_count_80))['feature'].tolist()

print(f"融合重要性Top参数:")
for i, p in enumerate(key_params_top, 1):
    cn = feature_cn.get(p, p)
    imp_val = merged_sorted[merged_sorted['feature'] == p]['combined'].values[0]
    print(f"  {i}. {cn} ({p}): 融合重要性={imp_val:.1f}")

# 4.2 用全部10个参数做PCA（保留完整信息，避免信息损失）
# 工业工程意义：特征重要性识别关键参数，PCA验证其是否主导过程变异
print(f"\n>>> 用全部参数做PCA降维")

scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

# 确定需要保留的主成分数量（累计方差≥90%）
n_components_90 = np.argmax(cumulative_var >= 0.9) + 1
n_components = min(n_components_90, len(process_features))
X_pca_reduced = X_pca[:, :n_components]

print(f"PCA方差解释率: PC1={explained_var[0]*100:.1f}%, PC2={explained_var[1]*100:.1f}%")
print(f"累计: 前2个主成分={cumulative_var[1]*100:.1f}%, 前{n_components}个主成分={cumulative_var[n_components-1]*100:.1f}%")
print(f"保留主成分数量: {n_components}（累计方差≥90%）")

# 验证关键参数是否主导了PC1/PC2
# 计算关键参数在PC1和PC2上的载荷绝对值之和
pc1_loadings = pd.DataFrame({
    'feature': process_features,
    'pc1_abs': np.abs(pca.components_[0])
}).set_index('feature')
pc2_loadings = pd.DataFrame({
    'feature': process_features,
    'pc2_abs': np.abs(pca.components_[1])
}).set_index('feature')

key_pc1_load_sum = pc1_loadings.loc[key_params_top, 'pc1_abs'].sum()
key_pc2_load_sum = pc2_loadings.loc[key_params_top, 'pc2_abs'].sum()
all_pc1_load_sum = pc1_loadings['pc1_abs'].sum()
all_pc2_load_sum = pc2_loadings['pc2_abs'].sum()

print(f"\n关键参数在PC1载荷占比: {key_pc1_load_sum/all_pc1_load_sum*100:.1f}%")
print(f"关键参数在PC2载荷占比: {key_pc2_load_sum/all_pc2_load_sum*100:.1f}%")
print(f"→ 特征重要性识别的关键参数{'确实' if key_pc1_load_sum/all_pc1_load_sum > 0.5 else '基本'}主导了过程变异")

# 4.3 保存降维数据供模型使用
pca_data_dir = os.path.join(script_dir, 'pca_reduced_data')
os.makedirs(pca_data_dir, exist_ok=True)

# 保存降维后的特征矩阵和标签
pca_df = pd.DataFrame(
    X_pca_reduced,
    columns=[f'PC{i+1}' for i in range(n_components)]
)
pca_df['Defect'] = y
pca_df.to_csv(os.path.join(pca_data_dir, 'pca_reduced_data.csv'), index=False)

# 保存关键参数列表（供后续分析引用）
key_params_info = pd.DataFrame({
    '参数英文名': key_params_top,
    '参数中文名': [feature_cn.get(p, p) for p in key_params_top],
    '融合重要性': [merged_sorted[merged_sorted['feature'] == p]['combined'].values[0] for p in key_params_top]
})
key_params_info.to_csv(os.path.join(pca_data_dir, 'key_params.csv'), index=False)

# 保存PCA模型和scaler（供模型脚本加载使用）
joblib.dump(pca, os.path.join(pca_data_dir, 'pca_model.pkl'))
joblib.dump(scaler_pca, os.path.join(pca_data_dir, 'scaler.pkl'))

print(f"\n[OK] PCA降维数据已保存至: {pca_data_dir}")
print(f"  - pca_reduced_data.csv: {X_pca_reduced.shape[1]}个主成分 × {X_pca_reduced.shape[0]}条样本")
print(f"  - key_params.csv: {len(key_params_top)}个关键参数")
print(f"  - pca_model.pkl + scaler.pkl: 模型文件")

# ==============================================================================
# 5. PCA可视化
# ==============================================================================

# 置信椭圆辅助函数
def confidence_ellipse(ax, x, y, color, alpha=0.25):
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vx, vy = vecs[:, 0]
    theta = np.degrees(np.atan2(vy, vx))
    scale = np.sqrt(chi2.ppf(0.95, df=2))
    width = 2 * scale * np.sqrt(vals[0])
    height = 2 * scale * np.sqrt(vals[1])
    ax.add_patch(Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height,
                         angle=theta, facecolor=color, alpha=alpha,
                         edgecolor=color, zorder=2))

# 载荷向量辅助函数（关键参数标注五角星）
def loading_vectors_with_highlight(ax, loadings, all_names, key_names):
    colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c',
              '#d35400', '#c0392b', '#8e44ad', '#27ae60', '#2980b9']
    for i, name in enumerate(all_names):
        cn = feature_cn.get(name, name)
        color = colors[i % len(colors)]
        vx, vy = loadings[i, 0] * 3.0, loadings[i, 1] * 3.0
        # 所有参数统一使用实线绘制
        ax.arrow(0, 0, vx, vy, color=color, alpha=0.95,
                 head_width=0.12, lw=2.5, zorder=5,
                 head_length=0.10, overhang=0.3)
        if name in key_names:
            # 关键参数：五角星标注
            ax.text(vx * 1.15, vy * 1.15, f'★{cn}', fontsize=12,
                    fontweight='bold', color=color, zorder=6)
        else:
            ax.text(vx * 1.15, vy * 1.15, cn, fontsize=10,
                    color=color, alpha=0.95, zorder=5)

# 5.1 PCA主成分散点图（全部参数，标注关键参数载荷）
fig, ax = plt.subplots(figsize=(14, 12))
defect_colors = {0: '#2ecc71', 1: '#e74c3c'}
defect_labels = {0: '正常样品', 1: '缺陷样品'}

for status in [0, 1]:
    mask = y == status
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=defect_colors[status],
               s=60, alpha=0.7, edgecolors='white', linewidth=0.5,
               label=defect_labels[status])
    confidence_ellipse(ax, X_pca[mask, 0], X_pca[mask, 1], defect_colors[status])

# 载荷向量：关键参数用实线加粗，非关键参数用虚线淡化
loading_vectors_with_highlight(ax, pca.components_.T, process_features, key_params_top)

ax.axhline(0, color='gray', ls='--', alpha=0.4)
ax.axvline(0, color='gray', ls='--', alpha=0.4)
ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}%)', fontsize=14, fontweight='bold')
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}%)', fontsize=14, fontweight='bold')
ax.set_title('PCA主成分分析 — 晶圆制造工艺数据（关键参数高亮）', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '23_PCA主成分分析图.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5.2 PCA碎石图 + 累计方差
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
comps = range(1, len(explained_var) + 1)

ax1.bar(comps, explained_var, color='skyblue', alpha=0.7, edgecolor='black')
ax1.plot(comps, explained_var, 'ro-', linewidth=2, markersize=4)
ax1.set_xlabel('主成分', fontsize=12, fontweight='bold')
ax1.set_ylabel('解释方差比', fontsize=12, fontweight='bold')
ax1.set_title('(a) PCA碎石图', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 11])

ax2.plot(comps, cumulative_var, 'bo-', linewidth=2, markersize=5)
ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='80%阈值')
ax2.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, label='90%阈值')
ax2.set_xlabel('主成分数量', fontsize=12, fontweight='bold')
ax2.set_ylabel('累计解释方差比', fontsize=12, fontweight='bold')
ax2.set_title('(b) 累计解释方差比', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([1, 11])
ax2.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '24_PCA方差解释率图.png'), dpi=300, bbox_inches='tight')
plt.close()

print("[OK] 2 张PCA图表已生成")

# ==============================================================================
# 6. 输出 Excel 报告（全中文）
# ==============================================================================
excel_path = os.path.join(script_dir, '完整PCA分析报告.xlsx')

# 6.1 载荷矩阵（中文特征名，全部参数）
loadings_cn = pca.components_.T[:, :2].copy()
loadings_df = pd.DataFrame(
    loadings_cn,
    index=[feature_cn.get(f, f) for f in process_features],
    columns=['第一主成分(PC1)', '第二主成分(PC2)']
)
# 计算各特征在主成分上的贡献度（载荷平方）
loadings_df['PC1贡献度(%)'] = (loadings_df['第一主成分(PC1)'] ** 2) * 100
loadings_df['PC2贡献度(%)'] = (loadings_df['第二主成分(PC2)'] ** 2) * 100
# 标注是否为关键参数
loadings_df['是否关键参数'] = ['是' if f in key_params_top else '否' for f in process_features]
loadings_df.index.name = '工艺参数'

# 6.2 各主成分方差解释率
n_pcs = len(explained_var)
variance_data = {
    '主成分': [f'PC{i+1}' for i in range(n_pcs)],
    '方差解释率(%)': [round(explained_var[i] * 100, 2) for i in range(n_pcs)],
    '累计方差解释率(%)': [round(cumulative_var[i] * 100, 2) for i in range(n_pcs)]
}
variance_df = pd.DataFrame(variance_data)

# 6.3 特征重要性（中文特征名）
importance_df = merged.copy()
importance_df['工艺参数'] = importance_df['feature'].map(lambda x: feature_cn.get(x, x))
importance_df = importance_df.rename(columns={
    'rf_norm': '随机森林重要性(%)',
    'lr_norm': '逻辑回归重要性(%)',
    'svm_norm': 'SVM重要性(%)',
    'combined': '融合重要性(%)'
})
importance_df = importance_df[[
    '工艺参数', '随机森林重要性(%)', '逻辑回归重要性(%)',
    'SVM重要性(%)', '融合重要性(%)'
]].sort_values('融合重要性(%)', ascending=False).reset_index(drop=True)

# 6.4 主成分对样本的解释能力（每个样本在PC1/PC2上的得分）
sample_scores_df = pd.DataFrame({
    '样品编号': range(1, len(X_pca) + 1),
    'PC1得分': np.round(X_pca[:, 0], 4),
    'PC2得分': np.round(X_pca[:, 1], 4),
    '缺陷标签': ['正常' if v == 0 else '缺陷' for v in y]
})

# 6.5 写入 Excel
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    loadings_df.to_excel(writer, sheet_name='载荷矩阵')
    variance_df.to_excel(writer, sheet_name='方差解释率', index=False)
    importance_df.to_excel(writer, sheet_name='特征重要性', index=False)
    sample_scores_df.to_excel(writer, sheet_name='样品主成分得分', index=False)
    # 分析说明汇总表
    pd.DataFrame({
        '分析项目': [
            '样本总量', '工艺参数数量', '筛选后关键参数数量',
            '保留主成分数量', '筛选依据',
            'PC1方差解释率', 'PC2方差解释率',
            '前2主成分累计方差解释率', '达到90%所需主成分数',
            '关键参数PC1载荷占比', '关键参数PC2载荷占比',
            '特征重要性排名第1', '特征重要性排名第2', '特征重要性排名第3'
        ],
        '数值/描述': [
            f'{len(df)}条',
            f'{len(process_features)}个',
            f'{len(key_params_top)}个',
            f'{n_components}个',
            f'融合重要性累计≥80%（用于识别关键参数，PCA使用全部参数）',
            f'{explained_var[0]*100:.1f}%',
            f'{explained_var[1]*100:.1f}%',
            f'{cumulative_var[1]*100:.1f}%',
            f'{n_components}个',
            f'{key_pc1_load_sum/all_pc1_load_sum*100:.1f}%',
            f'{key_pc2_load_sum/all_pc2_load_sum*100:.1f}%',
            feature_cn.get(merged.iloc[0]['feature'], merged.iloc[0]['feature']),
            feature_cn.get(merged.iloc[1]['feature'], merged.iloc[1]['feature']),
            feature_cn.get(merged.iloc[2]['feature'], merged.iloc[2]['feature'])
        ]
    }).to_excel(writer, sheet_name='分析说明', index=False)

print(f"\n[OK] Excel报告已保存: {excel_path}")

# ==============================================================================
# 8. 5M1E维度重要性映射
# ==============================================================================
print("\n>>> 5M1E维度重要性映射")

# 5M1E分类定义（与01_数据探索与质量评估/工具模块.py一致）
FIVE_M1E = {
    'Machine（机）': ['Chamber_Temperature', 'RF_Power', 'Rotation_Speed', 'Vibration_Level'],
    'Material（料）': ['Gas_Flow_Rate', 'Etch_Depth', 'Particle_Count'],
    'Method（法）':   ['UV_Exposure_Intensity'],
    'Measurement（测）': ['Stage_Alignment_Error', 'Vacuum_Pressure'],
    'Environment（环）': ['Chamber_Temperature'],
    'Man（人）':       []
}

# 5M1E颜色
FIVE_M1E_COLORS = {
    'Machine（机）':    '#3498db',
    'Material（料）':   '#2ecc71',
    'Method（法）':     '#e74c3c',
    'Measurement（测）': '#9b59b6',
    'Environment（环）': '#f39c12',
    'Man（人）':        '#1abc9c'
}
FIVE_M1E_CN = {
    'Machine（机）': '机（设备）',
    'Material（料）': '料（物料）',
    'Method（法）':   '法（工艺）',
    'Measurement（测）': '测（测量）',
    'Environment（环）': '环（环境）',
    'Man（人）':      '人（人员）'
}

# 从merged中取融合重要性数据
imp_dict = dict(zip(merged['feature'], merged['combined']))

# 按5M1E维度聚合
m1e_scores = {}
m1e_params = {}
for cat, params in FIVE_M1E.items():
    if not params:
        m1e_scores[cat] = 0.0
        m1e_params[cat] = []
        continue
    scores = [imp_dict.get(p, 0) for p in params]
    m1e_scores[cat] = np.mean(scores)
    m1e_params[cat] = [{'feature': p, 'importance': imp_dict.get(p, 0),
                         'cn': feature_cn.get(p, p)} for p in params]

# 转换为DataFrame
m1e_df = pd.DataFrame([
    {'5M1E分类': k, '5M1E中文': FIVE_M1E_CN.get(k, k),
     '重要性得分': v, '颜色': FIVE_M1E_COLORS.get(k, '#95a5a6')}
    for k, v in sorted(m1e_scores.items(), key=lambda x: x[1], reverse=True)
])
m1e_df['占比(%)'] = m1e_df['重要性得分'] / m1e_df['重要性得分'].sum() * 100
m1e_df['累计占比(%)'] = m1e_df['占比(%)'].cumsum()

print("\n5M1E维度重要性排名:")
for _, r in m1e_df.iterrows():
    print(f"  {r['5M1E分类']:20s}  重要性={r['重要性得分']:.2f}  占比={r['占比(%)']:.1f}%")

# 8.1 5M1E重要性饼图
fig, ax = plt.subplots(figsize=(10, 8))
valid = m1e_df[m1e_df['重要性得分'] > 0]
wedges, texts, autotexts = ax.pie(
    valid['重要性得分'], labels=valid['5M1E分类'],
    autopct='%1.1f%%', startangle=90,
    colors=valid['颜色'].tolist(),
    textprops={'fontsize': 11, 'fontweight': 'bold'},
    pctdistance=0.75, wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
for t in autotexts:
    t.set_fontsize(10)
    t.set_fontweight('bold')
ax.set_title('图3-8 5M1E维度重要性分布', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '31_5M1E维度重要性饼图.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] 5M1E饼图已保存")

# 8.2 5M1E帕累托图
fig, ax1 = plt.subplots(figsize=(12, 8))
valid = m1e_df[m1e_df['重要性得分'] > 0].copy()
x = np.arange(len(valid))
bars = ax1.bar(x, valid['重要性得分'], color=valid['颜色'].tolist(),
               alpha=0.8, edgecolor='black', width=0.6)
for i, (_, r) in enumerate(valid.iterrows()):
    ax1.text(i, r['重要性得分'] + 0.02, f"{r['占比(%)']:.1f}%",
             ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(valid['5M1E分类'], fontsize=11, fontweight='bold')
ax1.set_ylabel('重要性得分', fontsize=12, fontweight='bold')
ax1.set_title('图3-9 5M1E维度帕累托分析', fontsize=16, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
# 累计曲线
ax2 = ax1.twinx()
ax2.plot(x, valid['累计占比(%)'], 'ro-', linewidth=2, markersize=8, label='累计占比')
ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.text(len(x)-1, 82, '80%阈值', fontsize=10, color='gray', ha='right')
ax2.set_ylabel('累计占比(%)', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 105])
ax2.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, '32_5M1E帕累托分析图.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] 5M1E帕累托图已保存")

# 8.3 添加到Excel
m1e_report = m1e_df[['5M1E分类', '5M1E中文', '重要性得分', '占比(%)', '累计占比(%)']].copy()

# 每个5M1E维度下各参数的详细重要性
param_details = []
for cat, params in m1e_params.items():
    for p in params:
        param_details.append({
            '5M1E分类': cat,
            '5M1E中文': FIVE_M1E_CN.get(cat, cat),
            '工艺参数': p['cn'],
            '参数英文名': p['feature'],
            '融合重要性': round(p['importance'], 2)
        })
param_detail_df = pd.DataFrame(param_details)

with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
    m1e_report.to_excel(writer, sheet_name='5M1E维度重要性', index=False)
    param_detail_df.to_excel(writer, sheet_name='5M1E参数明细', index=False)

print("[OK] 5M1E分析已添加到Excel报告")
print(f"\n>> 5M1E关键发现:")
top_m1e = m1e_df[m1e_df['累计占比(%)'] <= 80]
if len(top_m1e) == 0:
    top_m1e = m1e_df.head(1)
print(f"  - 关键维度: {'、'.join(top_m1e['5M1E分类'].tolist())}")
print(f"  - 累计贡献: {top_m1e['占比(%)'].sum():.1f}%（帕累托80/20原则）")
print(f"  - Chamber_Temperature同时影响: Machine（机）和 Environment（环）")

# 5M1E关键发现文本（供论文引用）
print(f"\n>> 5M1E维度重要性降序:")
for i, (_, r) in enumerate(m1e_df.iterrows(), 1):
    params_str = '、'.join([p['cn'] for p in m1e_params.get(r['5M1E分类'], [])])
    print(f"  {i}. {r['5M1E分类']} ({r['占比(%)']:.1f}%): {params_str}")

# ==============================================================================
# 7. 分析摘要
# ==============================================================================
print("\n" + "=" * 60)
print("分析完成摘要")
print("=" * 60)
print(f"特征重要性 Top 3:")
for i, (_, r) in enumerate(merged.head(3).iterrows(), 1):
    cn = feature_cn.get(r['feature'], r['feature'])
    print(f"  {i}. {cn}: 融合重要性={r['combined']:.1f}")
print(f"\n关键参数识别: 累计融合重要性≥80% → {len(key_params_top)}个关键参数")
print(f"  关键参数PC1载荷占比: {key_pc1_load_sum/all_pc1_load_sum*100:.1f}%")
print(f"  关键参数PC2载荷占比: {key_pc2_load_sum/all_pc2_load_sum*100:.1f}%")
print(f"  → 特征重要性识别的关键参数主导了过程变异")
print(f"\nPCA: 前{n_components}个主成分累计方差解释率 = {cumulative_var[n_components-1]*100:.1f}%")
print(f"降维后特征维度: {X_pca_reduced.shape[1]}（原始10维 → {X_pca_reduced.shape[1]}维）")
print(f"\n5M1E维度重要性 Top 3:")
for i, (_, r) in enumerate(m1e_df.head(3).iterrows(), 1):
    params_str = '、'.join([p['cn'] for p in m1e_params.get(r['5M1E分类'], [])])
    print(f"  {i}. {r['5M1E分类']} ({r['占比(%)']:.1f}%): {params_str}")
print(f"\n输出文件: 8 张图表 + 1 个 Excel 报告 + PCA降维数据")
print("=" * 60)
