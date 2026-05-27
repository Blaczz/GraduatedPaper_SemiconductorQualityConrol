import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 原始工艺参数数据路径
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'semiconductor_quality_control.csv')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')

# 10个原始工艺参数
FEATURE_COLS = ['Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
                'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
                'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count']


def load_and_preprocess_data():
    """加载原始10个工艺参数数据"""
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"数据加载（原始工艺参数）: {df.shape[0]}行, {df.shape[1]}列")

    X = df[FEATURE_COLS].copy()
    y = df['Defect'].values

    print(f"特征数: {len(FEATURE_COLS)}, 缺陷率: {y.mean():.1%}")
    return X, y, FEATURE_COLS


def build_random_forest(X, y):
    """随机森林模型训练与超参数优化
    SMOTE过采样 + 网格搜索 + 5折交叉验证，ROC-AUC为指标
    SMOTE平衡训练集后，GridSearchCV搜索最优树参数
    random_state=42 确保结果可复现
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE过采样平衡训练集
    print(f"\n[SMOTE过采样] 训练集缺陷率: {y_train.mean():.1%} -> ", end='')
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"{y_train_res.mean():.1%} (已平衡)")
    print(f"  过采样前: {X_train.shape[0]}条 -> 过采样后: {X_train_res.shape[0]}条")

    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }

    print("\n[GridSearchCV] 搜索最优参数...")
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_res, y_train_res)

    print(f"最佳参数: {grid.best_params_}")
    print(f"最佳CV AUC: {grid.best_score_:.4f}")

    model = grid.best_estimator_

    # 特征重要性
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特征重要性 Top5:")
    print(feature_imp.head(5).to_string(index=False))

    return model, X_train, X_test, y_train, y_test, feature_imp


def evaluate(model, X_test, y_test):
    """模型性能评估"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }

    print("\n性能指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['正常', '缺陷'])}")

    return y_pred, y_proba, metrics


def plot_roc(y_test, y_proba, auc_val):
    """绘制ROC曲线（含AUC积分底纹）"""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC (AUC={auc_val:.3f})')
    plt.fill_between(fpr, tpr, alpha=0.15, color='darkgreen')  # AUC积分底纹
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('假正率', fontsize=12, fontweight='bold')
    plt.ylabel('真正率', fontsize=12, fontweight='bold')
    plt.title('图4-3 随机森林ROC曲线', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    optimal_idx = np.argmax(tpr - fpr)
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '43_随机森林ROC曲线.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] ROC曲线已保存")


def plot_confusion_matrix(y_test, y_pred):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['预测正常', '预测缺陷'],
                yticklabels=['实际正常', '实际缺陷'])
    plt.xlabel('预测标签', fontsize=12, fontweight='bold')
    plt.ylabel('实际标签', fontsize=12, fontweight='bold')
    plt.title('图4-4 随机森林混淆矩阵热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '44_随机森林混淆矩阵热力图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] 混淆矩阵已保存")


def plot_feature_importance(feature_imp):
    """绘制特征重要性图（原始工艺参数）"""
    top = feature_imp.head(len(feature_imp))  # 显示所有特征
    plt.figure(figsize=(10, 6))
    colors = ['#27ae60' if v > top['importance'].quantile(0.75) else '#2ecc71'
              for v in top['importance']]
    plt.barh(range(len(top)), top['importance'], color=colors, alpha=0.7, edgecolor='black')
    for i, val in enumerate(top['importance']):
        plt.text(val, i, f'  {val*100:.2f}%', va='center', fontweight='bold')
    plt.yticks(range(len(top)), top['feature'])
    plt.gca().invert_yaxis()
    plt.xlabel('Gini重要性', fontsize=12, fontweight='bold')
    plt.title('图4-5 随机森林工艺参数重要性', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '45_随机森林特征重要性.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] 特征重要性图已保存")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("【随机森林模型构建与性能评估】")

    X, y, _ = load_and_preprocess_data()
    model, X_train, X_test, y_train, y_test, feat_imp = build_random_forest(X, y)
    y_pred, y_proba, metrics = evaluate(model, X_test, y_test)
    plot_roc(y_test, y_proba, metrics['auc'])
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(feat_imp)

    print(f"\n=== 分析摘要 ===")
    print(f"AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, "
          f"Top1特征={feat_imp.iloc[0]['feature']}({feat_imp.iloc[0]['importance']*100:.2f}%)")
    print("[OK] 随机森林模型分析完成")


if __name__ == "__main__":
    main()
