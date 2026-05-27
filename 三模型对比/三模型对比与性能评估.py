import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
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
os.makedirs(FIGURES_DIR, exist_ok=True)

# 10个原始工艺参数
FEATURE_COLS = ['Chamber_Temperature', 'Gas_Flow_Rate', 'RF_Power', 'Etch_Depth',
                'Rotation_Speed', 'Vacuum_Pressure', 'Stage_Alignment_Error',
                'Vibration_Level', 'UV_Exposure_Intensity', 'Particle_Count']

# ============================================================
# 模型颜色
# ============================================================
MODEL_COLORS = {
    '逻辑回归': 'darkorange',
    '随机森林': 'darkgreen',
    '支持向量机': 'darkred'
}


def load_data():
    """加载原始10个工艺参数数据"""
    print("[进度 1/5] 加载原始工艺参数数据...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"  -> 数据加载成功: {df.shape[0]}行 × {df.shape[1]}列")

    X = df[FEATURE_COLS].copy()
    y = df['Defect'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"  -> 特征数: {len(FEATURE_COLS)}（10个原始工艺参数）")
    print(f"  -> 训练集: {len(X_train)}条, 测试集: {len(X_test)}条")
    print(f"  -> 缺陷率: 训练集={y_train.mean():.1%}, 测试集={y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """逻辑回归：使用GridSearchCV确定的最优参数（与独立脚本一致）"""
    print("\n" + "=" * 50)
    print("[进度 2/5] 逻辑回归 — 使用最优参数训练...")
    print("  C=0.001, penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=1000")
    print("=" * 50)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    t0 = time.time()
    model = LogisticRegression(
        C=0.001, penalty='l2', solver='lbfgs',
        class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    elapsed = time.time() - t0
    print(f"  -> 训练完成 (耗时 {elapsed:.1f}s)")

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }

    print(f"  -> 测试集: Acc={metrics['accuracy']:.4f}, Pre={metrics['precision']:.4f}, "
          f"Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")

    return y_test, y_pred, y_proba, metrics


def train_random_forest(X_train, X_test, y_train, y_test):
    """随机森林：SMOTE过采样 + 最优参数（与独立脚本一致）"""
    print("\n" + "=" * 50)
    print("[进度 3/5] 随机森林 — SMOTE过采样 + 最优参数训练...")
    print("  n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1, class_weight='balanced'")
    print("=" * 50)

    # SMOTE过采样（与独立脚本一致）
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  -> SMOTE: {X_train.shape[0]}条 -> {X_train_res.shape[0]}条 (已平衡)")

    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, class_weight='balanced',
        random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)
    elapsed = time.time() - t0
    print(f"  -> 训练完成 (耗时 {elapsed:.1f}s)")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }

    print(f"  -> 测试集: Acc={metrics['accuracy']:.4f}, Pre={metrics['precision']:.4f}, "
          f"Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")

    return y_test, y_pred, y_proba, metrics


def train_svm(X_train, X_test, y_train, y_test):
    """SVM：使用RBF核+类别平衡的最优参数（与独立脚本一致）"""
    print("\n" + "=" * 50)
    print("[进度 4/5] 支持向量机 — 使用最优参数训练...")
    print("  kernel='rbf', C=1, gamma=0.001, class_weight='balanced', max_iter=5000")
    print("=" * 50)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    t0 = time.time()
    model = SVC(kernel='rbf', C=1, gamma=0.001,
                class_weight='balanced', max_iter=5000,
                random_state=42, probability=True)
    model.fit(X_train_s, y_train)
    elapsed = time.time() - t0
    print(f"  -> 训练完成 (耗时 {elapsed:.1f}s)")

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }

    print(f"  -> 测试集: Acc={metrics['accuracy']:.4f}, Pre={metrics['precision']:.4f}, "
          f"Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")

    return y_test, y_pred, y_proba, metrics


def plot_roc_comparison(all_results):
    """绘制三模型ROC曲线对比图（含AUC积分底纹）"""
    print("\n[进度 5/5] 生成三模型ROC曲线对比图...")

    fig, ax = plt.subplots(figsize=(10, 8))

    for name in ['逻辑回归', '随机森林', '支持向量机']:
        data = all_results[name]
        fpr, tpr, _ = roc_curve(data['y_test'], data['y_proba'])
        auc_val = data['metrics']['auc']
        color = MODEL_COLORS[name]

        # ROC曲线 + AUC积分底纹
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC={auc_val:.3f})')
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)  # AUC积分底纹

        # 最佳阈值点
        optimal_idx = np.argmax(tpr - fpr)
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'o',
                color=color, markersize=8, markeredgecolor='white',
                markeredgewidth=1)

    # 随机分类器基线
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率', fontsize=12, fontweight='bold')
    ax.set_ylabel('真正率', fontsize=12, fontweight='bold')
    ax.set_title('三模型ROC曲线对比', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, '三模型ROC曲线对比.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> 图表已保存: {filepath}")


def save_performance_csv(all_results):
    """保存性能指标CSV汇总表"""
    rows = []
    for name in ['逻辑回归', '随机森林', '支持向量机']:
        m = all_results[name]['metrics']
        rows.append({
            '模型': name,
            '准确率': f"{m['accuracy']:.4f}",
            '精确率': f"{m['precision']:.4f}",
            '召回率': f"{m['recall']:.4f}",
            'F1分数': f"{m['f1']:.4f}",
            'AUC-ROC': f"{m['auc']:.4f}"
        })

    df = pd.DataFrame(rows)
    filepath = os.path.join(FIGURES_DIR, '三模型性能指标汇总.csv')
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"  -> 汇总表已保存: {filepath}")


def main():
    t_total = time.time()
    print("=" * 60)
    print("【三模型对比与性能评估】")
    print("  使用 GridSearchCV + 5折交叉验证确定的最优参数（已写死）")
    print("=" * 60)

    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_data()

    # 2-4. 训练三个模型
    lr_y_test, lr_pred, lr_proba, lr_metrics = \
        train_logistic_regression(X_train, X_test, y_train, y_test)
    rf_y_test, rf_pred, rf_proba, rf_metrics = \
        train_random_forest(X_train, X_test, y_train, y_test)
    svm_y_test, svm_pred, svm_proba, svm_metrics = \
        train_svm(X_train, X_test, y_train, y_test)

    all_results = {
        '逻辑回归': {'y_test': lr_y_test, 'y_pred': lr_pred,
                     'y_proba': lr_proba, 'metrics': lr_metrics},
        '随机森林': {'y_test': rf_y_test, 'y_pred': rf_pred,
                     'y_proba': rf_proba, 'metrics': rf_metrics},
        '支持向量机': {'y_test': svm_y_test, 'y_pred': svm_pred,
                       'y_proba': svm_proba, 'metrics': svm_metrics}
    }

    # 5. ROC对比图 + CSV
    plot_roc_comparison(all_results)
    save_performance_csv(all_results)

    # 摘要
    elapsed_total = time.time() - t_total
    print("\n" + "=" * 60)
    print(f"【性能指标汇总】(总耗时 {elapsed_total:.1f}s)")
    print("=" * 60)
    header = f"{'模型':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'AUC-ROC':<10}"
    print(header)
    print("-" * 60)
    for name in ['逻辑回归', '随机森林', '支持向量机']:
        m = all_results[name]['metrics']
        print(f"{name:<10} {m['accuracy']:<10.4f} {m['precision']:<10.4f} "
              f"{m['recall']:<10.4f} {m['f1']:<10.4f} {m['auc']:<10.4f}")

    sorted_models = sorted(all_results.items(),
                           key=lambda x: x[1]['metrics']['auc'], reverse=True)
    print(f"\n最佳模型（按AUC-ROC）: {sorted_models[0][0]} "
          f"(AUC={sorted_models[0][1]['metrics']['auc']:.4f})")
    print("[OK] 三模型对比分析完成")


if __name__ == "__main__":
    main()
