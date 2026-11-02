"""
DAMA61 - Written Assignment 1 - Problem 2: Breast Cancer Wisconsin (binary classification)

Organised as an end-to-end Machine Learning project following the checklist of
A. Geron, "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow", Ch. 2:

    Step 1  Look at the big picture                  (assignment Q1 discussion)
    Step 2  Get the data                             (Q1, Q2 split)
    Step 3  Discover and visualise the data          (Q1 balance, Q2 feature scales)
    Step 4  Prepare the data for ML algorithms       (Q2 standardisation)
    Step 5  Select a model and train it              (Q2 SGD, Q3, Q4)
    Step 6  Fine-tune the model                      (Q5 threshold, Q6 ROC + hyper-parameter search)
    Step 7  Present the solution                     (Q6 test-set evaluation)
    Step 8  Launch, monitor and maintain the system  (beyond the assignment)

Malignant = positive class (1), benign = negative class (0).

Run from the WA1 folder:
    python problem2.py            # prints every step, shows + saves figures (./figures), model (./models)
    python problem2.py --no-show  # only save the figures
In Spyder just press F5: figures appear in the Plots pane as each step finishes.

The functions are also imported by the Streamlit app (app.py).
"""
from pathlib import Path
from datetime import datetime
import argparse
import os
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

import sklearn
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, roc_curve, roc_auc_score)
from sklearn.model_selection import (train_test_split, cross_val_score, cross_val_predict,
                                     cross_validate, StratifiedKFold, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({'font.size': 11})

RANDOM_STATE = 42
# Parallel jobs: all CPU cores locally (-1); app.py sets WA1_N_JOBS=1 on Streamlit Cloud (1 GB memory limit)
N_JOBS = int(os.environ.get('WA1_N_JOBS', '-1'))
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'problem2_model.joblib'
TARGET_RECALL = 0.95


# ============================================================================ STEP 1
BIG_PICTURE = {
    'Objective': 'Support the diagnosis of breast masses: from 30 measurements of cell nuclei in a '
                 'digitised image, predict whether a mass is malignant (1) or benign (0), so that '
                 'suspicious cases are prioritised for further examination.',
    'Framing': 'Supervised learning (labelled cases), binary classification, batch/offline learning.',
    'Performance measure': 'Accuracy is not enough (classes are imbalanced and errors have different costs). '
                           'Main focus: RECALL of the malignant class (missed cancers are the costliest error) '
                           'subject to acceptable precision; also F1, confusion matrix and ROC AUC.',
    'Baseline': 'Always predict "benign" -> ~63% accuracy and 0% recall. A useful model must clearly beat this.',
    'Assumptions to check': 'Target coding in scikit-learn is 0 = malignant, 1 = benign -> must be recoded. '
                            'The model is a decision-support tool; a clinician makes the final decision.',
}


def step1_big_picture():
    return BIG_PICTURE


# ============================================================================ STEP 2
def load_cancer():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = (data.target == 0).astype(int)        # recode: malignant -> 1, benign -> 0
    y.name = 'malignant'
    return X, y, data


def class_table(y):
    counts = y.value_counts().sort_index()
    return pd.DataFrame({'class': ['benign (0)', 'malignant (1)'],
                         'count': counts.values,
                         'percentage': (100 * counts / counts.sum()).round(2).values})


def step2_get_data(random_state=RANDOM_STATE):
    X, y, data = load_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state)
    split_table = pd.DataFrame({
        'n': [len(y), len(y_train), len(y_test)],
        'malignant': [y.sum(), y_train.sum(), y_test.sum()],
        'malignant %': [100 * y.mean(), 100 * y_train.mean(), 100 * y_test.mean()],
    }, index=['full', 'train', 'test']).round(2)
    recode_check = pd.crosstab(data.target.map({0: 'malignant', 1: 'benign'}), y)
    return {'X': X, 'y': y, 'feature_names': list(X.columns),
            'original_target_names': list(data.target_names),
            'recode_check': recode_check, 'missing': int(X.isnull().sum().sum()),
            'class_table': class_table(y), 'split_table': split_table,
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


# ============================================================================ STEP 3
def step3_explore(data):
    """Explore ONLY the training set."""
    X, y = data['X_train'], data['y_train']
    scales = X.describe().T[['mean', 'std', 'min', 'max']].sort_values('max')
    corr_target = X.corrwith(y).sort_values(key=abs, ascending=False).rename('corr with malignant')
    feat_corr = X.corr()
    upper = feat_corr.where(np.triu(np.ones(feat_corr.shape, dtype=bool), k=1))
    high_pairs = (upper.stack().dropna().rename('r').reset_index()
                  .rename(columns={'level_0': 'feature 1', 'level_1': 'feature 2'})
                  .sort_values('r', key=abs, ascending=False))
    return {'scales': scales, 'corr_target': corr_target, 'feature_corr': feat_corr,
            'high_corr_pairs': high_pairs, 'n_pairs_above_0_9': int((high_pairs['r'].abs() > 0.9).sum())}


# ============================================================================ STEP 4
def step4_prepare(data):
    """No missing values, all numeric -> the only transformation is standardisation (inside the pipeline)."""
    scaler = StandardScaler().fit(data['X_train'])
    Xs = pd.DataFrame(scaler.transform(data['X_train']), columns=data['X_train'].columns)
    return {'scaler': scaler,
            'before': data['X_train'].describe().T[['mean', 'std']],
            'after': Xs.describe().T[['mean', 'std']]}


def build_sgd_pipeline(random_state=RANDOM_STATE, **sgd_params):
    return Pipeline([('scaler', StandardScaler()),
                     ('sgd', SGDClassifier(random_state=random_state, **sgd_params))])


# ============================================================================ STEP 5
def metrics_at_threshold(y_true, scores, threshold):
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {'threshold': float(threshold),
            'accuracy': accuracy_score(y_true, pred),
            'precision': precision_score(y_true, pred, zero_division=0),
            'recall': recall_score(y_true, pred, zero_division=0),
            'f1': f1_score(y_true, pred, zero_division=0),
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)}


def step5_select_train(data, random_state=RANDOM_STATE):
    X_train, y_train = data['X_train'], data['y_train']
    sgd = build_sgd_pipeline(random_state)

    # Q2: train the SGD pipeline
    sgd_fitted = clone(sgd).fit(X_train, y_train)

    # Q3: 3-fold CV accuracy + out-of-fold predictions (cv=3 -> StratifiedKFold(3) for classifiers)
    cv_acc = cross_val_score(sgd, X_train, y_train, cv=3, scoring='accuracy')
    cv_pred = cross_val_predict(sgd, X_train, y_train, cv=3)
    resub_acc = accuracy_score(y_train, sgd_fitted.predict(X_train))

    # Q4: confusion matrix and class-1 metrics
    cm = confusion_matrix(y_train, cv_pred)

    # Shortlist of candidate models (same 3 folds)
    candidates = {
        'Baseline (always benign)': DummyClassifier(strategy='most_frequent'),
        'SGD linear classifier': SGDClassifier(random_state=random_state),
        'Logistic Regression': LogisticRegression(max_iter=5000),
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=N_JOBS),
    }
    rows = []
    for name, model in candidates.items():
        res = cross_validate(Pipeline([('scaler', StandardScaler()), ('model', model)]), X_train, y_train,
                             cv=StratifiedKFold(3), scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        rows.append({'model': name, **{m: res[f'test_{m}'].mean() for m in
                                       ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}})
    comparison = pd.DataFrame(rows).set_index('model')

    return {'sgd_pipeline': sgd, 'sgd_fitted': sgd_fitted,
            'cv_acc': cv_acc, 'cv_pred': cv_pred, 'resub_acc': resub_acc,
            'cv_pred_acc': accuracy_score(y_train, cv_pred),
            'baseline_acc': 1 - y_train.mean(),
            'cm': cm,
            'precision': precision_score(y_train, cv_pred, zero_division=0),
            'recall': recall_score(y_train, cv_pred),
            'f1': f1_score(y_train, cv_pred),
            'comparison': comparison}


# ============================================================================ STEP 6
def threshold_for_recall(y_true, scores, target_recall=TARGET_RECALL):
    """Highest threshold whose recall is still >= target_recall (keeps precision as high as possible)."""
    _, recalls, thresholds = precision_recall_curve(y_true, scores)
    return float(thresholds[recalls[:-1] >= target_recall].max())


def threshold_table(y_true, scores, extra_thresholds=()):
    grid = np.r_[np.quantile(scores, [0.2, 0.35, 0.5, 0.65, 0.8]), 0.0, list(extra_thresholds)]
    rows = [metrics_at_threshold(y_true, scores, t) for t in np.sort(np.unique(grid))]
    return pd.DataFrame(rows).round(4)


def step6_fine_tune(data, s5, random_state=RANDOM_STATE, target_recall=TARGET_RECALL):
    X_train, y_train = data['X_train'], data['y_train']

    # (a) Decision-threshold tuning of the assignment SGD model (Q5) + ROC (Q6)
    cv_scores = cross_val_predict(s5['sgd_pipeline'], X_train, y_train, cv=3, method='decision_function')
    thr = threshold_for_recall(y_train, cv_scores, target_recall)

    # (b) Hyper-parameter search for the SGD classifier (threshold-free metric: ROC AUC)
    param_grid = {'sgd__loss': ['hinge', 'log_loss', 'modified_huber'],
                  'sgd__alpha': [1e-4, 1e-3, 1e-2, 1e-1],
                  'sgd__penalty': ['l2', 'l1', 'elasticnet']}
    search = GridSearchCV(build_sgd_pipeline(random_state), param_grid, cv=StratifiedKFold(3),
                          scoring='roc_auc', n_jobs=N_JOBS)
    search.fit(X_train, y_train)
    res = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({'loss': res['param_sgd__loss'], 'alpha': res['param_sgd__alpha'],
                          'penalty': res['param_sgd__penalty'],
                          'CV ROC AUC mean': res['mean_test_score'], 'CV ROC AUC std': res['std_test_score'],
                          'rank': res['rank_test_score']}).sort_values('rank')
    tuned = search.best_estimator_
    tuned_scores = cross_val_predict(clone(tuned), X_train, y_train, cv=3, method='decision_function')
    tuned_thr = threshold_for_recall(y_train, tuned_scores, target_recall)

    weights = pd.Series(tuned.named_steps['sgd'].coef_.ravel(), index=X_train.columns, name='weight')
    return {'cv_scores': cv_scores, 'threshold': thr, 'target_recall': target_recall,
            'cv_auc': roc_auc_score(y_train, cv_scores),
            'default_cv': metrics_at_threshold(y_train, cv_scores, 0.0),
            'selected_cv': metrics_at_threshold(y_train, cv_scores, thr),
            'threshold_table': threshold_table(y_train, cv_scores, [thr]),
            'search': search, 'search_results': table,
            'best_params': {k.replace('sgd__', ''): v for k, v in search.best_params_.items()},
            'tuned_model': tuned, 'tuned_scores': tuned_scores, 'tuned_threshold': tuned_thr,
            'tuned_cv_auc': roc_auc_score(y_train, tuned_scores),
            'tuned_selected_cv': metrics_at_threshold(y_train, tuned_scores, tuned_thr),
            'weights': weights.reindex(weights.abs().sort_values(ascending=False).index)}


# ============================================================================ STEP 7
def step7_present(data, s5, s6):
    """Final evaluation on the test set (used only here)."""
    X_test, y_test = data['X_test'], data['y_test']
    final = s5['sgd_fitted']                                   # assignment model, trained on the full training set
    test_scores = final.decision_function(X_test)
    tuned_test_scores = s6['tuned_model'].decision_function(X_test)

    cols = {
        'CV (train) - default thr': s6['default_cv'],
        'Test - default thr': metrics_at_threshold(y_test, test_scores, 0.0),
        'CV (train) - Q5 thr': s6['selected_cv'],
        'Test - Q5 thr': metrics_at_threshold(y_test, test_scores, s6['threshold']),
        'Tuned SGD - test - its recall-95 thr': metrics_at_threshold(y_test, tuned_test_scores, s6['tuned_threshold']),
    }
    keys = ['threshold', 'accuracy', 'precision', 'recall', 'f1', 'TP', 'FP', 'FN', 'TN']
    table = pd.DataFrame({k: [v[m] for m in keys] for k, v in cols.items()}, index=keys).round(4)
    return {'test_scores': test_scores, 'tuned_test_scores': tuned_test_scores,
            'results': table,
            'test_auc': roc_auc_score(y_test, test_scores),
            'tuned_test_auc': roc_auc_score(y_test, tuned_test_scores)}


KEY_FINDINGS = [
    'Dataset: 569 cases, 30 numeric features, 212 malignant (37.3%) vs 357 benign (62.7%) - moderately imbalanced.',
    'Features have very different scales (area in the thousands vs fractal dimension ~0.05) and many are '
    'highly correlated (radius / perimeter / area) - standardisation is essential for SGD.',
    'The standardised SGD linear classifier reaches ~95%+ accuracy and ROC AUC close to 1 in cross-validation.',
    'The decision threshold controls the precision/recall trade-off; a threshold giving >= 95% recall '
    'is preferred in a diagnostic setting (missed cancers are costlier than false alarms).',
    'Test-set results are consistent with cross-validation, so the model generalises to unseen cases.',
]


# ============================================================================ STEP 8
def save_model(model, metadata, path=MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    joblib.dump({'model': model, 'metadata': metadata}, path)
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2, default=str))
    return path


def load_model(path=MODEL_PATH):
    bundle = joblib.load(path)
    return bundle['model'], bundle['metadata']


def predict_cases(model, X_new, threshold):
    """Return decision scores and labels (1 = malignant) using the operating threshold chosen in Step 6."""
    scores = model.decision_function(X_new)
    return pd.DataFrame({'score': scores, 'prediction': (scores >= threshold).astype(int),
                         'label': np.where(scores >= threshold, 'malignant', 'benign')}, index=X_new.index)


def drift_report(reference, new, alpha=0.01):
    """Kolmogorov-Smirnov test for every feature: flags inputs whose distribution changed."""
    rows = []
    for col in reference.columns:
        stat, p = stats.ks_2samp(reference[col], new[col])
        rows.append({'feature': col, 'KS statistic': stat, 'p_value': p, 'drift': p < alpha})
    return pd.DataFrame(rows).sort_values('KS statistic', ascending=False).round(4)


def performance_check(model, X_new, y_new, threshold, min_recall=TARGET_RECALL - 0.05):
    """When confirmed diagnoses (biopsy results) arrive: check live recall/precision."""
    m = metrics_at_threshold(y_new, model.decision_function(X_new), threshold)
    m['min recall'] = min_recall
    m['alert'] = bool(m['recall'] < min_recall)
    return m


MAINTENANCE_PLAN = [
    'Deploy the saved pipeline (scaler + SGD in one joblib file) together with the chosen threshold.',
    'Log every input, score and prediction; run drift_report on each new batch (e.g. a new imaging device '
    'or lab procedure changes the measurement scales).',
    'When biopsy results arrive, compute live recall/precision (performance_check); alert if recall drops below 90%.',
    'Retrain periodically on the enlarged dataset and re-select the threshold on cross-validated scores, never on the test set.',
    'Keep a clinician in the loop: the model prioritises cases, it does not replace diagnosis. Version data, code and models.',
]


def step8_launch(data, s5, s6, s7, save=True, path=MODEL_PATH):
    final = s5['sgd_fitted']
    metadata = {'project': 'DAMA61 WA1 - Problem 2 (breast cancer)',
                'model': 'StandardScaler + SGDClassifier (hinge)',
                'threshold': s6['threshold'],
                'target_recall': s6['target_recall'],
                'trained_at': datetime.now().isoformat(timespec='seconds'),
                'n_train': int(len(data['y_train'])),
                'features': data['feature_names'],
                'cv_auc': float(s6['cv_auc']),
                'test_auc': float(s7['test_auc']),
                'sklearn_version': sklearn.__version__}
    saved_to = save_model(final, metadata, path) if save else None

    # Monitoring demo: normal batch (test cases) and a simulated batch from a new, mis-calibrated device
    shifted = data['X_test'].copy()
    for col in [c for c in shifted.columns if 'radius' in c or 'perimeter' in c]:
        shifted[col] = shifted[col] * 1.15
    for col in [c for c in shifted.columns if 'area' in c]:
        shifted[col] = shifted[col] * 1.15 ** 2
    return {'metadata': metadata, 'saved_to': saved_to,
            'drift_normal': drift_report(data['X_train'], data['X_test']),
            'drift_shifted': drift_report(data['X_train'], shifted),
            'performance': performance_check(final, data['X_test'], data['y_test'], s6['threshold']),
            'example_predictions': predict_cases(final, data['X_test'].head(10), s6['threshold'])
                                   .assign(actual=data['y_test'].head(10).values)}


# ============================================================================ run everything
def run_all(save_model_file=True):
    s1 = step1_big_picture()
    s2 = step2_get_data()
    s3 = step3_explore(s2)
    s4 = step4_prepare(s2)
    s5 = step5_select_train(s2)
    s6 = step6_fine_tune(s2, s5)
    s7 = step7_present(s2, s5, s6)
    s8 = step8_launch(s2, s5, s6, s7, save=save_model_file)
    return {'s1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5, 's6': s6, 's7': s7, 's8': s8}


# ============================================================================ figures
def fig_class_balance(table):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(table['class'], table['count'], color=['tab:green', 'tab:red'], edgecolor='k')
    ax.set(ylabel='number of observations', title='Class distribution')
    fig.tight_layout()
    return fig


def fig_feature_scales(scales):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(scales.index, scales['max'], color='tab:blue', edgecolor='k')
    ax.set_xscale('log')
    ax.set(xlabel='maximum value (log scale)', title='Raw feature scales differ by orders of magnitude')
    fig.tight_layout()
    return fig


def fig_top_features_by_class(X, y, features):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, col in zip(axes.ravel(), features[:6]):
        bins = np.histogram_bin_edges(X[col], bins=30)
        ax.hist(X.loc[y == 0, col], bins=bins, alpha=0.6, label='benign', color='tab:green', density=True)
        ax.hist(X.loc[y == 1, col], bins=bins, alpha=0.6, label='malignant', color='tab:red', density=True)
        ax.set(title=col, ylabel='density')
        ax.legend()
    fig.suptitle('Most discriminative features (training set)')
    fig.tight_layout()
    return fig


def fig_feature_corr(feat_corr):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(feat_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feat_corr)))
    ax.set_xticklabels(feat_corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(feat_corr)))
    ax.set_yticklabels(feat_corr.columns, fontsize=8)
    fig.colorbar(im)
    ax.set_title('Correlation between the 30 features (training set)')
    fig.tight_layout()
    return fig


def fig_model_comparison(comparison):
    fig, ax = plt.subplots(figsize=(11, 5))
    comparison[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].plot.bar(ax=ax, edgecolor='k')
    ax.set(ylim=(0, 1.05), ylabel='3-fold CV score', title='Candidate classifiers (training set)')
    ax.set_xticklabels(comparison.index, rotation=0)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def fig_confusion(cm, title, cmap='Blues'):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['benign (0)', 'malignant (1)']).plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_pr(y_true, scores, threshold):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    d = metrics_at_threshold(y_true, scores, 0.0)
    s = metrics_at_threshold(y_true, scores, threshold)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].plot(thresholds, precisions[:-1], 'b-', lw=2, label='precision')
    axes[0].plot(thresholds, recalls[:-1], 'g-', lw=2, label='recall')
    axes[0].axvline(0, color='k', ls=':', label='default threshold (0)')
    axes[0].axvline(threshold, color='r', ls='--', label=f'selected threshold ({threshold:.2f})')
    axes[0].set(xlabel='threshold (decision score)', ylabel='value', title='Precision and recall vs threshold')
    axes[0].legend(loc='center left')
    axes[0].grid(True)
    axes[1].plot(recalls, precisions, lw=2)
    axes[1].plot(d['recall'], d['precision'], 'ko', ms=9, label='default threshold')
    axes[1].plot(s['recall'], s['precision'], 'r^', ms=10, label='selected threshold')
    axes[1].set(xlabel='recall', ylabel='precision', title='Precision-Recall curve (3-fold CV scores)')
    axes[1].legend(loc='lower left')
    axes[1].grid(True)
    fig.tight_layout()
    return fig


def fig_roc(y_true, scores, threshold):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, lw=2, label=f'SGD classifier (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='random classifier (AUC = 0.5)')
    for t, style, lab in [(0.0, 'ko', 'default threshold (0)'), (threshold, 'r^', f'selected threshold ({threshold:.2f})')]:
        m = metrics_at_threshold(y_true, scores, t)
        ax.plot(m['FP'] / (m['FP'] + m['TN']), m['recall'], style, ms=9, label=lab)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate (recall)', title='ROC curve (3-fold CV scores)')
    ax.legend(loc='lower right')
    ax.grid(True)
    fig.tight_layout()
    return fig


def fig_weights(weights, top=15):
    w = weights.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(w.index, w.values, color=np.where(w.values > 0, 'tab:red', 'tab:green'), edgecolor='k')
    ax.set(xlabel='weight (standardised features; > 0 pushes towards malignant)',
           title=f'Top {top} weights of the tuned SGD model')
    fig.tight_layout()
    return fig


# ============================================================================ main
def setup_display():
    """Inside Spyder / Jupyter (IPython): force the inline backend so every figure goes to Spyder's
    Plots pane (or the notebook). Returns 'ipython' there, 'terminal' otherwise."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except ImportError:
        ip = None
    if ip is not None:
        ip.run_line_magic('matplotlib', 'inline')
        return 'ipython'
    return 'terminal'


def show_figure(fig, mode, keep_open=True):
    """Display a figure: inline in Spyder/Jupyter, or keep it open for plt.show() in a terminal."""
    if mode == 'ipython':
        from IPython.display import display
        display(fig)
        plt.close(fig)
    elif not keep_open:
        plt.close(fig)


def save_individual_axes(fig, stem, folder):
    """Save every subplot of a multi-panel figure as its own PNG (figures/individual/<stem>_01.png ...)."""
    panels = [ax for ax in fig.axes if ax.axison and ax.get_label() != '<colorbar>']
    if len(panels) < 2:
        return []
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    paths = []
    for i, ax in enumerate(panels, 1):
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(0.15)
        path = folder / f'{stem}_{i:02d}.png'
        fig.savefig(path, dpi=120, bbox_inches=bbox)
        paths.append(path)
    return paths


def show_images(paths, mode):
    """Show saved individual plots one by one (Spyder Plots pane / notebook)."""
    if mode != 'ipython':
        return
    from IPython.display import Image, display
    for p in paths:
        display(Image(filename=str(p)))


def main():
    parser = argparse.ArgumentParser(description='DAMA61 WA1 - Problem 2 (end-to-end ML project)')
    parser.add_argument('--no-show', action='store_true', help='only save the figures, do not display them')
    parser.add_argument('--outdir', default='figures', help='folder for the saved figures')
    args, _ = parser.parse_known_args()          # tolerate extra args passed by Spyder / IPython
    mode = setup_display()
    if mode == 'terminal' and args.no_show:
        matplotlib.use('Agg')
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', 20)

    def save(fig, name, split=True):
        """Save the combined figure + each subplot separately; in Spyder show the combined
        figure first and then every individual plot one by one."""
        fig.savefig(outdir / f'p2_{name}.png', dpi=120, bbox_inches='tight')
        parts = save_individual_axes(fig, f'p2_{name}', outdir / 'individual') if split else []
        show_figure(fig, mode, keep_open=not args.no_show)
        show_images(parts, mode)

    def header(text):
        print('\n' + '=' * 90 + '\n' + text + '\n' + '=' * 90)

    # Each step is computed right before it is printed, so its figures appear immediately.
    s1 = step1_big_picture()
    header('STEP 1 - Look at the big picture')
    for k, v in s1.items():
        print(f'- {k}: {v}')

    s2 = step2_get_data()
    header('STEP 2 - Get the data  [Q1, Q2]')
    print('Predictors:', s2['X'].shape, '| missing values:', s2['missing'])
    print('Feature names:', s2['feature_names'])
    print('Original target names:', s2['original_target_names'], '-> recoded: malignant = 1, benign = 0')
    print(s2['recode_check'])
    print(s2['class_table'])
    print('Stratified 80/20 split:\n', s2['split_table'])
    save(fig_class_balance(s2['class_table']), 'step2_class_balance')

    s3 = step3_explore(s2)
    header('STEP 3 - Discover and visualise the data (training set)  [Q1, Q2]')
    print('Feature scales:\n', pd.concat([s3['scales'].head(4), s3['scales'].tail(4)]).round(4))
    print('Correlation with malignant:\n', s3['corr_target'].head(10).round(3))
    print('Feature pairs with |r| > 0.9:', s3['n_pairs_above_0_9'])
    print(s3['high_corr_pairs'].head(8).round(3).to_string(index=False))
    save(fig_feature_scales(s3['scales']), 'step3_feature_scales')
    save(fig_top_features_by_class(s2['X_train'], s2['y_train'], list(s3['corr_target'].index)), 'step3_top_features')
    save(fig_feature_corr(s3['feature_corr']), 'step3_feature_correlation')

    s4 = step4_prepare(s2)
    header('STEP 4 - Prepare the data for ML algorithms  [Q2]')
    print('Before scaling:\n', s4['before'].head(6).round(3))
    print('After scaling (mean ~0, std ~1):\n', s4['after'].head(6).round(3))

    s5 = step5_select_train(s2)
    header('STEP 5 - Select a model and train it  [Q2, Q3, Q4]')
    print(s5['sgd_fitted'])
    for i, a in enumerate(s5['cv_acc'], 1):
        print(f'Fold {i}: accuracy = {a:.4f}')
    print(f"Mean = {s5['cv_acc'].mean():.4f}, std = {s5['cv_acc'].std():.4f}")
    print(f"Accuracy of CV predictions = {s5['cv_pred_acc']:.4f} | resubstitution = {s5['resub_acc']:.4f} "
          f"| always-benign baseline = {s5['baseline_acc']:.4f}")
    tn, fp, fn, tp = s5['cm'].ravel()
    print(f'Confusion matrix: TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}')
    print(f"Precision = {s5['precision']:.4f}, Recall = {s5['recall']:.4f}, F1 = {s5['f1']:.4f}")
    print('\nCandidate models (3-fold CV):\n', s5['comparison'].round(4))
    save(fig_confusion(s5['cm'], 'Confusion matrix - 3-fold CV (train)'), 'step5_confusion_matrix')
    save(fig_model_comparison(s5['comparison']), 'step5_model_comparison')

    s6 = step6_fine_tune(s2, s5)
    header('STEP 6 - Fine-tune the model  [Q5, Q6]')
    thr = s6['threshold']
    print(s6['threshold_table'])
    d, s = s6['default_cv'], s6['selected_cv']
    print(f"Default threshold 0      : precision = {d['precision']:.4f}, recall = {d['recall']:.4f}")
    print(f"Selected threshold {thr:.3f}: precision = {s['precision']:.4f}, recall = {s['recall']:.4f}")
    print('Selected threshold is', 'LOWER' if thr < 0 else 'HIGHER (or equal)', 'than the default.')
    print(f"CV ROC AUC = {s6['cv_auc']:.4f}")
    print('\nHyper-parameter search (ROC AUC):\n', s6['search_results'].head(8).round(4).to_string(index=False))
    print('Best params:', s6['best_params'], f"| tuned CV AUC = {s6['tuned_cv_auc']:.4f}, "
          f"tuned recall-95 threshold = {s6['tuned_threshold']:.3f}")
    save(fig_pr(s2['y_train'], s6['cv_scores'], thr), 'step6_precision_recall')
    save(fig_roc(s2['y_train'], s6['cv_scores'], thr), 'step6_roc')
    save(fig_weights(s6['weights']), 'step6_weights')

    s7 = step7_present(s2, s5, s6)
    header('STEP 7 - Present the solution (test set)  [Q6]')
    print(s7['results'])
    print(f"Test ROC AUC = {s7['test_auc']:.4f} (tuned: {s7['tuned_test_auc']:.4f})")
    print('Key findings:')
    for f in KEY_FINDINGS:
        print(' -', f)
    y_test = s2['y_test']
    save(fig_confusion(confusion_matrix(y_test, (s7['test_scores'] >= 0).astype(int)),
                       'Test set - default threshold'), 'step7_test_cm_default')
    save(fig_confusion(confusion_matrix(y_test, (s7['test_scores'] >= thr).astype(int)),
                       f'Test set - Q5 threshold ({thr:.2f})', cmap='Reds'), 'step7_test_cm_q5')

    s8 = step8_launch(s2, s5, s6, s7, save=True)
    header('STEP 8 - Launch, monitor and maintain')
    print('Model saved to:', s8['saved_to'])
    print(json.dumps({k: v for k, v in s8['metadata'].items() if k != 'features'}, indent=2, default=str))
    print('Example predictions:\n', s8['example_predictions'])
    print('Drift - normal batch (features flagged):', int(s8['drift_normal']['drift'].sum()))
    print('Drift - simulated mis-calibrated device:\n', s8['drift_shifted'].head(10))
    print('Performance check:', s8['performance'])
    print('Maintenance plan:')
    for p in MAINTENANCE_PLAN:
        print(' -', p)

    print(f'\nFigures saved to: {outdir.resolve()}')
    if mode == 'terminal' and not args.no_show:
        plt.show()                                # terminal: open all figure windows at the end


if __name__ == '__main__':
    main()
