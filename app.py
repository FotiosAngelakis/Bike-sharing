"""

"""
import io

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

import gc
import os
os.environ.setdefault('WA1_N_JOBS', '1')   # one process at a time: Streamlit Cloud has ~1 GB memory

import problem1 as P1
import problem2 as P2

st.set_page_config(page_title='DAMA61 WA1 - End-to-end ML', layout='wide')

STEPS = ['1 · Big picture', '2 · Get the data', '3 · Explore', '4 · Prepare',
         '5 · Select & train', '6 · Fine-tune', '7 · Present', '8 · Launch & monitor']


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def model_download(model, metadata, filename):
    buf = io.BytesIO()
    joblib.dump({'model': model, 'metadata': metadata}, buf)
    st.download_button('Download trained model (.joblib)', buf.getvalue(), file_name=filename)


@st.cache_resource(show_spinner='Running Problem 1 (incl. grid search, ~1 min the first time)...')
def get_p1():
    r = P1.run_all(save_model_file=False)
    r['s6'].pop('search', None)            # the GridSearchCV object (all fitted candidates) is no longer needed
    gc.collect()
    return r


@st.cache_resource(show_spinner='Running Problem 2...')
def get_p2():
    r = P2.run_all(save_model_file=False)
    r['s6'].pop('search', None)
    gc.collect()
    return r


page = st.sidebar.radio('Problem', ['Problem 1 - Bike Sharing (regression)',
                                    'Problem 2 - Breast Cancer (classification)'])
st.sidebar.markdown('---')
st.sidebar.markdown('**End-to-end ML checklist** (Geron, ch. 2)')
for s in STEPS:
    st.sidebar.markdown(f'- {s}')
st.sidebar.caption('DAMA61 - Written Assignment 1 (2026-27)')

# =========================================================================== PROBLEM 1
if page.startswith('Problem 1'):
    st.title('Problem 1 - Predicting daily bike-sharing demand')
    r = get_p1()
    s1, s2, s3, s4, s5, s6, s7, s8 = (r[f's{i}'] for i in range(1, 9))
    tabs = st.tabs(STEPS)

    with tabs[0]:
        st.subheader('Step 1 - Look at the big picture  (Q1)')
        for k, v in s1.items():
            st.markdown(f'**{k}.** {v}')

    with tabs[1]:
        st.subheader('Step 2 - Get the data  (Q1, Q2, Q5)')
        ov = s2['inspection']
        st.markdown('**Fetch** - download the zip from UCI, extract it, load `day.csv` with pandas:')
        st.code("DOWNLOAD_URL = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'\n"
                "def fetch_bike_data(url=DOWNLOAD_URL, data_dir=Path('datasets/bike_sharing')):\n"
                "    data_dir.mkdir(parents=True, exist_ok=True)\n"
                "    zip_path = data_dir / 'bike_sharing_dataset.zip'\n"
                "    urllib.request.urlretrieve(url, zip_path)\n"
                "    with zipfile.ZipFile(zip_path) as archive:\n"
                "        archive.extractall(data_dir)\n"
                "df = pd.read_csv(data_dir / 'day.csv')", language='python')
        st.caption(f"Data source in this run: {s2['source']}")
        c = st.columns(5)
        c[0].metric('Days', int(ov['shape'][0]))
        c[1].metric('Columns', int(ov['shape'][1]))
        c[2].metric('Empty cells', ov['missing_total'])
        c[3].metric('Duplicated rows', ov['duplicates'])
        c[4].metric('casual + registered = cnt', str(ov['sum_check']))
        st.markdown('**Inspect the data**')
        st.dataframe(ov['head'])
        st.text(ov['info'])
        st.markdown('**Meaning of each column**')
        st.dataframe(s2['column_info'])
        st.dataframe(ov['summary'])
        st.dataframe(ov['describe'].round(3))
        st.markdown('**Missing data**')
        st.dataframe(ov['implausible'])
        st.markdown('\n'.join(f'- {p}' for p in s2['missing_plan']))
        st.markdown('**Histogram of every numerical attribute**')
        show(P1.fig_all_histograms(s2['df']))
        st.markdown('**Summary**\n' + '\n'.join(f'- {p}' for p in s2['data_summary']))
        st.markdown('**casual / registered vs cnt (leakage check)**')
        st.dataframe(ov['corr_counts'].round(3))
        show(P1.fig_q2_counts(s2['df']))
        st.markdown(f'**Excluded:** `{P1.EXCLUDED}` - identifiers (`instant`, `dteday`) and target components '
                    '(`casual`, `registered`), which would leak the answer.')
        st.markdown('**Test set created immediately** (80/20, shuffled, random_state=42) before exploring, '
                    'to avoid data-snooping bias.')
        c = st.columns(2)
        c[0].metric('Training days', len(s2['y_train']))
        c[1].metric('Test days', len(s2['y_test']))
        st.dataframe(s2['target_summary'].round(1))
        st.write(f"KS test: statistic = {s2['ks'].statistic:.3f}, p-value = {s2['ks'].pvalue:.3f}")
        show(P1.fig_q5_hist(s2['y'], s2['y_train'], s2['y_test']))

    with tabs[2]:
        st.subheader('Step 3 - Discover and visualise the data (training set)  (Q3, Q4)')
        sk = s3['skew']
        c = st.columns(3)
        c[0].dataframe(sk['numerical'].round(3).to_frame('skew'))
        c[1].dataframe(sk['categorical'].round(3).to_frame('skew'))
        c[2].dataframe(sk['windspeed_versions'].round(3).to_frame('windspeed skew'))
        show(P1.fig_q3_grid(s3['train_df']))
        show(P1.fig_q3_wind(s3['train_df']))
        st.markdown('**Looking for correlations**')
        st.dataframe(s3['assoc'].round(3))
        show(P1.fig_q4_heatmap(s3['pearson']))
        show(P1.fig_scatter_matrix(s3['train_df']))
        show(P1.fig_q4_scatter(s3['train_df']))
        show(P1.fig_q4_box(s3['train_df']))
        st.markdown('**Experimenting with attribute combinations**')
        st.write(f"Comfort temperature found on the training set: {s3['comfort_c']} °C")
        st.dataframe(s3['experiments'].round(3))
        show(P1.fig_attribute_experiments(s3['train_df'], s3['experiments'], s3['comfort_c'], s3['comfort_curve']))
        st.markdown('\n'.join(f'- {f}' for f in s3['attribute_findings']))
        st.markdown('**Extra calendar information (Christmas, Thanksgiving, July 4, summer break, bridge days)**')
        ev = s3['events']
        st.dataframe(ev['info'])
        st.dataframe(ev['table'])
        st.dataframe(ev['comparison'])
        show(P1.fig_events(ev))

    with tabs[3]:
        st.subheader('Step 4 - Prepare the data for ML algorithms  (Q3, Q6)')
        st.markdown('**Do we need cleaning, transformations, scaling?**')
        st.table(s4['decisions'])
        st.write(f"hum = 0 values in the training set: {s4['hum_missing_train']} → imputed with the training "
                 f"median {s4['hum_median']:.4f}")
        st.code(str(s4['preprocessor']))
        st.write('Prepared training matrix shape:', s4['prepared_shape'])
        st.dataframe(s4['scaler_params'].round(4))
        st.dataframe(s4['prepared_head'].round(3))

    with tabs[4]:
        st.subheader('Step 5 - Select a model and train it  (Q6, Q8)')
        st.code(str(s5['linear_model']))
        st.write(f"Linear Regression: intercept = {s5['intercept']:.1f}, training R² = {s5['train_r2']:.3f}")
        st.dataframe(s5['coefs'].round(1).to_frame('coefficient'))
        st.write('Candidate models - 10-fold cross-validation on the training set:')
        st.dataframe(s5['comparison'].round(3))
        show(P1.fig_model_comparison(s5['comparison']))

    with tabs[5]:
        st.subheader('Step 6 - Fine-tune the model (grid search, 10-fold CV)')
        c = st.columns(2)
        c[0].metric('Best model', s6['best_name'])
        c[1].metric('Best CV RMSE', f"{s6['best_cv_rmse']:.0f}")
        st.write('Best parameters:', s6['best_params'])
        tbl = s6['results'].copy()
        tbl['params'] = tbl['params'].astype(str)
        st.dataframe(tbl.round(2))
        show(P1.fig_importance(s6['importance']))

    with tabs[6]:
        st.subheader('Step 7 - Present the solution (test set)  (Q7, Q8)')
        st.dataframe(s7['metrics'].round(3))
        st.dataframe(s7['rmse_ci'].round(1))
        lin_label, best_label = s7['labels']
        show(P1.fig_actual_vs_predicted(s2['y_test'], s7['y_pred_linear'], lin_label))
        show(P1.fig_actual_vs_predicted(s2['y_test'], s7['y_pred_best'], best_label))
        st.write('Largest errors of the linear model:')
        st.dataframe(s7['errors'].head(10))
        cv = s7['cv_r2_linear']
        c = st.columns(3)
        c[0].metric('Linear CV R² (mean)', f'{cv.mean():.3f}')
        c[1].metric('Linear CV R² (std)', f'{cv.std():.3f}')
        c[2].metric('Linear test R²', f"{s7['test_r2_linear']:.3f}")
        show(P1.fig_q8(cv, s7['test_r2_linear']))
        st.markdown('#### 7.4 Time-based validation (extra)')
        tv = s7['time']
        st.markdown('The random split of Q5 lets the model learn from days that come *after* the test days. '
                    'A real forecast only knows the past, so here the model is trained on the earliest 80% '
                    f"of days ({tv['train_period'][0]} to {tv['train_period'][1]}) and tested on the latest "
                    f"20% ({tv['test_period'][0]} to {tv['test_period'][1]}).")
        st.dataframe(tv['comparison'].round(3))
        st.dataframe(tv['tscv_folds'].round(3))
        c = st.columns(2)
        c[0].metric('Shuffled 10-fold CV R² (linear)', f"{tv['kfold_r2'][0]:.3f} ± {tv['kfold_r2'][1]:.3f}")
        c[1].metric('TimeSeriesSplit CV R² (linear)', f"{tv['tscv_r2'][0]:.3f} ± {tv['tscv_r2'][1]:.3f}")
        show(P1.fig_time_validation(tv))
        st.markdown('**Key findings**\n' + '\n'.join(f'- {f}' for f in P1.KEY_FINDINGS))

    with tabs[7]:
        st.subheader('Step 8 - Launch, monitor and maintain')
        st.markdown('#### Try the deployed model')
        with st.form('p1_predict'):
            c = st.columns(4)
            season = c[0].selectbox('Season', [1, 2, 3, 4], format_func=lambda v: {1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'}[v])
            yr = c[1].selectbox('Year', [0, 1], index=1, format_func=lambda v: str(2011 + v))
            mnth = c[2].selectbox('Month', list(range(1, 13)), index=5)
            weekday = c[3].selectbox('Weekday', list(range(7)), index=3,
                                     format_func=lambda v: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][v])
            c = st.columns(4)
            holiday = c[0].selectbox('Holiday', [0, 1])
            weathersit = c[1].selectbox('Weather', [1, 2, 3], format_func=lambda v: {1: 'clear', 2: 'mist/cloudy', 3: 'light rain/snow'}[v])
            temp_c = c[2].slider('Temperature (°C)', -8.0, 39.0, 22.0)
            atemp_c = c[3].slider('Feels-like (°C)', -16.0, 50.0, 24.0)
            c = st.columns(2)
            hum_pct = c[0].slider('Humidity (%)', 0.0, 100.0, 60.0)
            wind_kmh = c[1].slider('Wind speed (km/h)', 0.5, 67.0, 12.0)
            submitted = st.form_submit_button('Predict demand')
        if submitted:
            day = pd.DataFrame([{'temp': temp_c / 41, 'atemp': atemp_c / 50, 'hum': hum_pct / 100,
                                 'windspeed': wind_kmh / 67, 'yr': yr, 'holiday': holiday,
                                 'workingday': int(weekday in [1, 2, 3, 4, 5] and holiday == 0),
                                 'season': season, 'mnth': mnth, 'weekday': weekday, 'weathersit': weathersit}])
            pred = P1.predict_demand(s6['best_model'], day)[0]
            st.success(f'Predicted rentals: **{pred:,.0f} bikes** ({s6["best_name"]})')
        model_download(s6['best_model'], s8['metadata'], 'problem1_model.joblib')
        st.json(s8['metadata'])
        st.markdown('#### Monitoring')
        c = st.columns(2)
        c[0].write('Input drift - normal batch (test days)')
        c[0].dataframe(s8['drift_normal'])
        c[1].write('Input drift - simulated hotter / drier batch')
        c[1].dataframe(s8['drift_shifted'])
        st.write('Performance check (when true counts arrive):', s8['performance'])
        st.markdown('#### Maintenance plan\n' + '\n'.join(f'- {p}' for p in P1.MAINTENANCE_PLAN))

# =========================================================================== PROBLEM 2
else:
    st.title('Problem 2 - Classifying breast masses (malignant = 1)')
    r = get_p2()
    s1, s2, s3, s4, s5, s6, s7, s8 = (r[f's{i}'] for i in range(1, 9))
    tabs = st.tabs(STEPS)

    with tabs[0]:
        st.subheader('Step 1 - Look at the big picture')
        for k, v in s1.items():
            st.markdown(f'**{k}.** {v}')

    with tabs[1]:
        st.subheader('Step 2 - Get the data  (Q1, Q2)')
        c = st.columns(3)
        c[0].metric('Observations', int(s2['X'].shape[0]))
        c[1].metric('Features', int(s2['X'].shape[1]))
        c[2].metric('Missing values', s2['missing'])
        st.dataframe(s2['X'].head())
        st.write('Original target names:', s2['original_target_names'], '→ recoded: malignant = 1, benign = 0')
        st.dataframe(s2['recode_check'])
        st.dataframe(s2['class_table'])
        show(P2.fig_class_balance(s2['class_table']))
        st.write('Stratified 80/20 split (random_state=42):')
        st.dataframe(s2['split_table'])

    with tabs[2]:
        st.subheader('Step 3 - Discover and visualise the data (training set)')
        show(P2.fig_feature_scales(s3['scales']))
        c = st.columns(2)
        c[0].write('Correlation of each feature with malignant')
        c[0].dataframe(s3['corr_target'].round(3))
        c[1].write(f"Highly correlated feature pairs (|r| > 0.9: {s3['n_pairs_above_0_9']})")
        c[1].dataframe(s3['high_corr_pairs'].head(15).round(3))
        show(P2.fig_top_features_by_class(s2['X_train'], s2['y_train'], list(s3['corr_target'].index)))
        show(P2.fig_feature_corr(s3['feature_corr']))

    with tabs[3]:
        st.subheader('Step 4 - Prepare the data for ML algorithms  (Q2)')
        st.markdown('All 30 features are numeric with no missing values, so the only preparation is '
                    '**standardisation** (mean 0, std 1), fitted on the training set inside the pipeline. '
                    'SGD uses one learning rate for every weight and L2 regularisation, so features must be on a common scale.')
        c = st.columns(2)
        c[0].write('Before scaling')
        c[0].dataframe(s4['before'].round(3))
        c[1].write('After scaling')
        c[1].dataframe(s4['after'].round(3))

    with tabs[4]:
        st.subheader('Step 5 - Select a model and train it  (Q2, Q3, Q4)')
        st.code(str(s5['sgd_fitted']))
        acc = s5['cv_acc']
        st.dataframe(pd.Series(acc, index=['fold 1', 'fold 2', 'fold 3']).round(4).to_frame('accuracy').T)
        c = st.columns(4)
        c[0].metric('Mean CV accuracy', f'{acc.mean():.4f}')
        c[1].metric('Std', f'{acc.std():.4f}')
        c[2].metric('Resubstitution accuracy', f"{s5['resub_acc']:.4f}")
        c[3].metric('Always-benign baseline', f"{s5['baseline_acc']:.4f}")
        tn, fp, fn, tp = (int(v) for v in s5['cm'].ravel())
        c = st.columns(7)
        for col, (lab, val) in zip(c, [('TP', tp), ('TN', tn), ('FP', fp), ('FN', fn),
                                       ('Precision', f"{s5['precision']:.4f}"), ('Recall', f"{s5['recall']:.4f}"),
                                       ('F1', f"{s5['f1']:.4f}")]):
            col.metric(lab, val)
        c = st.columns(2)
        with c[0]:
            show(P2.fig_confusion(s5['cm'], 'Confusion matrix - 3-fold CV (train)'))
        st.write('Candidate classifiers (3-fold CV):')
        st.dataframe(s5['comparison'].round(4))
        show(P2.fig_model_comparison(s5['comparison']))

    with tabs[5]:
        st.subheader('Step 6 - Fine-tune the model  (Q5, Q6)')
        thr = s6['threshold']
        d, s = s6['default_cv'], s6['selected_cv']
        c = st.columns(3)
        c[0].metric('Recall-95 threshold', f'{thr:.3f}', help='Default threshold = 0')
        c[1].metric('Precision / recall @ 0', f"{d['precision']:.3f} / {d['recall']:.3f}")
        c[2].metric('Precision / recall @ selected', f"{s['precision']:.3f} / {s['recall']:.3f}")
        st.dataframe(s6['threshold_table'])
        show(P2.fig_pr(s2['y_train'], s6['cv_scores'], thr))
        c = st.columns(2)
        with c[0]:
            show(P2.fig_roc(s2['y_train'], s6['cv_scores'], thr))
        c[1].metric('CV ROC AUC', f"{s6['cv_auc']:.4f}")
        st.markdown('#### Hyper-parameter search (SGD, scoring = ROC AUC)')
        st.write('Best parameters:', s6['best_params'], f"- tuned CV AUC = {s6['tuned_cv_auc']:.4f}")
        st.dataframe(s6['search_results'].round(4))
        show(P2.fig_weights(s6['weights']))

    with tabs[6]:
        st.subheader('Step 7 - Present the solution (test set)  (Q6)')
        c = st.columns(2)
        c[0].metric('Test ROC AUC', f"{s7['test_auc']:.4f}")
        c[1].metric('Tuned model test ROC AUC', f"{s7['tuned_test_auc']:.4f}")
        st.dataframe(s7['results'])
        y_test = s2['y_test']
        c = st.columns(2)
        with c[0]:
            show(P2.fig_confusion(confusion_matrix(y_test, (s7['test_scores'] >= 0).astype(int)),
                                  'Test set - default threshold'))
        with c[1]:
            show(P2.fig_confusion(confusion_matrix(y_test, (s7['test_scores'] >= s6['threshold']).astype(int)),
                                  f"Test set - Q5 threshold ({s6['threshold']:.2f})", cmap='Reds'))
        st.markdown('**Key findings**\n' + '\n'.join(f'- {f}' for f in P2.KEY_FINDINGS))

    with tabs[7]:
        st.subheader('Step 8 - Launch, monitor and maintain')
        st.markdown('#### Try the deployed model on unseen test cases')
        idx = st.selectbox('Test case', list(s2['X_test'].index))
        lo, hi = float(np.floor(s7['test_scores'].min())), float(np.ceil(s7['test_scores'].max()))
        op_thr = st.slider('Operating threshold', lo, hi, float(np.clip(s6['threshold'], lo, hi)), 0.1)
        res = P2.predict_cases(s5['sgd_fitted'], s2['X_test'].loc[[idx]], op_thr).iloc[0]
        actual = 'malignant' if s2['y_test'].loc[idx] == 1 else 'benign'
        (st.error if res['label'] == 'malignant' else st.success)(
            f"Prediction: **{res['label']}** (score {res['score']:.2f}) - actual diagnosis: {actual}")
        m_live = P2.metrics_at_threshold(s2['y_test'], s7['test_scores'], op_thr)
        st.dataframe(pd.Series(m_live).to_frame('test set at this threshold').T.round(4))
        model_download(s5['sgd_fitted'], s8['metadata'], 'problem2_model.joblib')
        st.json({k: v for k, v in s8['metadata'].items() if k != 'features'})
        st.markdown('#### Monitoring')
        c = st.columns(2)
        c[0].write(f"Input drift - normal batch: {int(s8['drift_normal']['drift'].sum())} features flagged")
        c[0].dataframe(s8['drift_normal'].head(10))
        c[1].write(f"Input drift - simulated mis-calibrated device: {int(s8['drift_shifted']['drift'].sum())} features flagged")
        c[1].dataframe(s8['drift_shifted'].head(10))
        st.write('Performance check (when biopsy results arrive):', s8['performance'])
        st.markdown('#### Maintenance plan\n' + '\n'.join(f'- {p}' for p in P2.MAINTENANCE_PLAN))
