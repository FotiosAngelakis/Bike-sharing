"""
DAMA61 - Written Assignment 1 - Problem 1: Bike Sharing demand (regression)

End-to-end Machine Learning project following A. Geron, "Hands-On Machine Learning
with Scikit-Learn, Keras & TensorFlow", Chapter 2:

    Step 1  Look at the big picture                  (assignment Q1)
    Step 2  Get the data: fetch, inspect, missing values, histograms, test set   (Q1, Q2, Q5)
    Step 3  Discover and visualise: correlations, attribute combinations          (Q3, Q4)
    Step 4  Prepare the data: cleaning, transformations, scaling, pipeline        (Q3, Q6)
    Step 5  Select a model and train it              (Q6, Q8)
    Step 6  Fine-tune the model                      (beyond the assignment)
    Step 7  Present the solution                     (Q7, Q8)
    Step 8  Launch, monitor and maintain the system  (beyond the assignment)

Run from the WA1 folder (inside the virtual environment, see README.md):
    python problem1.py            # prints every step, shows + saves figures (./figures), model (./models)
    python problem1.py --no-show  # only save the figures
In Spyder just press F5: figures appear in the Plots pane as each step finishes.

The functions are also imported by the Streamlit app (app.py).
"""
from pathlib import Path
from datetime import datetime
import argparse
import os
import io
import json
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy import stats

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (r2_score, mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error)
from sklearn.base import clone
from sklearn.model_selection import (train_test_split, cross_validate, cross_val_predict, KFold,
                                     GridSearchCV, TimeSeriesSplit)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

plt.rcParams.update({'font.size': 11})

RANDOM_STATE = 42
# Parallel jobs: all CPU cores locally (-1); app.py sets WA1_N_JOBS=1 on Streamlit Cloud (1 GB memory limit)
N_JOBS = int(os.environ.get('WA1_N_JOBS', '-1'))
ROOT = Path(__file__).resolve().parent
DOWNLOAD_URL = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
DATA_DIR = ROOT / 'datasets' / 'bike_sharing'                     # where the download is extracted
LOCAL_COPY = ROOT / 'bike+sharing+dataset' / 'day.csv'             # fallback if offline
MODEL_PATH = ROOT / 'models' / 'problem1_model.joblib'

TARGET = 'cnt'
EXCLUDED = ['instant', 'dteday', 'casual', 'registered']            # identifiers + leakage (Q2)
CAT_FEATURES = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
NUM_FEATURES = ['temp', 'atemp', 'hum', 'windspeed']
NUMERICAL_COLUMNS = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']

NUM_RAW = ['temp', 'atemp', 'hum']                                   # impute + standardise
WIND = ['windspeed']                                                 # sqrt + standardise
NOMINAL = ['season', 'mnth', 'weekday', 'weathersit']                # one-hot encoded
BINARY = ['yr', 'holiday', 'workingday']                             # already 0/1
FEATURES = NUM_RAW + WIND + BINARY + NOMINAL
NUMERIC_EXPLORE = ['temp', 'atemp', 'hum', 'windspeed_sqrt']

# Extra calendar information derived from dteday (known in advance -> allowed, no leakage)
EVENT_FLAGS = ['christmas_newyear', 'thanksgiving', 'july4', 'summer_break', 'holiday_bridge']
EVENT_INFO = pd.DataFrame([
    ('christmas_newyear', 'Christmas / New Year period, 20 Dec - 2 Jan'),
    ('thanksgiving', 'Thanksgiving Thursday + Black Friday (24-25 Nov 2011, 22-23 Nov 2012)'),
    ('july4', 'Independence Day, 4 July'),
    ('summer_break', 'School summer vacation, approx. 20 Jun - 25 Aug (approximate D.C. school calendar)'),
    ('holiday_bridge', 'Working day directly before or after a public holiday'),
], columns=['flag', 'meaning']).set_index('flag')


# ============================================================================ STEP 1
BIG_PICTURE = {
    'End goal': 'Predict how many bikes will be rented on a given day (cnt) from calendar and weather '
                'information. The prediction feeds downstream decisions: how many bikes to have in service, '
                'staff for redistributing bikes between stations, and on which low-demand days to schedule '
                'maintenance.',
    'Current solution': 'Geron (Ch. 2, "Frame the Problem") asks what the current solution looks like, because '
                        'it gives a reference for performance and insights on how to solve the problem. The '
                        'assignment does not describe one, so we assume planning is done by experience and '
                        'historical averages, e.g. "a working day in June is usually about X rentals". We '
                        'reproduce this rule as a model (average cnt per month and working-day flag) and '
                        'evaluate it in Step 5 exactly like the ML models: the ML model is only worth '
                        'building and maintaining if it clearly beats it.',
    'Supervised, unsupervised or reinforcement?': 'Supervised: every day in the data comes with the correct answer '
                                                 '(the label cnt). Not unsupervised (we are not looking for '
                                                 'structure without labels) and not reinforcement learning (no '
                                                 'agent acting in an environment and receiving rewards).',
    'Classification or regression?': 'Regression: cnt is a numeric quantity (22 to 8714). More precisely a '
                                     'univariate (one target) multiple (several predictors) regression.',
    'Batch or online?': 'Batch (offline) learning: the data is small (731 rows) and arrives slowly (one row per '
                        'day), so the model can be retrained from scratch periodically, e.g. monthly. Online '
                        'learning would only be needed for a continuous fast data stream.',
    'Performance measure': 'RMSE (root mean squared error) as the main measure, in bikes/day; R2, MAE, MAPE and '
                           'MSE are also reported as the assignment requires.',
    'RMSE or MAE?': 'RMSE squares the errors, so it penalises large mistakes much more than MAE. For fleet '
                    'planning a day with a huge error (e.g. 3000 bikes too few) is far worse than several small '
                    'errors, so RMSE is preferred. MAE is more robust to outliers (storm days) and is reported '
                    'next to it; if MAE << RMSE, a few days have very large errors.',
    'Assumptions': '(1) Predictions are made before the day ends, from the calendar and the weather (forecast) '
                   'only, so casual and registered cannot be inputs (they are parts of the target = leakage). '
                   '(2) The downstream system needs a number, not a category (e.g. "low/medium/high" demand); '
                   'if it only needed categories, this would become a classification task. '
                   '(3) The weather values are normalised as documented (temp/41, atemp/50, hum/100, '
                   'windspeed/67). (4) The 2011-2012 behaviour of Washington D.C. users is representative of '
                   'the near future.',
}


def step1_big_picture():
    return BIG_PICTURE


# ============================================================================ STEP 2
COLUMN_INFO = pd.DataFrame([
    ('instant', 'integer', 'Record index 1..731 (row number).'),
    ('dteday', 'date', 'Calendar date, 2011-01-01 to 2012-12-31.'),
    ('season', 'categorical', '1 = spring, 2 = summer, 3 = fall, 4 = winter.'),
    ('yr', 'binary', '0 = 2011, 1 = 2012.'),
    ('mnth', 'categorical', 'Month 1-12.'),
    ('holiday', 'binary', '1 = public holiday (D.C. holiday schedule), else 0.'),
    ('weekday', 'categorical', 'Day of the week 0-6 (0 = Sunday).'),
    ('workingday', 'binary', '1 = neither weekend nor holiday, else 0.'),
    ('weathersit', 'categorical', '1 = clear/partly cloudy, 2 = mist/cloudy, 3 = light rain or snow, '
                                  '4 = heavy rain/snow/fog (never occurs).'),
    ('temp', 'numerical', 'Temperature in Celsius divided by 41 (max).'),
    ('atemp', 'numerical', 'Feeling ("feels-like") temperature in Celsius divided by 50 (max).'),
    ('hum', 'numerical', 'Relative humidity divided by 100.'),
    ('windspeed', 'numerical', 'Wind speed divided by 67 (max).'),
    ('casual', 'count', 'Rentals by casual (non-registered) users.'),
    ('registered', 'count', 'Rentals by registered users.'),
    ('cnt', 'count (target)', 'Total rentals = casual + registered.'),
], columns=['column', 'type', 'meaning / values']).set_index('column')

MISSING_DATA_PLAN = [
    'No cell is empty (isnull() = 0 everywhere), but one value is impossible: hum = 0 (0% relative '
    'humidity) on 2011-03-10. It is a sensor/recording error, i.e. a hidden missing value.',
    'Options (Geron): (1) drop the row, (2) drop the whole attribute, (3) impute a value (zero, mean, median).',
    'Choice: treat hum = 0 as missing and impute the MEDIAN humidity of the training set. Dropping the row '
    'would lose a valid target value; dropping hum loses a useful feature; the median is robust to outliers.',
    'Implemented inside the pipeline (BikeFeatureEngineer marks it as NaN, SimpleImputer(median) fills it), so '
    'the same rule is applied automatically to the test set and to any future data.',
]

DATA_SUMMARY = [
    '731 daily records, 16 columns, no empty cells and no duplicated rows; one implausible humidity value.',
    'Most columns are integer codes for categories (season, mnth, weekday, weathersit) or 0/1 flags '
    '(yr, holiday, workingday) - their histograms are bar-like and must not be treated as continuous.',
    'temp and atemp are broad and roughly symmetric (bimodal: winter vs summer); hum is roughly symmetric; '
    'windspeed has a long right tail (right-skewed).',
    'casual is strongly right-skewed (a few very busy leisure days); registered and cnt are broad and '
    'roughly symmetric. casual + registered = cnt exactly.',
    'The attributes are on very different scales (0-1 for the weather, thousands for the counts), and '
    'the weather variables are already normalised, so their units must be remembered when interpreting.',
]


def fetch_bike_data(url=DOWNLOAD_URL, data_dir=DATA_DIR):
    """Download the zip from the UCI repository (once) and extract it. Returns the path of day.csv."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / 'day.csv'
    if not csv_path.exists():
        zip_path = data_dir / 'bike_sharing_dataset.zip'
        if not zip_path.exists():
            urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(data_dir)
    return csv_path


def load_bike():
    """Fetch the data (download + extract); fall back to the local copy if there is no internet."""
    try:
        path, source = fetch_bike_data(), 'downloaded from UCI and extracted'
    except Exception as exc:                                   # offline, firewall, ...
        path, source = LOCAL_COPY, f'local copy (download failed: {type(exc).__name__})'
    df = pd.read_csv(path)
    df['dteday'] = pd.to_datetime(df['dteday'])
    return df, f'{source}: {path}'


def inspect(df):
    """Q1/Q2: structure, types, ranges, completeness, implausible values, leakage check."""
    buf = io.StringIO()
    df.info(buf=buf)
    summary = pd.DataFrame({'dtype': df.dtypes.astype(str),
                            'missing': df.isnull().sum(),
                            'n_unique': df.nunique(),
                            'min': df.min().astype(str),
                            'max': df.max().astype(str)})
    implausible = df.loc[df['hum'] <= 0, ['dteday', 'temp', 'hum', 'windspeed', 'weathersit', 'cnt']]
    return {
        'shape': df.shape,
        'head': df.head(),
        'info': buf.getvalue(),
        'summary': summary,
        'describe': df.describe().T,
        'missing_total': int(df.isnull().sum().sum()),
        'implausible': implausible,
        'value_counts': {c: df[c].value_counts().sort_index() for c in CAT_FEATURES},
        'duplicates': int(df.duplicated().sum()),
        'sum_check': bool((df['casual'] + df['registered'] == df['cnt']).all()),
        'instant_is_index': bool((df['instant'] == np.arange(1, len(df) + 1)).all()),
        'corr_counts': df[['casual', 'registered', 'cnt']].corr(),
    }


def step2_get_data(random_state=RANDOM_STATE):
    """Fetch, inspect, then create the test set right away (Q5)."""
    df, source = load_bike()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=random_state)
    return {
        'df': df, 'source': source,
        'inspection': inspect(df),
        'column_info': COLUMN_INFO,
        'missing_plan': MISSING_DATA_PLAN,
        'data_summary': DATA_SUMMARY,
        'X_train': X_train, 'X_test': X_test, 'y': y, 'y_train': y_train, 'y_test': y_test,
        'target_summary': pd.DataFrame({'train': y_train.describe(), 'test': y_test.describe()}),
        'ks': stats.ks_2samp(y_train, y_test),
    }


# ============================================================================ STEP 3
def skewness(df):
    return {
        'numerical': df[NUM_FEATURES + [TARGET]].skew().sort_values(key=abs, ascending=False),
        'categorical': df[CAT_FEATURES].skew(),
        'windspeed_versions': pd.Series({'original': df['windspeed'].skew(),
                                         'sqrt': np.sqrt(df['windspeed']).skew(),
                                         'log': np.log(df['windspeed']).skew()}),
    }


def associations(df):
    cols = CAT_FEATURES + NUMERIC_EXPLORE + [TARGET]
    pearson = df[cols].corr(method='pearson')
    spearman = df[cols].corr(method='spearman')
    assoc = pd.DataFrame({'pearson_r': pearson[TARGET],
                          'spearman_rho': spearman[TARGET]}).drop(TARGET)
    assoc = assoc.reindex(assoc['pearson_r'].abs().sort_values(ascending=False).index)
    return assoc, pearson


def comfort_temperature(df):
    """Experiment: which 'ideal' temperature makes |temp - ideal| most correlated with cnt?"""
    temps_c = df['temp'] * 41
    grid = np.arange(15, 36)
    curve = pd.Series([np.corrcoef((temps_c - t).abs(), df[TARGET])[0, 1] for t in grid], index=grid,
                      name='corr(|temp - T|, cnt)')
    return int(curve.idxmin()), curve


def attribute_experiments(df, comfort_c):
    """Try attribute combinations (Geron: 'rooms per household') and compare their correlation with cnt."""
    t_c = df['temp'] * 41
    candidates = pd.DataFrame({
        'temp (original)': df['temp'],
        'temp_squared': df['temp'] ** 2,
        f'temp_distance_from_{comfort_c}C': (t_c - comfort_c).abs(),
        'yr_x_temp': df['yr'] * df['temp'],
        'temp_x_hum': df['temp'] * df['hum'],
        'feels_minus_actual_C': df['atemp'] * 50 - t_c,
        'is_weekend': df['weekday'].isin([0, 6]).astype(int),
        'bad_weather (weathersit=3)': (df['weathersit'] == 3).astype(int),
        'hum (original)': df['hum'],
    })
    out = pd.DataFrame({'pearson_r with cnt': candidates.corrwith(df[TARGET]),
                        'spearman_rho with cnt': candidates.corrwith(df[TARGET], method='spearman')})
    return out.reindex(out['pearson_r with cnt'].abs().sort_values(ascending=False).index)


ATTRIBUTE_FINDINGS = [
    'yr_x_temp (temperature effect allowed to differ between 2011 and 2012) is more correlated with cnt than '
    'temp alone: warm days in 2012 are much busier than warm days in 2011 (system growth x weather).',
    'The distance from a comfort temperature is strongly NEGATIVELY correlated with cnt: demand drops both '
    'when it is too cold and when it is too hot - a non-linear effect a straight line in temp cannot capture.',
    'temp_squared, temp_x_hum and yr_x_temp are therefore offered to the model as optional extra attributes '
    '(BikeFeatureEngineer(add_extras=True)) and whether they help is decided by the grid search in Step 6.',
    'is_weekend and feels_minus_actual add little; bad_weather is a strong negative signal but is already '
    'captured by one-hot encoding weathersit.',
]


def step3_explore(data):
    """Explore ONLY the training set (avoids data-snooping bias on the test set)."""
    train = data['df'].loc[data['X_train'].index].copy()
    train['windspeed_sqrt'] = np.sqrt(train['windspeed'])
    assoc, pearson = associations(train)
    comfort_c, curve = comfort_temperature(train)
    return {'train_df': train, 'skew': skewness(train), 'assoc': assoc, 'pearson': pearson,
            'comfort_c': comfort_c, 'comfort_curve': curve,
            'experiments': attribute_experiments(train, comfort_c),
            'attribute_findings': ATTRIBUTE_FINDINGS,
            'events': event_analysis(data)}


# ---------------------------------------------------------------------------- STEP 3.4
def add_calendar_events(df):
    """Derive 0/1 flags for special periods from the date. They are known in advance, so using them
    for forecasting is legitimate (unlike storms, which are only known afterwards)."""
    d = df['dteday']
    md = d.dt.month * 100 + d.dt.day                       # e.g. 24 Dec -> 1224
    out = df.copy()
    out['christmas_newyear'] = ((md >= 1220) | (md <= 102)).astype(int)
    thanksgiving = pd.to_datetime(['2011-11-24', '2011-11-25', '2012-11-22', '2012-11-23'])
    out['thanksgiving'] = d.isin(thanksgiving).astype(int)
    out['july4'] = ((d.dt.month == 7) & (d.dt.day == 4)).astype(int)
    out['summer_break'] = ((md >= 620) & (md <= 825)).astype(int)
    hol = df['holiday'] == 1                                # rows are in date order
    next_to_holiday = hol.shift(1, fill_value=False) | hol.shift(-1, fill_value=False)
    out['holiday_bridge'] = (next_to_holiday & ~hol & (df['workingday'] == 1)).astype(int)
    return out


def event_analysis(data, random_state=RANDOM_STATE):
    """Do special calendar periods change demand beyond what season/weather/holiday already explain?
    (1) mean cnt on event days vs other days, (2) mean residual of the cross-validated linear model on
    event days (negative = model over-predicts them), (3) CV RMSE with and without the flags."""
    events = add_calendar_events(data['df'])
    train = events.loc[data['X_train'].index]
    y = train[TARGET]

    cv_pred = cross_val_predict(build_reg_pipeline(), data['X_train'], data['y_train'],
                                cv=cv_splitter(random_state))
    resid = y - cv_pred
    rows = []
    for flag in EVENT_FLAGS:
        on = train[flag] == 1
        rows.append({'event': flag,
                     'days in training set': int(on.sum()),
                     'mean cnt on event days': y[on].mean() if on.any() else np.nan,
                     'mean cnt on other days': y[~on].mean(),
                     'mean residual (actual - predicted)': resid[on].mean() if on.any() else np.nan})
    table = pd.DataFrame(rows).set_index('event').round(0)

    def cv_rmse(pipe, cols):
        res = cross_validate(pipe, train[cols], y, cv=cv_splitter(random_state),
                             scoring='neg_root_mean_squared_error')
        return -res['test_score'].mean(), res['test_score'].std()

    without = cv_rmse(build_reg_pipeline(), FEATURES)
    with_flags = cv_rmse(Pipeline([('engineer', BikeFeatureEngineer()),
                                   ('preprocess', build_preprocessor(extra_passthrough=EVENT_FLAGS)),
                                   ('model', LinearRegression())]), FEATURES + EVENT_FLAGS)
    comparison = pd.DataFrame({'CV RMSE mean': [without[0], with_flags[0]],
                               'CV RMSE std': [without[1], with_flags[1]]},
                              index=['Linear Regression (11 predictors)',
                                     'Linear Regression + 5 calendar-event flags']).round(1)
    return {'info': EVENT_INFO, 'table': table, 'comparison': comparison,
            'train_events': train, 'residuals': resid}


# ============================================================================ STEP 4
class BikeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Data cleaning + optional attribute combinations, usable inside a Pipeline.
    - hum <= 0 (impossible) is set to NaN, to be imputed later by SimpleImputer(median)
    - add_extras=True adds temp^2, temp*hum and yr*temp (found useful in Step 3)"""

    def __init__(self, add_extras=False):
        self.add_extras = add_extras

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['hum'] = X['hum'].where(X['hum'] > 0)
        if self.add_extras:
            X['extra_temp_squared'] = X['temp'] ** 2
            X['extra_temp_x_hum'] = X['temp'] * X['hum']
            X['extra_yr_x_temp'] = X['yr'] * X['temp']
        return X


def build_preprocessor(extra_passthrough=()):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')),
                        ('scale', StandardScaler())])
    wind = Pipeline([('sqrt', FunctionTransformer(np.sqrt, feature_names_out='one-to-one')),
                     ('scale', StandardScaler())])
    transformers = [
        ('num', numeric, make_column_selector(pattern=r'^(?:temp|atemp|hum|extra_.*)$')),
        ('wind', wind, WIND),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), NOMINAL),
        ('bin', 'passthrough', BINARY),
    ]
    if extra_passthrough:                       # e.g. the 0/1 calendar-event flags (Step 3.4)
        transformers.append(('events', 'passthrough', list(extra_passthrough)))
    return ColumnTransformer(transformers)


def build_prep_pipeline(add_extras=False):
    return Pipeline([('engineer', BikeFeatureEngineer(add_extras=add_extras)),
                     ('preprocess', build_preprocessor())])


def build_reg_pipeline(model=None, add_extras=False):
    return Pipeline([('engineer', BikeFeatureEngineer(add_extras=add_extras)),
                     ('preprocess', build_preprocessor()),
                     ('model', model if model is not None else LinearRegression())])


PREPARATION_DECISIONS = pd.DataFrame([
    ('Data cleaning', 'Yes',
     'Drop instant, dteday (identifiers) and casual, registered (leakage). Treat the impossible hum = 0 as '
     'missing and impute the training median.'),
    ('Missing values', 'Yes (1 hidden)', 'No empty cells; the single hum = 0 is imputed (SimpleImputer, median).'),
    ('Text / categorical attributes', 'Yes',
     'season, mnth, weekday, weathersit are nominal codes -> one-hot encoding (first level dropped). '
     'Ordinal codes would impose a fake order/spacing (e.g. month 12 far from month 1).'),
    ('Binary attributes', 'No transformation', 'yr, holiday, workingday are already 0/1 -> passed through.'),
    ('Transformation of skewed features', 'Yes (windspeed)',
     'windspeed is right-skewed (skew ~0.7); sqrt makes it ~symmetric (log over-corrects). temp, atemp, hum '
     'are ~symmetric -> no transformation.'),
    ('Target transformation', 'No', 'cnt is roughly symmetric (skew ~0) - a log would distort it.'),
    ('Attribute combinations', 'Optional (tuned)',
     'temp^2, temp*hum, yr*temp added by BikeFeatureEngineer(add_extras=True); kept only if the grid '
     'search finds they lower the CV error.'),
    ('Feature scaling', 'Yes',
     'StandardScaler on the numerical features. Required by the assignment; makes linear-regression '
     'coefficients comparable; essential for regularised models (Ridge) and gradient-based training. '
     '(Tree models would not need it, but it does not hurt them.) Min-max scaling is not chosen because '
     'standardisation is less affected by outliers.'),
    ('Where are the parameters learned?', 'Training set only',
     'Everything is inside one Pipeline: fit on the training set, then transform the test set with the '
     'same medians, means, standard deviations and categories -> no leakage.'),
], columns=['decision', 'needed?', 'why / how']).set_index('decision')


def step4_prepare(data):
    prep = build_prep_pipeline().fit(data['X_train'])              # fitted on TRAINING data only
    Xp = prep.transform(data['X_train'])
    ct = prep.named_steps['preprocess']
    names = ct.get_feature_names_out()
    num_pipe = ct.named_transformers_['num']
    wind_scaler = ct.named_transformers_['wind'].named_steps['scale']
    num_names = list(num_pipe.get_feature_names_out())
    params = pd.DataFrame({'train_mean': np.r_[num_pipe.named_steps['scale'].mean_, wind_scaler.mean_],
                           'train_std': np.r_[num_pipe.named_steps['scale'].scale_, wind_scaler.scale_]},
                          index=num_names + ['sqrt(windspeed)'])
    hum_median = num_pipe.named_steps['impute'].statistics_[num_names.index('hum')]
    return {'preprocessor': prep,
            'prepared_shape': Xp.shape,
            'feature_names': list(names),
            'prepared_head': pd.DataFrame(Xp[:5], columns=names),
            'scaler_params': params,
            'hum_missing_train': int((data['X_train']['hum'] <= 0).sum()),
            'hum_median': float(hum_median),
            'decisions': PREPARATION_DECISIONS}


# ============================================================================ STEP 5
class GroupMeanRegressor(BaseEstimator, RegressorMixin):
    """The 'current solution': predict the historical average of cnt for the same group of days.

    Why: Geron's checklist (Ch. 2, Step 1) asks for the current solution because it is the reference
    performance an ML system must beat. Writing the rule of thumb as an estimator lets us evaluate it
    with the same data, folds and metric as the ML models, so the comparison is fair."""

    def __init__(self, groups=('mnth', 'workingday')):
        self.groups = groups

    def fit(self, X, y):
        cols = list(self.groups)
        frame = X[cols].copy()
        frame['_y'] = np.asarray(y)
        self.means_ = frame.groupby(cols)['_y'].mean()
        self.global_mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        keys = pd.MultiIndex.from_frame(X[list(self.groups)])
        return self.means_.reindex(keys).fillna(self.global_mean_).to_numpy()


def cv_splitter(random_state=RANDOM_STATE):
    return KFold(n_splits=10, shuffle=True, random_state=random_state)


def step5_select_train(data, random_state=RANDOM_STATE):
    X_train, y_train = data['X_train'], data['y_train']

    # The assignment model (Q6): preprocessing + Linear Regression trained on the training set
    linear = build_reg_pipeline().fit(X_train, y_train)
    coefs = pd.Series(linear.named_steps['model'].coef_,
                      index=linear.named_steps['preprocess'].get_feature_names_out()).sort_values()

    # Shortlist of candidate models evaluated with 10-fold CV (Q8 for the linear model)
    candidates = {
        'Baseline (overall mean)': build_reg_pipeline(DummyRegressor(strategy='mean')),
        'Current solution (monthly average)': GroupMeanRegressor(),
        'Linear Regression': build_reg_pipeline(LinearRegression()),
        'Decision Tree': build_reg_pipeline(DecisionTreeRegressor(random_state=random_state)),
        'Random Forest': build_reg_pipeline(RandomForestRegressor(n_estimators=200, random_state=random_state,
                                                                  n_jobs=N_JOBS)),
    }
    rows, cv_r2 = [], {}
    for name, estimator in candidates.items():
        res = cross_validate(estimator, X_train, y_train, cv=cv_splitter(random_state),
                             scoring=['r2', 'neg_root_mean_squared_error'], return_train_score=True)
        cv_r2[name] = res['test_r2']
        rows.append({'model': name,
                     'train RMSE': -res['train_neg_root_mean_squared_error'].mean(),
                     'CV RMSE mean': -res['test_neg_root_mean_squared_error'].mean(),
                     'CV RMSE std': res['test_neg_root_mean_squared_error'].std(),
                     'CV R2 mean': res['test_r2'].mean(),
                     'CV R2 std': res['test_r2'].std()})
    comparison = pd.DataFrame(rows).set_index('model')

    return {'linear_model': linear,
            'intercept': linear.named_steps['model'].intercept_,
            'coefs': coefs,
            'train_r2': r2_score(y_train, linear.predict(X_train)),
            'comparison': comparison,
            'cv_r2': cv_r2}


# ============================================================================ STEP 6
def step6_fine_tune(data, random_state=RANDOM_STATE):
    """Grid search: with/without the extra attributes x {Linear, Ridge, Random Forest}."""
    param_grid = [
        {'engineer__add_extras': [False, True], 'model': [LinearRegression()]},
        {'engineer__add_extras': [False, True], 'model': [Ridge()], 'model__alpha': [0.1, 1, 10, 100]},
        {'engineer__add_extras': [False, True],
         'model': [RandomForestRegressor(random_state=random_state, n_jobs=N_JOBS)],
         'model__n_estimators': [100, 300], 'model__max_features': [0.33, 0.66, 1.0]},
    ]
    search = GridSearchCV(build_reg_pipeline(), param_grid, cv=cv_splitter(random_state),
                          scoring='neg_root_mean_squared_error', return_train_score=True, n_jobs=N_JOBS)
    search.fit(data['X_train'], data['y_train'])

    res = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({
        'model': res['param_model'].astype(str).str.replace(r'\(.*\)', '', regex=True),
        'params': res['params'].apply(lambda p: {k.split('__')[-1]: v for k, v in p.items() if k != 'model'}),
        'CV RMSE mean': -res['mean_test_score'],
        'CV RMSE std': res['std_test_score'],
        'train RMSE': -res['mean_train_score'],
        'rank': res['rank_test_score'],
    }).sort_values('rank')

    best = search.best_estimator_
    names = best.named_steps['preprocess'].get_feature_names_out()
    model = best.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=names, name='importance')
    else:
        importance = pd.Series(np.abs(model.coef_), index=names, name='|coefficient|')
    best_name = type(model).__name__
    return {'search': search,
            'best_model': best,
            'best_name': best_name,
            'best_params': {k: (type(v).__name__ if k == 'model' else v) for k, v in search.best_params_.items()},
            'best_cv_rmse': -search.best_score_,
            'results': table,
            'importance': importance.sort_values(ascending=False)}


# ============================================================================ STEP 7
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return pd.Series({'R2': r2_score(y_true, y_pred),
                      'MAE': mean_absolute_error(y_true, y_pred),
                      'MAPE (%)': 100 * mean_absolute_percentage_error(y_true, y_pred),
                      'MSE': mse,
                      'RMSE': np.sqrt(mse)})


def rmse_confidence_interval(y_true, y_pred, confidence=0.95):
    sq = (np.asarray(y_true) - np.asarray(y_pred)) ** 2
    lo, hi = stats.t.interval(confidence, len(sq) - 1, loc=sq.mean(), scale=stats.sem(sq))
    return np.sqrt(max(lo, 0)), np.sqrt(hi)


def step7_present(data, s5, s6):
    """Final evaluation on the test set (used only here)."""
    X_test, y_test = data['X_test'], data['y_test']
    y_lin = s5['linear_model'].predict(X_test)
    y_best = s6['best_model'].predict(X_test)
    lin_label, best_label = 'Linear Regression (assignment)', f"Fine-tuned {s6['best_name']}"

    metrics = pd.DataFrame({lin_label: regression_metrics(y_test, y_lin),
                            best_label: regression_metrics(y_test, y_best)})
    ci = pd.DataFrame({lin_label: rmse_confidence_interval(y_test, y_lin),
                       best_label: rmse_confidence_interval(y_test, y_best)},
                      index=['RMSE 95% CI low', 'RMSE 95% CI high'])

    errors = pd.DataFrame({'dteday': data['df'].loc[X_test.index, 'dteday'].dt.date,
                           'weathersit': X_test['weathersit'],
                           'actual': y_test,
                           'predicted (linear)': np.round(y_lin),
                           'residual (linear)': np.round(y_test - y_lin)})
    errors = errors.reindex(errors['residual (linear)'].abs().sort_values(ascending=False).index)

    cv_lin = s5['cv_r2']['Linear Regression']
    test_r2 = metrics.loc['R2', lin_label]
    return {'y_pred_linear': y_lin, 'y_pred_best': y_best,
            'labels': (lin_label, best_label),
            'metrics': metrics, 'rmse_ci': ci, 'errors': errors,
            'n_negative_pred': int((y_lin < 0).sum()),
            'cv_r2_linear': cv_lin, 'test_r2_linear': test_r2,
            'z_test_vs_cv': (test_r2 - cv_lin.mean()) / cv_lin.std()}


# ---------------------------------------------------------------------------- STEP 7.4
def time_based_validation(data, s5, s6, s7, test_fraction=0.2, n_splits=5):
    """Extra check (beyond the assignment): the random split of Q5 lets the model learn from days that
    come AFTER the test days (temporal leakage). Here we validate like a real forecast: train on the
    earliest 80% of days, test on the latest 20%, and cross-validate with TimeSeriesSplit (every fold
    trains on the past and tests on the following period)."""
    df = data['df'].sort_values('dteday')
    X, y = df[FEATURES], df[TARGET]
    cut = int(round(len(df) * (1 - test_fraction)))
    X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

    lin = build_reg_pipeline().fit(X_tr, y_tr)
    best = clone(s6['best_model']).fit(X_tr, y_tr)              # same settings as the fine-tuned model
    y_lin, y_best = lin.predict(X_te), best.predict(X_te)
    lin_label, best_label = s7['labels']

    comparison = pd.DataFrame({
        f'{lin_label} - random split': s7['metrics'][lin_label],
        f'{lin_label} - time split': regression_metrics(y_te, y_lin),
        f'{best_label} - random split': s7['metrics'][best_label],
        f'{best_label} - time split': regression_metrics(y_te, y_best),
    })

    tscv = TimeSeriesSplit(n_splits=n_splits)
    res = cross_validate(build_reg_pipeline(), X, y, cv=tscv,
                         scoring=['r2', 'neg_root_mean_squared_error'])
    folds = pd.DataFrame({
        'train days': [len(tr) for tr, _ in tscv.split(X)],
        'test period': [f"{df['dteday'].iloc[te[0]].date()} to {df['dteday'].iloc[te[-1]].date()}"
                        for _, te in tscv.split(X)],
        'R2': res['test_r2'],
        'RMSE': -res['test_neg_root_mean_squared_error'],
    }, index=[f'fold {i}' for i in range(1, n_splits + 1)])

    return {'train_period': (df['dteday'].iloc[0].date(), df['dteday'].iloc[cut - 1].date()),
            'test_period': (df['dteday'].iloc[cut].date(), df['dteday'].iloc[-1].date()),
            'comparison': comparison, 'tscv_folds': folds,
            'kfold_r2': (s5['cv_r2']['Linear Regression'].mean(), s5['cv_r2']['Linear Regression'].std()),
            'tscv_r2': (res['test_r2'].mean(), res['test_r2'].std()),
            'dates_test': df['dteday'].iloc[cut:], 'y_test': y_te, 'y_lin': y_lin, 'y_best': y_best,
            'dates_train': df['dteday'].iloc[:cut], 'y_train': y_tr,
            'labels': (lin_label, best_label)}


KEY_FINDINGS = [
    'casual and registered add up exactly to cnt -> excluded to avoid data leakage; instant and dteday are identifiers.',
    'Temperature (temp/atemp, r ~ 0.63), year (system growth) and season are the strongest drivers; '
    'bad weather and wind reduce demand; holiday/weekday/workingday barely affect total daily demand.',
    'Demand peaks at a comfortable temperature and falls when it is too hot - attribute combinations '
    '(temp^2, yr x temp) capture this non-linear, interacting effect.',
    'The linear model explains about 84% of the variance on unseen days and clearly beats the '
    '"monthly average" current solution; it over-predicts very low-demand days (storms, special events).',
    'Fine-tuning (extra attributes / Ridge / Random Forest) shows how much a more flexible model can reduce the error.',
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


def predict_demand(model, new_days):
    """Predict cnt for new days (DataFrame with the FEATURES columns). Negative values are clipped to 0."""
    return np.clip(model.predict(new_days[FEATURES]), 0, None)


def drift_report(reference, new, alpha=0.01, tvd_threshold=0.2):
    """Compare a new batch of inputs with the training data.
    Numeric features: two-sample Kolmogorov-Smirnov test. Categorical: total variation distance."""
    rows = []
    for col in NUM_RAW + WIND:
        stat, p = stats.ks_2samp(reference[col], new[col])
        rows.append({'feature': col, 'test': 'KS', 'statistic': stat, 'p_value': p, 'drift': p < alpha})
    for col in NOMINAL + BINARY:
        ref_p = reference[col].value_counts(normalize=True)
        new_p = new[col].value_counts(normalize=True)
        tvd = 0.5 * ref_p.subtract(new_p, fill_value=0).abs().sum()
        rows.append({'feature': col, 'test': 'TVD', 'statistic': tvd, 'p_value': np.nan, 'drift': tvd > tvd_threshold})
    return pd.DataFrame(rows).round(4)


def performance_check(model, X_new, y_new, reference_rmse, tolerance=1.25):
    """When the true counts become available, compare live RMSE with the validated RMSE."""
    rmse = np.sqrt(mean_squared_error(y_new, model.predict(X_new)))
    return {'live RMSE': rmse, 'reference RMSE': reference_rmse,
            'ratio': rmse / reference_rmse, 'alert': bool(rmse > tolerance * reference_rmse)}


MAINTENANCE_PLAN = [
    'Deploy the saved pipeline (cleaning + preprocessing + model in one joblib file) behind a small API or the Streamlit app.',
    'Every day: log inputs and predictions; check input drift (drift_report) against the training data.',
    'When the true daily counts arrive: track live RMSE (performance_check) and alert if it exceeds the validated RMSE by >25%.',
    'Retrain regularly (e.g. monthly) on a rolling window that includes the newest data - the yr effect shows that demand grows over time.',
    'Version data, code and models (Git + saved metadata) so any model can be rolled back; keep a human in the loop for anomalies (storms, events).',
]


def step8_launch(data, s6, s7, save=True, path=MODEL_PATH):
    final = s6['best_model']
    metadata = {'project': 'DAMA61 WA1 - Problem 1 (bike demand)',
                'model': s6['best_name'],
                'best_params': s6['best_params'],
                'trained_at': datetime.now().isoformat(timespec='seconds'),
                'n_train': int(len(data['y_train'])),
                'features': FEATURES,
                'cv_rmse': float(s6['best_cv_rmse']),
                'test_rmse': float(s7['metrics'].loc['RMSE', s7['labels'][1]]),
                'sklearn_version': sklearn.__version__}
    saved_to = save_model(final, metadata, path) if save else None

    # Monitoring demo: a "normal" batch (test days) and a simulated shifted batch (hotter, drier year)
    normal_batch = data['X_test']
    shifted_batch = data['X_test'].copy()
    shifted_batch['temp'] = (shifted_batch['temp'] + 0.15).clip(0, 1)
    shifted_batch['atemp'] = (shifted_batch['atemp'] + 0.12).clip(0, 1)
    shifted_batch['hum'] = shifted_batch['hum'] * 0.8
    return {'metadata': metadata, 'saved_to': saved_to,
            'drift_normal': drift_report(data['X_train'], normal_batch),
            'drift_shifted': drift_report(data['X_train'], shifted_batch),
            'performance': performance_check(final, data['X_test'], data['y_test'], s6['best_cv_rmse'])}


# ============================================================================ run everything
def run_all(save_model_file=True):
    s1 = step1_big_picture()
    s2 = step2_get_data()
    s3 = step3_explore(s2)
    s4 = step4_prepare(s2)
    s5 = step5_select_train(s2)
    s6 = step6_fine_tune(s2)
    s7 = step7_present(s2, s5, s6)
    s7['time'] = time_based_validation(s2, s5, s6, s7)
    s8 = step8_launch(s2, s6, s7, save=save_model_file)
    return {'s1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5, 's6': s6, 's7': s7, 's8': s8}


# ============================================================================ figures
def fig_all_histograms(df):
    """Step 2: a histogram of every numerical attribute (Geron: housing.hist(bins=50))."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    for ax, col in zip(axes.ravel(), NUMERICAL_COLUMNS):
        ax.hist(df[col], bins=50, color='tab:blue', edgecolor='k')
        ax.axvline(df[col].mean(), color='red', ls='--', label=f'mean {df[col].mean():.2f}')
        ax.axvline(df[col].median(), color='black', ls=':', label=f'median {df[col].median():.2f}')
        ax.set(title=f'{col} (skew = {df[col].skew():.2f})', xlabel=col, ylabel='days')
        ax.legend(fontsize=9)
    for ax in axes.ravel()[len(NUMERICAL_COLUMNS):]:
        ax.axis('off')
    fig.suptitle('Histograms of the numerical attributes (full dataset, before the split)')
    fig.tight_layout()
    return fig


def fig_q2_counts(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].scatter(df['casual'], df['cnt'], s=10, alpha=0.6)
    axes[0].set(xlabel='casual', ylabel='cnt', title='casual vs cnt')
    axes[1].scatter(df['registered'], df['cnt'], s=10, alpha=0.6, color='tab:green')
    axes[1].set(xlabel='registered', ylabel='cnt', title='registered vs cnt')
    for col in ['cnt', 'registered', 'casual']:
        axes[2].plot(df['dteday'], df[col], label=col, lw=1)
    axes[2].set(xlabel='date', ylabel='rentals', title='Rentals over time')
    axes[2].legend()
    fig.tight_layout()
    return fig


def fig_q5_hist(y, y_train, y_test):
    bins = np.histogram_bin_edges(y, bins=20)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(y_train, bins=bins, density=True, alpha=0.5, edgecolor='k', label=f'train (n={len(y_train)})')
    ax.hist(y_test, bins=bins, density=True, alpha=0.5, edgecolor='k', label=f'test (n={len(y_test)})')
    ax.set(xlabel='cnt', ylabel='density', title='Normalised distribution of cnt: train vs test')
    ax.legend()
    fig.tight_layout()
    return fig


def fig_q3_grid(df):
    cols = CAT_FEATURES + NUM_FEATURES + [TARGET]   # 12 variables -> 4 x 3
    fig, axes = plt.subplots(4, 3, figsize=(18, 18))
    for ax, col in zip(axes.ravel(), cols):
        if col in CAT_FEATURES:
            counts = df[col].value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, color='tab:orange', edgecolor='k')
            ax.set_ylabel('number of days')
        else:
            ax.hist(df[col], bins=30, color='tab:red' if col == TARGET else 'tab:blue', edgecolor='k')
            ax.set_ylabel('frequency')
        ax.set_title(f'{col} (skew = {df[col].skew():.2f})')
        ax.set_xlabel(col)
    fig.suptitle('Distributions of the predictors and of the target cnt (training set)')
    fig.tight_layout()
    return fig


def fig_q3_wind(df):
    versions = {'windspeed (original)': df['windspeed'],
                'sqrt(windspeed)': np.sqrt(df['windspeed']),
                'log(windspeed)': np.log(df['windspeed'])}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for j, (name, values) in enumerate(versions.items()):
        axes[0, j].hist(values, bins=30, edgecolor='k')
        axes[0, j].set_title(f'{name}: skew = {values.skew():.2f}')
        r = np.corrcoef(values, df[TARGET])[0, 1]
        axes[1, j].scatter(values, df[TARGET], s=10, alpha=0.5)
        axes[1, j].set(xlabel=name, ylabel='cnt', title=f'{name} vs cnt (r = {r:.3f})')
    fig.tight_layout()
    return fig


def fig_q4_heatmap(pearson):
    cols = list(pearson.columns)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(pearson, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{pearson.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im)
    ax.set_title('Pearson correlation matrix (training set)')
    fig.tight_layout()
    return fig


def fig_scatter_matrix(train):
    axes = scatter_matrix(train[['cnt', 'temp', 'atemp', 'hum', 'windspeed_sqrt']], figsize=(14, 14),
                          alpha=0.4, diagonal='hist')
    fig = axes[0, 0].get_figure()
    fig.suptitle('Scatter matrix of the most promising attributes (training set)')
    return fig


def fig_q4_scatter(df):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, col in zip(axes, NUMERIC_EXPLORE):
        ax.scatter(df[col], df[TARGET], s=10, alpha=0.5)
        slope, intercept = np.polyfit(df[col], df[TARGET], 1)
        xs = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(xs, slope * xs + intercept, color='red')
        r = np.corrcoef(df[col], df[TARGET])[0, 1]
        ax.set(xlabel=col, ylabel='cnt', title=f'{col} vs cnt (r = {r:.2f})')
    fig.tight_layout()
    return fig


def fig_q4_box(df):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for ax, col in zip(axes.ravel(), CAT_FEATURES):
        levels = sorted(df[col].unique())
        ax.boxplot([df.loc[df[col] == v, TARGET] for v in levels])
        ax.set_xticks(range(1, len(levels) + 1))
        ax.set_xticklabels(levels)
        r = np.corrcoef(df[col], df[TARGET])[0, 1]
        ax.set(xlabel=col, ylabel='cnt', title=f'cnt by {col} (r = {r:.2f})')
    axes.ravel()[-1].axis('off')
    fig.tight_layout()
    return fig


def fig_attribute_experiments(train, experiments, comfort_c, curve):
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    exp = experiments['pearson_r with cnt'][::-1]
    axes[0].barh(exp.index, exp.values, color=np.where(exp.values > 0, 'tab:blue', 'tab:red'), edgecolor='k')
    axes[0].axvline(0, color='k', lw=0.8)
    axes[0].set(xlabel='Pearson r with cnt', title='Attribute combinations vs originals')
    axes[1].plot(curve.index, curve.values, marker='o')
    axes[1].axvline(comfort_c, color='red', ls='--', label=f'best = {comfort_c} °C')
    axes[1].set(xlabel='assumed comfort temperature T (°C)', ylabel='corr(|temp - T|, cnt)',
                title='Searching the "comfort" temperature')
    axes[1].legend()
    for yr, colour in [(0, 'tab:gray'), (1, 'tab:green')]:
        part = train[train['yr'] == yr]
        axes[2].scatter(part['temp'] * 41, part[TARGET], s=10, alpha=0.6, color=colour, label=str(2011 + yr))
    axes[2].set(xlabel='temperature (°C)', ylabel='cnt', title='Temperature effect differs by year (yr x temp)')
    axes[2].legend()
    fig.tight_layout()
    return fig


def fig_events(ev):
    """Step 3.4: demand on special calendar periods and how far the linear model misses them."""
    train, table = ev['train_events'], ev['table']
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    groups = [train.loc[train[f] == 1, TARGET] for f in EVENT_FLAGS]
    normal = train.loc[train[EVENT_FLAGS].sum(axis=1) == 0, TARGET]
    axes[0].boxplot([g if len(g) else pd.Series([np.nan]) for g in groups] + [normal])
    axes[0].set_xticks(range(1, len(EVENT_FLAGS) + 2))
    axes[0].set_xticklabels([f'{f}\n(n={len(g)})' for f, g in zip(EVENT_FLAGS, groups)]
                            + [f'no event\n(n={len(normal)})'], fontsize=9)
    axes[0].set(ylabel='cnt', title='Daily rentals on special calendar periods (training set)')
    res = table['mean residual (actual - predicted)']
    axes[1].barh(res.index, res.values, color=np.where(res.values < 0, 'tab:red', 'tab:blue'), edgecolor='k')
    axes[1].axvline(0, color='k', lw=0.8)
    axes[1].set(xlabel='mean residual of the cross-validated linear model (bikes/day)',
                title='Does the model already explain these periods? (< 0 = over-predicted)')
    fig.tight_layout()
    return fig


def fig_model_comparison(comparison):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(comparison.index, comparison['CV RMSE mean'], xerr=comparison['CV RMSE std'],
            color='tab:blue', edgecolor='k', capsize=4, label='10-fold CV RMSE')
    ax.scatter(comparison['train RMSE'], comparison.index, color='red', zorder=3, label='training RMSE')
    ax.set(xlabel='RMSE (bikes/day)', title='Candidate models - training vs cross-validated error')
    ax.legend()
    fig.tight_layout()
    return fig


def fig_importance(importance, top=15):
    imp = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp.index, imp.values, color='tab:green', edgecolor='k')
    ax.set(xlabel=importance.name, title=f'Top {top} features of the fine-tuned model')
    fig.tight_layout()
    return fig


def fig_actual_vs_predicted(y_test, y_pred, title='Linear Regression'):
    r2 = r2_score(y_test, y_pred)
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', lw=2, label='perfect prediction (y = x)')
    axes[0].set(xlabel='actual cnt', ylabel='predicted cnt', title=f'{title}: Actual vs Predicted (test, $R^2$ = {r2:.3f})')
    axes[0].legend()
    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolor='k', color='tab:orange')
    axes[1].axhline(0, color='r', ls='--')
    axes[1].set(xlabel='predicted cnt', ylabel='residual (actual - predicted)', title=f'{title}: residuals (test)')
    fig.tight_layout()
    return fig


def fig_time_validation(tv):
    """Step 7.4: forecast the last 20% of days from the first 80%, and random vs time-based errors."""
    lin_label, best_label = tv['labels']
    fig, axes = plt.subplots(1, 2, figsize=(22, 6), gridspec_kw={'width_ratios': [2.2, 1]})
    ax = axes[0]
    ax.plot(tv['dates_train'], tv['y_train'], color='grey', lw=0.8, label='training period (actual)')
    ax.plot(tv['dates_test'], tv['y_test'], color='black', lw=1.2, label='test period (actual)')
    ax.plot(tv['dates_test'], tv['y_lin'], color='tab:orange', lw=1.2, label=f'{lin_label} forecast')
    ax.plot(tv['dates_test'], tv['y_best'], color='tab:green', lw=1.2, label=f'{best_label} forecast')
    ax.axvline(tv['dates_test'].iloc[0], color='red', ls='--', label='train / test cut')
    ax.set(xlabel='date', ylabel='cnt', title='Time-based split: trained on the past, forecasting the last 20% of days')
    ax.legend(fontsize=9, loc='upper left')
    rmse = tv['comparison'].loc['RMSE']
    names = ['Linear\nrandom split', 'Linear\ntime split', 'Fine-tuned\nrandom split', 'Fine-tuned\ntime split']
    bars = axes[1].bar(names, rmse.values, color=['tab:orange', 'tab:orange', 'tab:green', 'tab:green'],
                       edgecolor='k', alpha=0.85)
    for bar, h in zip(bars, ['', '//', '', '//']):
        bar.set_hatch(h)
    for i, v in enumerate(rmse.values):
        axes[1].text(i, v + 15, f'{v:.0f}', ha='center')
    axes[1].set(ylabel='test RMSE (bikes/day)', title='Random split vs time-based split')
    fig.tight_layout()
    return fig


def fig_q8(cv_r2, test_r2):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, len(cv_r2) + 1), cv_r2, color='tab:blue', edgecolor='k', label='CV fold $R^2$')
    ax.axhline(cv_r2.mean(), color='k', ls='--', label=f'CV mean = {cv_r2.mean():.3f}')
    ax.axhspan(cv_r2.mean() - cv_r2.std(), cv_r2.mean() + cv_r2.std(), color='grey', alpha=0.2, label='mean ± 1 std')
    ax.axhline(test_r2, color='r', lw=2, label=f'test $R^2$ = {test_r2:.3f}')
    ax.set_xticks(range(1, len(cv_r2) + 1))
    ax.set(xlabel='fold', ylabel='$R^2$', title='Linear Regression: 10-fold CV vs held-out test set',
           ylim=(min(cv_r2.min(), test_r2) - 0.1, 1))
    ax.legend(loc='lower right')
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
    parser = argparse.ArgumentParser(description='DAMA61 WA1 - Problem 1 (end-to-end ML project)')
    parser.add_argument('--no-show', action='store_true', help='only save the figures, do not display them')
    parser.add_argument('--outdir', default='figures', help='folder for the saved figures')
    args, _ = parser.parse_known_args()          # tolerate extra args passed by Spyder / IPython
    mode = setup_display()
    if mode == 'terminal' and args.no_show:
        matplotlib.use('Agg')
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    pd.set_option('display.width', 180)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_colwidth', 110)

    def save(fig, name, split=True):
        """Save the combined figure + each subplot separately; in Spyder show the combined
        figure first and then every individual plot one by one."""
        fig.savefig(outdir / f'p1_{name}.png', dpi=120, bbox_inches='tight')
        parts = save_individual_axes(fig, f'p1_{name}', outdir / 'individual') if split else []
        show_figure(fig, mode, keep_open=not args.no_show)
        show_images(parts, mode)

    def header(text):
        print('\n' + '=' * 100 + '\n' + text + '\n' + '=' * 100)

    # Each step is computed right before it is printed, so its figures appear immediately.
    s1 = step1_big_picture()
    header('STEP 1 - Look at the big picture  [Q1]')
    for k, v in s1.items():
        print(f'\n* {k}\n  {v}')

    s2 = step2_get_data()
    header('STEP 2 - Get the data  [Q1, Q2, Q5]')
    ins = s2['inspection']
    print('Data source:', s2['source'])
    print('Shape:', ins['shape'])
    print(ins['head'])
    print(ins['info'])
    print('Meaning of each column:\n', s2['column_info'])
    print('\nTypes, ranges and completeness:\n', ins['summary'])
    print(ins['describe'].round(3))
    for col, vc in ins['value_counts'].items():
        print(f'{col}: {vc.to_dict()}')
    print('\nEmpty cells:', ins['missing_total'], '| duplicated rows:', ins['duplicates'])
    print('Implausible values (hum = 0):\n', ins['implausible'])
    print('Missing-data plan:')
    for p in s2['missing_plan']:
        print(' -', p)
    print('\ninstant == 1..n:', ins['instant_is_index'],
          '| casual + registered == cnt for every row:', ins['sum_check'])
    print(ins['corr_counts'].round(3))
    print('Excluded from predictors (identifiers / leakage):', EXCLUDED)
    print('Summary of the data:')
    for p in s2['data_summary']:
        print(' -', p)
    print('\nTest set: train', s2['X_train'].shape, ' test', s2['X_test'].shape)
    print(s2['target_summary'].round(1))
    print(f"KS test train vs test cnt: statistic = {s2['ks'].statistic:.3f}, p-value = {s2['ks'].pvalue:.3f}")
    save(fig_all_histograms(s2['df']), 'step2_histograms_all_numerical')
    save(fig_q2_counts(s2['df']), 'step2_casual_registered')
    save(fig_q5_hist(s2['y'], s2['y_train'], s2['y_test']), 'step2_train_test_hist')

    s3 = step3_explore(s2)
    header('STEP 3 - Discover and visualise the data (training set)  [Q3, Q4]')
    sk = s3['skew']
    print('Numerical skewness:\n', sk['numerical'].round(3))
    print('Categorical skewness (reference only):\n', sk['categorical'].round(3))
    print('windspeed skew original / sqrt / log:\n', sk['windspeed_versions'].round(3))
    print('Correlations with cnt:\n', s3['assoc'].round(3))
    print(f"\nComfort temperature found on the training set: {s3['comfort_c']} °C")
    print('Attribute combinations:\n', s3['experiments'].round(3))
    for f in s3['attribute_findings']:
        print(' -', f)
    save(fig_q3_grid(s3['train_df']), 'step3_distributions')
    save(fig_q3_wind(s3['train_df']), 'step3_windspeed_transform')
    save(fig_q4_heatmap(s3['pearson']), 'step3_heatmap')
    save(fig_scatter_matrix(s3['train_df']), 'step3_scatter_matrix', split=False)   # 25 tiny panels: keep whole
    save(fig_q4_scatter(s3['train_df']), 'step3_scatter')
    save(fig_q4_box(s3['train_df']), 'step3_boxplots')
    save(fig_attribute_experiments(s3['train_df'], s3['experiments'], s3['comfort_c'], s3['comfort_curve']),
         'step3_attribute_combinations')

    header('STEP 3.4 - Extra calendar information: Christmas, Thanksgiving, July 4, summer break, bridge days')
    ev = s3['events']
    print(ev['info'])
    print('\nDemand and model error on special periods (training set):\n', ev['table'])
    print('\nDo the flags improve the linear model? (10-fold CV on the training set)\n', ev['comparison'])
    save(fig_events(ev), 'step3_calendar_events')

    s4 = step4_prepare(s2)
    header('STEP 4 - Prepare the data for ML algorithms  [Q3, Q6]')
    print(s4['decisions'])
    print(f"\nhum = 0 values in the training set: {s4['hum_missing_train']} -> imputed with median {s4['hum_median']:.4f}")
    print(s4['preprocessor'])
    print('Prepared training matrix shape:', s4['prepared_shape'])
    print('Scaler parameters learned on the training set:\n', s4['scaler_params'].round(4))
    print(s4['prepared_head'].round(3))

    print('\nTraining and cross-validating the candidate models...')
    s5 = step5_select_train(s2)
    header('STEP 5 - Select a model and train it  [Q6, Q8]')
    print(s5['linear_model'])
    print('Linear Regression intercept:', round(s5['intercept'], 1), '| training R2:', round(s5['train_r2'], 3))
    print(s5['coefs'].round(1))
    print('\nCandidate models (10-fold CV on the training set):')
    print(s5['comparison'].round(3))
    save(fig_model_comparison(s5['comparison']), 'step5_model_comparison')

    print('\nGrid search running (about 1-2 minutes)...')
    s6 = step6_fine_tune(s2)
    header('STEP 6 - Fine-tune the model')
    print('Best:', s6['best_params'], '| CV RMSE =', round(s6['best_cv_rmse'], 1))
    print(s6['results'].head(12).round(2).to_string(index=False))
    print('\nMost important features:\n', s6['importance'].head(10).round(3))
    save(fig_importance(s6['importance']), 'step6_feature_importance')

    s7 = step7_present(s2, s5, s6)
    header('STEP 7 - Present the solution (test set)  [Q7, Q8]')
    print(s7['metrics'].round(3))
    print(s7['rmse_ci'].round(1))
    print('Largest errors of the linear model:\n', s7['errors'].head(8))
    print('Negative predictions (linear):', s7['n_negative_pred'])
    cv = s7['cv_r2_linear']
    print(f"Linear: CV R2 = {cv.mean():.3f} ± {cv.std():.3f} | test R2 = {s7['test_r2_linear']:.3f} "
          f"({s7['z_test_vs_cv']:+.2f} std from the CV mean)")
    print('Key findings:')
    for f in KEY_FINDINGS:
        print(' -', f)
    lin_label, best_label = s7['labels']
    save(fig_actual_vs_predicted(s2['y_test'], s7['y_pred_linear'], lin_label), 'step7_linear_actual_vs_predicted')
    save(fig_actual_vs_predicted(s2['y_test'], s7['y_pred_best'], best_label), 'step7_best_actual_vs_predicted')
    save(fig_q8(cv, s7['test_r2_linear']), 'step7_cv_vs_test')

    header('STEP 7.4 - Time-based validation (extra: does the model forecast the future?)')
    s7['time'] = time_based_validation(s2, s5, s6, s7)
    tv = s7['time']
    print('Train period:', tv['train_period'][0], 'to', tv['train_period'][1],
          '| test period:', tv['test_period'][0], 'to', tv['test_period'][1])
    print(tv['comparison'].round(3))
    print('\nTimeSeriesSplit cross-validation (linear regression, always train on the past):\n',
          tv['tscv_folds'].round(3))
    print(f"Linear R2: shuffled 10-fold CV = {tv['kfold_r2'][0]:.3f} ± {tv['kfold_r2'][1]:.3f} | "
          f"TimeSeriesSplit CV = {tv['tscv_r2'][0]:.3f} ± {tv['tscv_r2'][1]:.3f}")
    save(fig_time_validation(tv), 'step7_time_based_validation')

    s8 = step8_launch(s2, s6, s7, save=True)
    header('STEP 8 - Launch, monitor and maintain')
    print('Model saved to:', s8['saved_to'])
    print(json.dumps(s8['metadata'], indent=2, default=str))
    print('Drift check - normal batch:\n', s8['drift_normal'])
    print('Drift check - simulated hotter/drier batch:\n', s8['drift_shifted'])
    print('Performance check:', s8['performance'])
    print('Maintenance plan:')
    for p in MAINTENANCE_PLAN:
        print(' -', p)

    print(f'\nFigures saved to: {outdir.resolve()}')
    if mode == 'terminal' and not args.no_show:
        plt.show()                                # terminal: open all figure windows at the end


if __name__ == '__main__':
    # Run through the importable module so that saved models reference problem1.BikeFeatureEngineer
    # (not __main__.BikeFeatureEngineer) and can be loaded later by app.py or other scripts.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import problem1
    problem1.main()
