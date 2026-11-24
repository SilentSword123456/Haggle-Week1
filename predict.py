import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split
import copy, numpy as np
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

df = pd.read_csv('train.csv').drop(['project_id', 'hours_spent', 'hours_bucket'], axis=1)

hp = copy.deepcopy(get_hyperparameter_config("commercial_2026_08_05"))
for model_name, configs in hp.items():
    for cfg in configs:
        cfg.pop("ag.min_rows", None)

train, hold = train_test_split(df, test_size=0.2, random_state=42)

#predictor = TabularPredictor(label="log_hours", eval_metric="rmse", path="models/ag").fit(train, presets="high_v150", hyperparameters=hp, time_limit=1200)

predicator = TabularPredictor.load("models/ag")
test = pd.read_csv("test.csv")
log_pred = predicator.predict(test)
hours = np.expm1(log_pred).clip(0.1)

out = test[["project_id"]].copy() if "project_id" in test.columns else pd.DataFrame(index=test.index)
out["hours_spent"] = hours
out.to_csv("predictions.csv", index=False)