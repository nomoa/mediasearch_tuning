import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

from utils import load, plot_grid_search, FEATURE_NAMES


filename = "MediaSearch_20210826.tsv"
model_name = "MediaSearch_20210826"
dataset = load(filename=filename)
feats, labels = dataset[FEATURE_NAMES[filename]], dataset['label']

default_params = {
    'eta': 0.1,  # 0.1 is always better don't search for better params
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'booster': 'gbtree'
}
param_grid = {
    'max_depth': [2, 3, 4, 5, 6],
    'n_estimators': np.linspace(start=1, stop=100, num=10, dtype=int)
}
model = XGBClassifier(objective=default_params['objective'], use_label_encoder=False)
model.set_params(**default_params)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=6)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X=feats, y=labels)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Calling Method
plot_grid_search(grid_result.cv_results_, param_grid['n_estimators'], param_grid['max_depth'],
                 'n_estimators', 'max_depth', 'nb trees vs max depth')
plt.show()

