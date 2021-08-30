import matplotlib.pyplot as plt
import numpy as np
from decorator import __init__
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, GridSearchCV, GroupKFold
from xgboost import XGBClassifier, XGBRanker

from utils import load, plot_grid_search, FEATURE_NAMES

# Total mess
#


class XGBRankerEstimator(BaseEstimator):
    def __init__(self, eta, objective, eval_metric, booster, max_depth, n_estimators):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.booster = booster
        self.eval_metric = eval_metric
        self.objective = objective
        self.eta = eta

    def fit(self, X, y, *, sample_weight=None, base_margin=None,
            eval_set=None, sample_weight_eval_set=None,
            eval_group=None, eval_metric=None,
            early_stopping_rounds=None, verbose=False, xgb_model=None,
            feature_weights=None, callbacks=None):
        feats = X.drop(['query_group'], axis=1)
        group_counts = X.groupby(['query_group']).size()

        params = self.get_params()
        self._ranker = XGBRanker(objective=params['objective'])
        self._ranker.set_params(**self.get_params())
        self._ranker.fit(feats, y, group_counts, sample_weight=sample_weight, base_margin=base_margin,
                         eval_set=eval_set,
                         sample_weight_eval_set=sample_weight_eval_set,
                         eval_group=eval_group,
                         eval_metric=eval_metric,
                         early_stopping_rounds=early_stopping_rounds,
                         verbose=verbose,
                         xgb_model=xgb_model,
                         feature_weights=feature_weights,
                         callbacks=callbacks)
        return self

    def predict(self, x):
        feats = x.drop(['query_group'], axis=1)
        self._ranker.predict(feats)

    def score(self, x, y):
        self._ranker.evals_result()



dataset = load()
group_and_feats, labels, groups = dataset[['query_group'] + FEATURE_NAMES], dataset['label'], dataset['query_group']
param_grid = {
    'eta': [0.1],  # 0.1 is always better don't search for better params
    'objective': ['rank:pairwise'],
    'eval_metric': ['map'],
    'booster': ['gbtree'],
    'max_depth': [2, 3, 4, 5, 6],
    'n_estimators': np.linspace(start=1, stop=100, num=10, dtype=int)
}
estimator = XGBRankerEstimator(**param_grid)

kfold = GroupKFold(n_splits=5)
folds = list(kfold.split(group_and_feats, labels, groups=groups))
grid_search = GridSearchCV(estimator, param_grid, scoring="neg_log_loss", n_jobs=1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X=group_and_feats, y=labels, groups=groups)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Calling Method
plot_grid_search(grid_result.cv_results_, param_grid['n_estimators'], param_grid['max_depth'],
                 'n_estimators', 'max_depth', 'nb trees vs max depth')
plt.show()



