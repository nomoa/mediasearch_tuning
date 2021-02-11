import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

from load import load, FEATURE_NAMES


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2, title):
    # Get Test Scores Mean and std for each grid search

    grid_param_1 = list(str(e) for e in grid_param_1)
    grid_param_2 = list(str(e) for e in grid_param_2)
    scores_mean = cv_results['mean_test_score']
    scores_std = cv_results['std_test_score']
    params_set = cv_results['params']

    scores_organized = {}
    std_organized = {}
    std_upper = {}
    std_lower = {}
    for p2 in grid_param_2:
        scores_organized[p2] = []
        std_organized[p2] = []
        std_upper[p2] = []
        std_lower[p2] = []
        for p1 in grid_param_1:
            for i in range(len(params_set)):
                if str(params_set[i][name_param_1]) == str(p1) and str(params_set[i][name_param_2]) == str(p2):
                    mean = scores_mean[i]
                    std = scores_std[i]
                    scores_organized[p2].append(mean)
                    std_organized[p2].append(std)
                    std_upper[p2].append(mean + std)
                    std_lower[p2].append(mean - std)

    _, ax = plt.subplots(1, 1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    # plot means
    for key in scores_organized.keys():
        ax.plot(grid_param_1, scores_organized[key], '-o', label= name_param_2 + ': ' + str(key))
        ax.fill_between(grid_param_1, std_lower[key], std_upper[key], alpha=0.1)

    ax.set_title(title)
    ax.set_xlabel(name_param_1)
    ax.set_ylabel('CV Average Score')
    ax.legend(loc="best")
    ax.grid('on')


dataset = load()
feats, labels = dataset[FEATURE_NAMES], dataset['label']

num_round = 100

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

