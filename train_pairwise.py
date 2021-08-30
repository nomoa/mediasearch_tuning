import json

import matplotlib.pyplot as plot
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from xgboost import plot_importance

from utils import FEATURE_NAMES, load

dataset = load(pairwise=True)

feats, labels, groups = dataset[FEATURE_NAMES + ['query_group']], dataset['label'], dataset['query_group']

train_idx, test_idx = next(GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=123).split(feats, groups=groups))
data_train = feats.iloc[train_idx]
label_train = labels.iloc[train_idx]

data_test = feats.iloc[test_idx]
label_test = labels.iloc[test_idx]

groups_train = data_train.groupby(['query_group']).size()
groups_test = data_test.groupby(['query_group']).size()

data_train = data_train.drop(['query_group'], axis=1)
data_test = data_test.drop(['query_group'], axis=1)

params = {
    'booster': 'gbtree',
    'eta': 0.1,
    'eval_metric': 'map',
    'max_depth': 4,
    'num_boost_round': 20,
    'objective': 'rank:map'
}

num_boost_round = params['num_boost_round']

matrix_train = xgb.DMatrix(data=data_train, label=label_train)
matrix_train.set_group(groups_train)
matrix_test = xgb.DMatrix(data=data_test, label=label_test)
matrix_test.set_group(groups_test)
watch_list = [(matrix_test, 'eval'), (matrix_train, 'train')]

bst = xgb.train(params=params, dtrain=matrix_train, num_boost_round=num_boost_round, evals=watch_list)
fig, axs = plot.subplots(2)
plot_importance(bst, importance_type='weight', title="", ylabel="", xlabel="weight", ax=axs[0], grid=True, show_values=False, height=0.5)
plot_importance(bst, importance_type='gain', title="", ylabel="", xlabel="gain", ax=axs[1], show_values=False, height=0.5)
fig.tight_layout()
plot.show()

with open("./MediaSearch_20210127_xgboost_map_v1_20t_4d.json", 'w') as fmodel:
    splits = bst.get_dump(dump_format="json")
    model = {
        "model": {
            "name": "MediaSearch_20210127_xgboost_map_v1_20t_4d",
            "model": {
                "type": "model/xgboost+json",
                "definition": '[' + ','.join(splits) + ']'
            }
        }
    }

    json.dump(model, fmodel)
