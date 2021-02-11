import json

import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import matplotlib.pyplot as plot

from load import FEATURE_NAMES, load

dataset = load()

feats, labels = dataset[FEATURE_NAMES], dataset['label']

data_train, data_test, label_train, label_test = train_test_split(feats, labels, test_size=0.2, random_state=123)
params = {
    'booster': 'gbtree',
    'eta': 0.1,
    'eval_metric': 'logloss',
    'max_depth': 4,
    'num_boost_round': 20,
    'objective': 'binary:logistic'
}

num_boost_round = params['num_boost_round']

matrix_all = xgb.DMatrix(data=feats, label=labels)

matrix_train = xgb.DMatrix(data=data_train, label=label_train)
matrix_test = xgb.DMatrix(data=data_test, label=label_test)
watch_list = [(matrix_test, 'eval'), (matrix_train, 'train')]

bst = xgb.train(params=params, dtrain=matrix_train, num_boost_round=num_boost_round, evals=watch_list)
fig, axs = plot.subplots(2)
plot_importance(bst, importance_type='weight', title="", ylabel="", xlabel="weight", ax=axs[0], grid=True, show_values=False, height=0.5)
plot_importance(bst, importance_type='gain', title="", ylabel="", xlabel="gain", ax=axs[1], show_values=False, height=0.5)
fig.tight_layout()
plot.show()

with open("./MediaSearch_20210127_xgboost_v1_20t_4d.json", 'w') as fmodel:
    splits = bst.get_dump(dump_format="json")
    model = {
        "model": {
            "name": "MediaSearch_20210127_xgboost_v1_20t_4d",
            "model": {
                "type": "model/xgboost+json",
                "definition": '[' + ','.join(splits) + ']'
            }
        }
    }

    json.dump(model, fmodel)
