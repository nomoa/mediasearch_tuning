import json
from os.path import basename

import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import matplotlib.pyplot as plot

from utils import FEATURE_NAMES, load

features_version = "MediaSearch_20210826"
filename = "%s.tsv" % features_version
dataset = load(filename=filename)
version = 2
trees = 34
depth = 4
model_name = "%s_xgboost_v%s_%dt_%dd" % (features_version, version, trees, depth)

feats, labels = dataset[FEATURE_NAMES[filename]], dataset['label']

data_train, data_test, label_train, label_test = train_test_split(feats, labels, test_size=0.2, random_state=123)
params = {
    'booster': 'gbtree',
    'eta': 0.1,
    'eval_metric': 'logloss',
    'max_depth': depth,
    'num_boost_round': trees,
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

with open(model_name + ".json", 'w') as fmodel:
    splits = bst.get_dump(dump_format="json")
    model = {
        "model": {
            "name": model_name,
            "model": {
                "type": "model/xgboost+json",
                "definition": '[' + ','.join(splits) + ']'
            }
        }
    }

    json.dump(model, fmodel)
