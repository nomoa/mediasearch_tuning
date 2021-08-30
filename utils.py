import matplotlib.pyplot as plt
import pandas as pd

FEATURE_NAMES = {
    'MediaSearch_20210127.tsv': [
        "match_descriptions_plain",
        "match_descriptions",
        "match_title",
        "match_title_plain",
        "match_category",
        "match_category_plain",
        "match_redirect_title",
        "match_redirect_title_plain",
        "match_suggest",
        "match_suggest_plain",
        "match_aux_text",
        "match_aux_text_plain",
        "match_text",
        "match_text_plain",
        "match_statements",
    ],
    'MediaSearch_20210826.tsv': [
        "match_descriptions_plain",
        "match_title_plain",
        "match_category",
        "match_redirect_title_plain",
        "match_suggest",
        "match_aux_text_plain",
        "match_text_plain",
        "match_statements",
    ]
}


def pairwise_labels(x):
    return (x + 1) * 2

def pointwise_labels(x):
    if x == -1:
        return 0
    else:
        return x


def load(filename="MediaSearch_20210127.tsv", pairwise=False):

    feat_names = FEATURE_NAMES[filename]
    with open(filename) as f:
        headers = ['label', 'qid'] + feat_names + ['image', 'query']
        dataset = pd.read_csv(f, delimiter='\t',
                              names=headers)

    ret = dataset.transform({**{'label': pairwise_labels if pairwise else pointwise_labels}, **{n: lambda x: float(x.split(':')[1]) for n in feat_names}})
    ret.insert(loc=0, column='query_group', value=pd.factorize(dataset['qid'])[0])
    return ret


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