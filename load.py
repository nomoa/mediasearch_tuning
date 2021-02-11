import pandas as pd

FEATURE_NAMES = [
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
]

def load():
    with open("./MediaSearch_20210127.tsv") as f:
        dataset = pd.read_csv(f, delimiter='\t')

    dataset.columns = ['label', 'qid'] + FEATURE_NAMES + ['image', 'query']
    dataset = dataset.transform({**{'label': lambda x: 0 if x == -1 else x}, **{n: lambda x: float(x.split(':')[1]) for n in FEATURE_NAMES}})
    return dataset
