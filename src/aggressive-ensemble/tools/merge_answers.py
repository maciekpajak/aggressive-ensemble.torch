from typing import List

import pandas as pd


def merge_answers_by_rankings(answers: List[pd.DataFrame]) -> pd.DataFrame:
    """

    :param answers:
    :type answers:
    :return:
    :rtype:
    """
    if not isinstance(answers, List):
        raise ValueError("Answers should be List of pandas.DataFrame")

    for a in answers:
        if not isinstance(a, pd.DataFrame):
            raise ValueError("Each answer in list should be type pandas.DataFrame")

    if not answers:
        raise ValueError("Answers list cannot be empty")

    tags = answers[0].iloc[:, 0].values
    columns = answers[0].columns.values

    dic = {col: pd.DataFrame(columns=tags, index=range(0, len(answers))) for col in columns}

    for ans, ans_idx in zip(answers, range(0, len(answers))):
        for col in ans:
            rank = 1
            for val in ans[col]:
                dic[col][val][ans_idx] = rank
                rank += 1

    answer = pd.DataFrame(columns=columns, index=tags)
    for d in dic:
        answer[d] = dic[d].mean()
    for col in columns:
        answer[col] = answer.sort_values(by=col, ascending=True).index
    answer = answer.reset_index(drop=True)

    return answer


def merge_answers_by_probabilities(answers: List[pd.DataFrame]) -> pd.DataFrame:
    """

    :param answers:
    :type answers:
    :return:
    :rtype:
    """
    if not isinstance(answers, List):
        raise ValueError("Answers should be List of pandas.DataFrame")

    for a in answers:
        if not isinstance(a, pd.DataFrame):
            raise ValueError("Each answer in list should be type pandas.DataFrame")

    if not answers:
        raise ValueError("Answers list cannot be empty")

    mean = answers[0] - answers[0]
    for ans in answers:
        mean = mean + ans
    mean = mean / len(answers)

    return mean