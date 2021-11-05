import pandas as pd


def rank_preds(preds: pd.DataFrame) -> pd.DataFrame:
    """

    :param preds: Przewidywane wartości
    :type preds: pd.DataFrame
    :return: Uszeregowane przewiywane wratości od najbardziej prawdopodobnych do najmniej
    :rtype: pd.DataFrame
    """
    if not isinstance(preds, pd.DataFrame):
        raise ValueError("Preds should be type pandas.DataFrame")

    if preds.empty:
        raise ValueError("Answers list cannot be empty")

    rpreds = pd.DataFrame(preds)
    for col in preds.columns.values:
        rpreds.loc[:, col] = preds.sort_values(by=col, ascending=False).index
    return rpreds


