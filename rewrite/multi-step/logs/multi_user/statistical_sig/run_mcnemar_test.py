from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
import argparse

def make_contingency_table(a_pred: list, b_pred: list, gold: list) -> list:
    """Make a contingency table to do McNemar test.
       https://machinelearningmastery.com/mcnemars-test-for-machine-learning/

    Args:
        a_pred (`list`): A list of predictions from system A.
        b_pred (`list`): A list of predictions from system B.
        gold (`list`): A list of gold values.

    Returns:
        `list` of `list`: A square contingency table.
    """

    tf_list_a = _compare(a_pred, gold)
    tf_list_b = _compare(b_pred, gold)

    both_correct = 0
    a_correct_b_incorrect = 0
    a_incorrect_b_correct = 0
    both_incorrect = 0

    for (a, b) in zip(tf_list_a, tf_list_b):
        if a == True and b == True:
            both_correct += 1
        elif a == True and b == False:
            a_correct_b_incorrect += 1
        elif a == False and b == True:
            a_incorrect_b_correct += 1
        elif a == False and b == False:
            both_incorrect += 1
        else:
            print('Error')

    return [[both_correct, a_correct_b_incorrect],
            [a_incorrect_b_correct, both_incorrect]]


def _compare(x_list: list, y_list: list) -> list:
    return list(np.char.equal(x_list, y_list))

def read_data(path: str) -> list:
    with open(path) as f:
        return [x.strip() for x in f.readlines()]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_a', type=str, required=True,
                        help='Path to system a predictions.')
    parser.add_argument('--pred_b', type=str, required=True,
                        help='Path to system b predictions.')
    parser.add_argument('--gold', type=str, required=True,
                        help='Path to gold reference data.')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level.')

    args = parser.parse_args()

    pred_a = read_data(args.pred_a)
    pred_b = read_data(args.pred_b)
    gold = read_data(args.gold)

    contingency_table = make_contingency_table(pred_a, pred_b, gold)

    # calculate mcnemar test
    result = mcnemar(contingency_table, exact=True)

    # summarize the findings
    print(f'Statistics:  {result.statistic: .2f}')
    print(f'P-value:     {result.pvalue: .3f}')

    # interpret the p-value
    if result.pvalue > args.alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')

