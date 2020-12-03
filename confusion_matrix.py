import numpy as np

#Confustion matrix format: first index is TRUE VALUE, second index is PREDICTED VALUE
TRUE_NEGATIVE = (0, 0)
FALSE_NEGATIVE = (1, 0)

FALSE_POSITIVE = (0, 1)
TRUE_POSITIVE = (1, 1)

ACTUAL_POSITIVES = TRUE_POSITIVE, FALSE_NEGATIVE
ACTUAL_NEGATIVES = TRUE_NEGATIVE, FALSE_POSITIVE
WRONG_ANSWERS = FALSE_POSITIVE, FALSE_NEGATIVE


def print_raw_accuracy(CM):
    total_data = np.sum(CM)
    accuracy = (CM[TRUE_POSITIVE] + CM[TRUE_NEGATIVE]) / total_data
    ans = f'Total examples {total_data}; accuracy {accuracy: 3.4}, '
    if CM[TRUE_POSITIVE] + CM[FALSE_POSITIVE] > 0:
        precision = CM[TRUE_POSITIVE] / (CM[TRUE_POSITIVE] + CM[FALSE_POSITIVE])
        ans += f'Precision {precision:3.4}, False alarm rate {CM[FALSE_POSITIVE]/(CM[TRUE_POSITIVE] + CM[FALSE_POSITIVE])}, '
    else:
        ans += f'Precision undefined: no positive answers, '
    if CM[TRUE_POSITIVE] + CM[FALSE_NEGATIVE] > 0:
        recall = CM[TRUE_POSITIVE] / (CM[FALSE_NEGATIVE] + CM[TRUE_POSITIVE])
        ans += f'Recall {recall:3.4}, '
    else:
        ans += f'Recall undefined, '

    if CM[TRUE_POSITIVE] + np.sum(CM[WRONG_ANSWERS]) > 0:
        f1score = 2 * CM[TRUE_POSITIVE] / (2 * CM[TRUE_POSITIVE] + np.sum(CM[WRONG_ANSWERS]))
        ans += f'F1score {f1score:3.4} '
    else:
        ans += f'F1score undefined: no positive data points '
    ans += '.'
    print(ans)
    return ans


def print_event_accuracy(CM):
    """
    Interpret confusion matrix as event-detection matrix. In this case, cell for TRUE_NEGATIVE will be assumed zero.
    :param CM:
    :return:
    """
    assert CM[TRUE_NEGATIVE] == 0, 'Provided matrix is not event-accuracy matrix'
    total_events = CM[ACTUAL_POSITIVES].sum()
    ans = f'Total events, {total_events}, '
    if total_events > 0:
        ans += f'detected, {CM[TRUE_POSITIVE]} , {CM[TRUE_POSITIVE] / total_events:3.4}, '
        ans += f'missed, {CM[FALSE_NEGATIVE]} , {CM[FALSE_NEGATIVE] / total_events:3.4}, '
        ans += f'false alarms, {CM[FALSE_POSITIVE]} , {CM[FALSE_POSITIVE] / total_events:3.4} '
    else:
        ans += f'false alarms, {CM[FALSE_POSITIVE]}, (no valid events for ratio)'

    print(ans)
    return ans


def to_mongo(cm:np.ndarray):
    def conv(x):
        if int(x) == x:
            return int(x)
        else:
            return float(x)
    return [conv(i) for i in cm.flatten()]


def from_mongo(arr):
    return np.array(arr).reshape([2, 2])

if __name__ == "__main__":
    cm = np.array([[50., 17.],
                   [3., 60.]])

    print(cm[FALSE_POSITIVE])
    print(cm[FALSE_NEGATIVE])
    print_raw_accuracy(cm)
    cm[TRUE_NEGATIVE] = 0
    print_event_accuracy(cm)
