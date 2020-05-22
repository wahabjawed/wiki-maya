import json
import sys
import math
import pandas as pd
import numpy as np

# Constants
C = 0.9
THRESHOLD = 0.25
PHI = 0.1
E = 0.3
MAX_ATF = 2

maximum_sending_value=0
previous_beta = 0
previous_aggregate_trust = 0
previous_trend_factor = 0
previous_adj_atfs = 0
previous_trend_factor = 0
previous_send_proportion = 0

previous_trust = 0
atfs = 0


def CalculateSendProportion(sending_value, maximum_sending_value):
    return sending_value / maximum_sending_value


def CalculateTrust(send_proportion):
    return math.log(send_proportion * ((math.e - 1) + 1))


def CalculateTrustValue(index):
    global previous_sending_value, previous_adj_atfs, atfs, previous_send_proportion, \
        previous_trend_factor, previous_trust, previous_beta, previous_aggregate_trust, maximum_sending_value

    sending_value = arr[index]

    send_proportion = CalculateSendProportion(sending_value, maximum_sending_value)

    current_trust = CalculateTrust(send_proportion)

    current_trust_change = abs(current_trust - previous_trust)

    beta = (C * current_trust_change) + ((1 - C) * previous_beta)

    alpha = (THRESHOLD + ((C * current_trust_change) / (1 + beta)))

    aggregate_trust = (alpha * current_trust) + ((1 - alpha) * previous_aggregate_trust)

    if (current_trust - aggregate_trust) > E:
        trend_factor = previous_trend_factor + PHI
    elif (aggregate_trust - current_trust) > E:
        trend_factor = previous_trend_factor - PHI
    else:
        trend_factor = previous_trend_factor

    if atfs > MAX_ATF:
        adj_atfs = atfs / 2
    else:
        adj_atfs = atfs

    if ((current_trust - aggregate_trust) > PHI):
        atfs = previous_adj_atfs + ((current_trust - aggregate_trust) / 2)
    elif ((aggregate_trust - current_trust) > PHI):
        atfs = previous_adj_atfs + (aggregate_trust - current_trust)
    else:
        atfs = previous_adj_atfs

    if (atfs > MAX_ATF):
        change_rate = 0
    else:
        change_rate = math.cos((math.pi / 2) * (atfs / MAX_ATF))

    expected_trust = (trend_factor * current_trust) + ((1 - trend_factor) * aggregate_trust)

    trust_value = expected_trust * change_rate

    previous_beta = beta
    previous_aggregate_trust = aggregate_trust
    previous_trend_factor = trend_factor
    previous_adj_atfs = adj_atfs
    previous_send_proportion = send_proportion
    previous_trust = current_trust



    return trust_value


if __name__ == "__main__":
    userid = '5834659'
    with open('rev_user/data/rev_list_' + userid + '-dp.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        del d['next_rev']

    graph_for = "size"

    series = pd.DataFrame(data=data)
    series = series[[graph_for]]

    maximum_sending_value = series[graph_for].max()

    arr = series[graph_for].to_numpy()

    # print(arr)
    # arr = [2,3,2,3,2,3,4,2]
    # maximum_sending_value=4
    for index in range(len(arr)):
        print("Trust for value: %0.2f =  %0.2f" % (CalculateTrustValue(index), arr[index]))
