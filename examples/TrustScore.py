import json
import sys
import math

# Constants
import pandas as pd

c = 0.9
threshold = 0.25
phi = 0.1
E = 0.3
MAX_ATF = 2

sending_value = 0
maximum_sending_value = 0
previous_sending_value = 0
previous_beta = 0
previous_aggregate_trust = 0
previous_trend_factor = 0
previous_adj_atfs = 0
atfs = 0


def CalculateSendProportion(sending_proportion):
    return sending_value / maximum_sending_value


def CalculateTrust(send_proportion):
    return math.log(send_proportion * ((sys.float_info.epsilon - 1) + 1))


def CalculateTrustValue(index):
    previous_send_proportion = CalculateSendProportion(previous_sending_value)
    send_proportion = CalculateSendProportion(sending_value)

    previous_trust = CalculateTrust(previous_send_proportion)
    current_trust = CalculateTrust(send_proportion)

    current_trust_change = abs(current_trust - previous_trust)

    beta = (c * current_trust_change) + ((1 - c) * previous_beta)

    alpha = (threshold + ((c * current_trust_change)) / (1 + beta))

    aggregate_trust = (alpha * current_trust) + ((1 - alpha) * previous_aggregate_trust)

    if ((current_trust - aggregate_trust) > E):
        trend_factor = previous_trend_factor + phi
    elif ((aggregate_trust - current_trust) > E):
        trend_factor = previous_trend_factor - phi
    else:
        trend_factor = previous_trend_factor

    if (atfs > MAX_ATF):
        adj_atfs = atfs / 2
    else:
        adj_atfs = atfs

    if ((current_trust - aggregate_trust) > phi):
        atfs = previous_adj_atfs + ((current_trust - aggregate_trust) / 2)
    elif ((aggregate_trust - current_trust) > phi):
        atfs = previous_adj_atfs + (aggregate_trust - current_trust)
    else:
        atfs = previous_adj_atfs

    if atfs > MAX_ATF:
        change_rate = 0
    else:
        change_rate = math.cos((math.pi / 2) * (atfs / MAX_ATF))

    expected_trust = (trend_factor * current_trust) + ((1 - trend_factor) * aggregate_trust)

    trust_value = expected_trust * change_rate

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

    print(arr)

    for i in arr:
        CalculateTrustValue(i)

