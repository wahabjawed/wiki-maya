import json
import sys
import math
import pandas as pd
import numpy as np

# Constants
from matplotlib import pyplot

C = 0.9
THRESHOLD = 0.25
PHI = 0.1
E = 0.3
MAX_ATF = 2

class TrustScore:

    def __init__(self, args):

        self.previous_beta = 0
        self.previous_aggregate_trust = 0
        self.previous_adj_atfs = 0
        self.previous_trend_factor = 0
        self.previous_send_proportion = 0

        self.previous_trust = 0
        self.atfs = 0

        self.input = []
        self.output = []

        self.arr = args[0].values
        self.maximum_sending_value = args[1]

    def calculate(self):
        for index in range(len(self.arr)):
            print("Trust for value: %0.2f =  %0.2f" % (self.CalculateTrustValue(index), self.arr[index]))

        return self.output


    def CalculateTrustValue(self, index):
        global previous_sending_value, previous_adj_atfs, atfs, previous_send_proportion, \
            previous_trend_factor, previous_trust, previous_beta, previous_aggregate_trust, maximum_sending_value, input,output

        sending_value = self.arr[index]

        send_proportion = sending_value / self.maximum_sending_value

        current_trust = math.log((send_proportion * (math.e - 1)) + 1)

        current_trust_change = abs(current_trust - self.previous_trust)

        beta = (C * current_trust_change) + ((1 - C) * self.previous_beta)

        alpha = (THRESHOLD + ((C * current_trust_change) / (1 + beta)))

        aggregate_trust = (alpha * current_trust) + ((1 - alpha) * self.previous_aggregate_trust)

        if (current_trust - aggregate_trust) > E:
            trend_factor = self.previous_trend_factor + PHI
        elif (aggregate_trust - current_trust) > E:
            trend_factor = self.previous_trend_factor - PHI
        else:
            trend_factor = self.previous_trend_factor

        if self.atfs > MAX_ATF:
            adj_atfs = self.atfs / 2
        else:
            adj_atfs = self.atfs

        if ((current_trust - aggregate_trust) > PHI):
            self.atfs = self.previous_adj_atfs + ((current_trust - aggregate_trust) / 2)
            #atfs = previous_adj_atfs
        elif ((aggregate_trust - current_trust) > PHI):
            self.atfs = self.previous_adj_atfs + ((aggregate_trust - current_trust))
        else:
            self.atfs = self.previous_adj_atfs

        if (self.atfs > MAX_ATF):
            change_rate = 0
        else:
            change_rate = math.cos((math.pi / 2) * (self.atfs / MAX_ATF))

        expected_trust = (trend_factor * current_trust) + ((1 - trend_factor) * aggregate_trust)

        trust_value = expected_trust * change_rate

        self.previous_beta = beta
        self.previous_aggregate_trust = aggregate_trust
        self.previous_trend_factor = trend_factor
        self.previous_adj_atfs = adj_atfs
        self.previous_send_proportion = send_proportion
        self.previous_trust = current_trust


        self.input.append(index)
        self.output.append(trust_value)
        return trust_value





def plotGraph():

    plot = pyplot.plot(input, output, 'b-')

    # for row in series.iterrows():
    #     if row[1][graph_for] > 1000:
    #         pyplot.annotate(row[1]['pageid'], (row[1]['timestamp'], row[1][graph_for]))

    pyplot.xticks(rotation=45, ha='right')
    pyplot.xlabel("Timestamp")
    pyplot.ylabel("Trust Value")
    ax = pyplot.gca()
    ax = pyplot.gca()

    pyplot.yticks(np.arange(0, 1.1, 0.1))
    #ax.yaxis.set_major_locator(pyplot.MaxNLocator(15))
    ax.set_xticklabels([])
    pyplot.show()



if __name__ == "__main__":
    userid = '415269'
    with open('rev_user/data/rev_list_' + userid + '-dp.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        del d['next_rev']

    graph_for = "longevityRev"

    series = pd.DataFrame(data=data)
    series = series[[graph_for]]
    series = series[series.longevityRev >=4]
    series = series.head(70)

   # maximum_sending_value = series[graph_for].max()
    maximum_sending_value =25

    arr = series[graph_for].to_numpy()
    #arr = [1,2,1,2,2,1,1,0,2]
    # print(arr)
    # arr = [2,3,2,3,2,3,4,2]
    # maximum_sending_value=4
