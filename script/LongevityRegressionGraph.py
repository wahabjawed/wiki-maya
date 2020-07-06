import json
import sys

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from script.TrustScore import TrustScore

sys.path.append("..")
from mwapi import Session
from maya.extractors import api
import pandas as pd
from matplotlib import pyplot

session = Session("https://en.wikipedia.org/w/api.php", user_agent="test")
api_extractor = api.Extractor(session)


def plotLongevityRegressionGraph():
    with open('user_data/rev_list_36440187-dp.json', 'r') as infile:
        data = json.loads(infile.read())
    for d in data:
        del d['next_rev']

    graph_for = "longevityRev"

    series1 = pd.DataFrame(data=data)
    series1 = series1[['pageid', 'timestamp', graph_for]]
    series1 = series1[series1.longevityRev >= 0]
    series1['longevityRevN'] = series1['longevityRev'].shift(-1)
    series1['longevityRevP'] = series1['longevityRev'].shift(1)
    series1 = series1[series1.longevityRevP >= 0]
    series1 = series1[series1.longevityRevN >= 0]
    y = TrustScore([series1['longevityRev'], 24]).calculate()

    arr1 = ['5834659','415269','39180130','9929111','29047545','36437087']

    for id in arr1:

        with open('user_data/rev_list_'+id+'-dp.json', 'r') as infile:
            data = json.loads(infile.read())
        for d in data:
            del d['next_rev']

        series5 = pd.DataFrame(data=data)
        series5 = series5[['pageid', 'timestamp', graph_for]]
        series5 = series5[series5.longevityRev >= 0]
        series5['longevityRevN'] = series5['longevityRev'].shift(-1)
        series5['longevityRevP'] = series5['longevityRev'].shift(1)
        series5 = series5[series5.longevityRevP >= 0]
        series5 = series5[series5.longevityRevN >= 0]
        y.extend(TrustScore([series5['longevityRev'], 24]).calculate())

        series1 = series1.append(series5)

    #series1["longevityRevP"] = series1["longevityRevP"].fillna(0)
    #series1["longevityRevN"] = series1["longevityRevN"].fillna(0)
    series1["Trust"] = y
    series1['TrustP'] = series1['Trust'].shift(1)
    series1["TrustP"] = series1["TrustP"].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        series1[['Trust', 'longevityRev','longevityRevP']].values.reshape(-1, 3),
        series1['longevityRevN'].values.reshape(-1, 1), test_size=0.10, random_state=17)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # To retrieve the intercept:
    print(regressor.intercept_)
    # For retrieving the slope:
    print(regressor.coef_)

    y_pred = regressor.predict(X_test)

    y_pred = y_pred.reshape(1, -1)[0]
    y_test = y_test.reshape(1, -1)[0]

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('R2 Score:', metrics.r2_score(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    df.plot(kind='bar', figsize=(10, 8))
    pyplot.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    pyplot.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    pyplot.xlabel("Samples")
    pyplot.ylabel("Longevity (Revision)")
    pyplot.show()


if __name__ == "__main__":
    # following method compute regression of 3 user user_data [36440187,415269,39180130]

    plotLongevityRegressionGraph()
