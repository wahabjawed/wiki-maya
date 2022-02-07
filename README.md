# Measuring user trust according to the quality of their contributions: the case of Wikipedia

Wikipedia is a distributed content repository which exhibits large scale collaboration. It is a
public encyclopedia that can be edited by anyone. The quality of the Wikipedia&#39;s content is
managed by a large community which actively reverts edits caused by spammers and
vandals. In a large-scale collaboration, we have a large number of users and we cannot
remember all collaborators and their performance. It would be good to have a trust score of
a user in order to help us choose our collaborators. In this paper, we propose a reputation
system based on trust score, that predicts the behavior of an author based on the survival
rate of its past edits. We desire to use these scores to determine the chances of vandalism
and predict longevity of future revision. Our results show that there is less chance of
vandalism if the obtained scores stabilize over time. Our system is solely based on evolution
of past contributions and helps us highlight users with low trust scores.
We also devised an algorithm to obtain trust score of a user based on the quality of their
contribution. We calculated past user contribution in terms of the quality of the article.
Determining quality of an article is another resource intensive task performed manually by
the community. This classification is essential in providing users with high quality articles
and identifying low quality content for improvement. Due to this massive throughput, the
community requires an automated tool to classify quality of articles. We used readability
scores, textual and structural features of an articles to predict its quality class. Our approach
is based on Deep Neural Network (DNN) and feature selection. Our results are comparable
with existing approaches and shows improvements in terms of accuracy and information
gain.

## Index

- [Demo](https://github.com/wahabjawed/wiki-maya#demo)
- [Issues](https://github.com/wahabjawed/wiki-maya#issues)
- [Contributing](https://github.com/wahabjawed/wiki-maya#contributing)
  - [Contributers](https://github.com/wahabjawed/wiki-maya#contributers)
- [License](https://github.com/wahabjawed/wiki-maya#license)


## Demo

[[Back to top]](https://github.com/wahabjawed/wiki-maya#index)

File | Description 
--- | ---
[testAlorithmReadiability.py](https://github.com/wahabjawed/wiki-maya/blob/master/script/testAlorithmReadiability.py) | This file contains several functions realted to "Feature Selection, Buidling testing multiple machine learning classifer, and Hyper-tune the classifer". It will output graph realted to feature importance, confusion metrics of different classifers before and after hypertuned. All the methods are called in the main section. You can run the execute the file by: "python testAlorithmReadiability.py"
[UserContributionMetric.py](https://github.com/wahabjawed/wiki-maya/blob/master/script/UserContributionMetric.py)  | The file contains function realted to plotting graph for user contribution and trust values of those contribution that are used in our thesis. It requires that the user computation ot be calculated first using [calcFeatureContrib.py](https://github.com/wahabjawed/wiki-maya/blob/master/script/calcFeatureContrib.py). All the methods are called in the main section. You can execute the file by: "python calcFeatureContrib.py" 
[UserLongevityMetric.py](https://github.com/wahabjawed/wiki-maya/blob/master/script/UserLongevityMetric.py)  | The file contains function realted to calculating text longevity, plotting graph for text longevity and its trust values that are used in our thesis. All the methods are called in the main section. You can execute the file by: "python UserLongevityMetric.py" 
[LongevityRegressionGraph.py](https://github.com/wahabjawed/wiki-maya/blob/master/script/LongevityRegressionGraph.py)  | The file contains function realted to calculating multi variable regression of Longevity and Trust Score. All the methods are called in the main section. You can run the execute the file by: "python LongevityRegressionGraph.py" 
[TrustScore.py](https://github.com/wahabjawed/wiki-maya/blob/master/script/TrustScore.py)  | The file contains function realted to calculating Trust Score. It is python implementation of [Trust Metric](https://hal.inria.fr/hal-01351250/document). You can execute the file by: "python TrustScore.py"
[calcFeatureContrib.py](https://github.com/wahabjawed/wiki-maya/blob/master/script/calcFeatureContrib.py)  | The file contains function realted to calculating user contribution and quality of contribution. All the methods are called in the main section. You can run the execute the file by: "python calcFeatureContrib.py" 
[calcFeatureNew.py](https://github.com/wahabjawed/wiki-maya/blob/master/script/calcFeatureNew.py)  | The file contains function realted to calculating readability scores and new text based features. It requires to downlaod 2015 wikipedia quality dataset from [2015 dataset](https://figshare.com/articles/English_Wikipedia_Quality_Asssessment_Dataset/1375406). All the methods are called in the main section. You can execute the file by: "python calcFeatureNew.py" 
[Machine Learning Classifiers](https://github.com/wahabjawed/wiki-maya/tree/master/maya/nltk/algorithm)  | The directory contains all the machine leanring classifers that we used for our prediction.

## Issues

[[Back to top]](https://github.com/wahabjawed/wiki-maya#index)

You can report the bugs at the [issue tracker](https://github.com/wahabjawed/wiki-maya/issues)

**OR**

You can [message me](https://www.facebook.com/wahab.jawed) if you can't get it to work.

## Contributing

[[Back to top]](https://github.com/wahabjawed/wiki-maya#index)

This program was developed to support results of my thesis, so the coding standards might not be up the mark. Don't be shy to make a Pull request :)

For making contribution:

1. Fork it
2. Clone it

```
    git clone https://github.com/wahabjawed/wiki-maya.git
    cd wiki-maya
```

use pycharm studio to open the project

### Contributers

[[Back to top]](https://github.com/wahabjawed/wiki-maya#index)

- [@wahabjawed](https://github.com/wahabjawed/)   [visit website](https://www.linkedin.com/in/abdul-wahab-47745163/)

## License

[[Back to top]](https://github.com/wahabjawed/wiki-maya#index)

Built with ♥ by [Abdul Wahab](https://www.linkedin.com/in/abdul-wahab-47745163/)[(@wahabjawed)](https://www.facebook.com/wahab.jawed) under [MIT License](http://wahabjawed.mit-license.org)

This is free software, and may be redistributed under the terms specified in the LICENSE file.

You can find a copy of the License at http://wahabjawed.mit-license.org/
