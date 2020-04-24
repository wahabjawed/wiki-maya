from maya.nltk.algorithm.RandomForest import RandomForest

if __name__ == "__main__":
    train_data_path = "../data/2017_english_wikipedia_quality_dataset/datasets/training-set-n.csv"
    test_data_path = "../data/2017_english_wikipedia_quality_dataset/datasets/test-set-n.csv"

    random = RandomForest([test_data_path, train_data_path])
    random.learn()
    random.fetchScore()