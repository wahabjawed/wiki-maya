import unittest

from maya.nltk import util


class TestUtil(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_detokenize(self):
        str = "A quick brown fox jump over a lazy dog"
        strAfterProcess = "A quick brown fox jump lazy dog"

        output = util.stop_word_removal(str)
        self.assertEqual(strAfterProcess, output)

if __name__ == '__main__':
    unittest.main()
