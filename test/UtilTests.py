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

    def test_shouldSentenceTokenWork(self):

        o_text = ['I am fine. how about you? the westher is hot']
        right_tokens = ['I am fine.', 'how about you?', 'the westher is hot']

        for text in o_text:
            sent_tokens = util.sent_tokenize(text)
            for (token, right_token) in zip(sent_tokens, right_tokens):
                self.assertEqual(token, right_token)


if __name__ == '__main__':
    unittest.main()
