import warnings
import string
import math
from functools import lru_cache

import spacy
from pyphen import Pyphen
from textstat.textstat import easy_word_set, textstat

langs = {
    "en": {  # Default config
        "fre_base": 206.835,
        "fre_sentence_length": 1.015,
        "fre_syll_per_word": 84.6,
        "syllable_threshold": 3,
    },
    "de": {
        # Toni Amstad
        "fre_base": 180,
        "fre_sentence_length": 1,
        "fre_syll_per_word": 58.5,
    },
    "es": {
        # Fernandez Huerta Readability Formula
        "fre_base": 206.84,
        "fre_sentence_length": 1.02,
        "fre_syll_per_word": 0.6,
    },
    "fr": {
        "fre_base": 207,
        "fre_sentence_length": 1.015,
        "fre_syll_per_word": 73.6,
    },
    "it": {
        # Flesch-Vacca
        "fre_base": 217,
        "fre_sentence_length": 1.3,
        "fre_syll_per_word": 0.6,
    },
    "nl": {
        # Flesch-Douma
        "fre_base": 206.835,
        "fre_sentence_length": 0.93,
        "fre_syll_per_word": 77,
    },
    "pl": {
        "syllable_threshold": 4,
    },
    "ru": {
        "fre_base": 206.835,
        "fre_sentence_length": 1.3,
        "fre_syll_per_word": 60.1,
    },
}


def legacy_round(number, points=0):
    p = 10 ** points
    return float(math.floor((number * p) + math.copysign(0.5, number))) / p


def get_grade_suffix(grade):
    """
    Select correct ordinal suffix
    """
    ordinal_map = {1: 'st', 2: 'nd', 3: 'rd'}
    teens_map = {11: 'th', 12: 'th', 13: 'th'}
    return teens_map.get(grade % 100, ordinal_map.get(grade % 10, 'th'))


class textstatistics_v2:
    __lang = "en_US"
    text_encoding = "utf-8"

    def __init__(self, text):
        self.v_remove_punctuation = None
        self.v_break_sentences = None
        self.v_lexicon_count = 0
        self.v_sentence_count = 0
        self.v_avg_sentence_length = 0
        self.text = text
        self.setup(text)

    def setup(self, text):
        self.v_remove_punctuation = self.remove_punctuation(text)
        self.v_lexicon_count = self.word_count(text)
        self.v_break_sentences = self.break_sentences(text)
        self.v_sentence_count = self.sentence_count(text)
        self.v_avg_sentence_length = self.avg_sentence_length(text)

    def compute(self):
        return [self.flesch_reading_ease(self.text), self.gunning_fog(self.text), self.smog_index(self.text),
                self.dale_chall_readability_score_v2(self.text),
                self.avg_sentence_length(self.text)]

    def _cache_clear(self):
        caching_methods = [
            method for method in dir(self)
            if callable(getattr(self, method))
               and hasattr(getattr(self, method), "cache_info")
        ]

        for method in caching_methods:
            getattr(self, method).cache_clear()

    @staticmethod
    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    @lru_cache(maxsize=128)
    def syllable_count(self, text, lang=None):
        """
        Function to calculate syllable words in a text.
        I/P - a text
        O/P - number of syllable words
        """
        if lang:
            warnings.warn(
                "The 'lang' argument has been moved to "
                "'textstats.set_lang(<lang>)'. This argument will be removed "
                "in the future.",
                DeprecationWarning
            )
        if isinstance(text, bytes):
            text = text.decode(self.text_encoding)

        text = text.lower()
        text = self.remove_punctuation(text)

        if not text:
            return 0

        dic = Pyphen(lang=self.__lang)
        count = 0
        for word in text.split(' '):
            word_hyphenated = dic.inserted(word)
            count += max(1, word_hyphenated.count("-") + 1)
        return count

    @lru_cache(maxsize=128)
    def avg_syllables_per_word(self, text, interval=None):
        syllable = self.syllable_count(text)
        words = self.word_count(text)
        try:
            if interval:
                syllables_per_word = float(syllable) * interval / float(words)
            else:
                syllables_per_word = float(syllable) / float(words)
            return legacy_round(syllables_per_word, 1)
        except ZeroDivisionError:
            return 0.0

    # Splits the text into sentences, using
    # Spacy's sentence segmentation which can
    # be found at https://spacy.io/usage/spacy-101
    @lru_cache(maxsize=128)
    def break_sentences(self, text):
        if self.v_break_sentences != None:
            return self.v_break_sentences
        else:
            nlp = spacy.load('en')
            doc = nlp(text)
            return list(doc.sents)

    # Returns Number of Words in the text
    @lru_cache(maxsize=128)
    def word_count(self, text):
        if self.v_lexicon_count > 0:
            return self.v_lexicon_count
        else:
            sentences = self.break_sentences(text)
            words = 0
            for sentence in sentences:
                words += len([token for token in sentence])
            return words

    @lru_cache(maxsize=128)
    def sentence_count(self, text):
        """
        Sentence count of a text
        """
        if self.v_sentence_count > 0:
            return self.v_sentence_count
        else:
            ignore_count = 0
            sentences = self.break_sentences(text)
            for sentence in sentences:
                if self.word_count(sentence) <= 2:
                    ignore_count += 1
            return max(1, len(sentences) - ignore_count)

    @lru_cache(maxsize=128)
    def avg_sentence_length(self, text):
        if self.v_avg_sentence_length > 0:
            return self.v_avg_sentence_length
        else:
            try:
                asl = float(self.word_count(text) / self.sentence_count(text))
                return legacy_round(asl, 1)
            except ZeroDivisionError:
                return 0.0

    # Return total Difficult Words in a text
    @lru_cache(maxsize=128)
    def difficult_words(self, text):
        # Find all words in the text
        words = []
        sentences = self.break_sentences(text)
        for sentence in sentences:
            words += [str(token) for token in sentence]

            # difficult words are those with syllables >= 2
        # easy_word_set is provide by Textstat as
        # a list of common words
        diff_words_set = set()

        for word in words:
            syllable_count = self.syllable_count(word)
            if word not in easy_word_set and syllable_count >= 2:
                diff_words_set.add(word)

        return len(diff_words_set)

    # A word is polysyllablic if it has more than 3 syllables
    # this functions returns the number of all such words
    # present in the text
    @lru_cache(maxsize=128)
    def poly_syllable_count(self, text):
        count = 0
        words = []
        sentences = self.break_sentences(text)
        for sentence in sentences:
            words += [token for token in sentence]

        for word in words:
            syllable_count = self.syllable_count(word)
            if syllable_count >= 3:
                count += 1
        return count

    @lru_cache(maxsize=128)
    def flesch_reading_ease(self, text):
        """
            Implements Flesch Formula:
            Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)
            Here,
              ASL = average sentence length (number of words
                    divided by number of sentences)
              ASW = average word length in syllables (number of syllables
                    divided by number of words)
        """
        FRE = 206.835 - float(1.015 * self.avg_sentence_length(text)) - \
              float(84.6 * self.avg_syllables_per_word(text))
        return legacy_round(FRE, 2)

    @lru_cache(maxsize=128)
    def gunning_fog(self, text):
        per_diff_words = (self.difficult_words(text) / self.word_count(text) * 100) + 5
        grade = 0.4 * (self.avg_sentence_length(text) + per_diff_words)
        return [grade, textstat.smog_index(text)]

    @lru_cache(maxsize=128)
    def smog_index(self, text):
        """
            Implements SMOG Formula / Grading
            SMOG grading = 3 + ?polysyllable count.
            Here,
               polysyllable count = number of words of more
              than two syllables in a sample of 30 sentences.
        """

        if self.sentence_count(text) >= 3:
            poly_syllab = self.poly_syllable_count(text)
            SMOG = (1.043 * (30 * (poly_syllab / self.sentence_count(text))) ** 0.5) \
                   + 3.1291
            return legacy_round(SMOG, 1)
        else:
            return 0

    @lru_cache(maxsize=128)
    def dale_chall_readability_score_v2(self, text):
        """
            Implements Dale Challe Formula:
            Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365
            Here,
                PDW = Percentage of difficult words.
                ASL = Average sentence length
        """
        words = self.word_count(text)
        # Number of words not termed as difficult words
        count = words - self.difficult_words(text)
        if words > 0:
            # Percentage of words not on difficult word list

            per = float(count) / float(words) * 100

        # diff_words stores percentage of difficult words
        diff_words = 100 - per

        raw_score = (0.1579 * diff_words) + \
                    (0.0496 * self.avg_sentence_length(text))

        # If Percentage of Difficult Words is greater than 5 %, then;
        # Adjusted Score = Raw Score + 3.6365,
        # otherwise Adjusted Score = Raw Score

        if diff_words > 5:
            raw_score += 3.6365

        return legacy_round(raw_score, 2)

#
# textstats = textstatistics(text="")
