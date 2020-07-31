import difflib
import math
import re
import string

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob


def convert_text_lower_case(text):
    lower_text = text.lower()
    print(lower_text)
    return lower_text


def legacy_round(number, points=0):
    p = 10 ** points
    return float(math.floor((number * p) + math.copysign(0.5, number))) / p


def word_tokenize(text):
    word_tokens = nltk.word_tokenize(text)
    return word_tokens


def sent_tokenize(text):
    sent_token = nltk.sent_tokenize(text)
    print(sent_token)
    return sent_token


def stop_word_removal(text, lang="en"):
    # from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    # from spacy.lang.en.stop_words import STOP_WORDS
    stopword = stopwords.words('english')
    word_tokens = nltk.word_tokenize(text)
    removing_stopwords = [word for word in word_tokens if word not in stopword]
    print(removing_stopwords)
    return removing_stopwords


def extract_lemma(text, lang="en"):
    wordnet_lemmatizer = WordNetLemmatizer()
    word_tokens = nltk.word_tokenize(text)
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
    print(lemmatized_word)
    return lemmatized_word


def remove_number(text):
    result = re.sub(r'\d +', '', text)
    print(result)
    return result


def remove_punctuation(text):
    result = text.translate(string.maketrans("", ""), string.punctuation)
    print(result)
    return result


def stemming(text):
    stemmer = PorterStemmer()
    input_str = word_tokenize(text)
    stemmed_word = [stemmer.stem(word) for word in input_str]
    return stemmed_word


def pos_tagging(text):
    result = TextBlob(text)
    print(result.tags)
    return result.tags


def chunking(text):
    reg_exp = "NP: { < DT >? < JJ > * < NN >}"
    rp = nltk.RegexpParser(reg_exp)
    result = rp.parse(pos_tagging(text))
    print(result)
    return result


def named_entity_recognition(text):
    ner = (ne_chunk(pos_tag(word_tokenize(text))))
    print(ner)
    return ner


def check_grammar_error_rate_o(tool,text):
    matches = tool.check(text)
    try:
        return (len(matches) / len(word_tokenize(text)))*100
    except:
        return 0

def check_grammar_error_rate_s(tool,text):
    matches = tool.check(text)
    try:
        return (len(matches) / len(sent_tokenize(text)))
    except:
        return 0


def check_grammar_error_rate(tool, text, count):
    matches = tool.check(text)
    try:
        return int(len(matches) / count * 100)
    except:
        return 0


def cleanhtml(raw_html):
    cleanr = re.compile('{{((?!{{).)*}}|\[\[(.*Category|category|Image|fr|nl|de).*\]\]')
    cleantext = re.sub(cleanr, '', raw_html)
    cleanr = re.compile('(==References|==Footnotes)((?:[^\n][\n]?)+)')
    cleantext = re.sub(cleanr, '', cleantext)
    cleanr = re.compile('({{)([\s\S]*?)(}})|\[\[|\]\]|(==.*==)|<.*?>|\[http((?!\]).)*\]')
    cleantext = re.sub(cleanr, '', cleantext)
    cleanr = re.compile('http\S+')
    cleantext = re.sub(cleanr, '', cleantext)
    return cleantext



def read_file(path):
    file = open(path, mode='r')
    # read all lines at once
    all_of_it = file.read()
    # close the file
    file.close()
    return all_of_it


def findDiffRevised(parent_rev, current_rev):
    #ftching diff sequence
    diff_sequence = difflib.ndiff(parent_rev, current_rev)
    add_diff_sequence = []

    #filtering out sequence with +
    for i, s in enumerate(diff_sequence):
        if s[0] == '+':
            add_diff_sequence.append([s[0], s[-1], i])

    # mering char sequence into words and sentences
    output_list2_r = []
    pre_char_i = -1
    word = ""
    w_index = -1
    for s in (add_diff_sequence):
        if (pre_char_i == -1):
            word = (s[1])
            w_index = s[2]
            pre_char_i = s[2]
        elif pre_char_i + 1 != s[2]:
            output_list2_r.append([w_index, word])
            w_index = s[2]
            word = (s[1])
            pre_char_i = s[2]
        else:
            word = word + (s[1])
            pre_char_i = s[2]

    output_list2_r.append([w_index, word])
    return output_list2_r


def textPreservedRatio(o_text, d_text):
    ratio = 0
    seq = []
    for text in o_text:
        seq = difflib.SequenceMatcher(None, text, d_text, autojunk=False).get_matching_blocks()
        total_matched = sum(int(v[2]) for v in seq)
        ratio += total_matched/len(text)

    return ratio/len(o_text)