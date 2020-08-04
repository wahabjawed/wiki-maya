import difflib
from difflib import SequenceMatcher
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
    """
       Compute differences between two revisions (string).
       The computed operations are filtered to keep only insertions.

       Args:
           parent_rev (str): text content of a revision.
           current_rev (str): text content of a newer revision.
       
       Result:
           sequence of insertions. 
           Each insertion is a sequence defined as follow: [index of insertion in parent_rev, text to be inserted]
    """
    diff_ops = SequenceMatcher(None, parent_rev, current_rev)
    insert_ops = []
    for (tag, i1, i2, j1, j2) in diff_ops.get_opcodes():
        if tag == 'insert' or tag == 'replace':
            insert_ops.append([i1, current_rev[j1:j2]])
    return insert_ops


def getInsertedContentSinceParentRevision(parent_rev, new_rev):
    """
   Compute text content that was inserted in a revision since another revision (string).
   The computed differences are computed, only insertions are kept. Then their text content is merged into a string.

   Args:
       parent_rev (str): text content of a revision.
       new_rev (str): text content of a newer revision.

   Result:
       the inserted text content (str)
    """
    ops = findDiffRevised(parent_rev, new_rev)
    content = "".join([o[1] for o in ops])
    return content


def textPreservedRatio(o_text, d_text):
    """
   Compute the ratio of preserved text between two revisions

   Args:
       o_text (list): a list of elements of the form [index of insertion position, text to be inserted]
       d_text (str): text content of the destination revision.

   Result:
       ratio of preserved text (real).
    """
    total = 0
    total_matched = 0
    for text in o_text:
        matches = difflib.SequenceMatcher(None, text, d_text, autojunk=False).get_matching_blocks()
        total_matched += sum(int(match[2]) for match in matches)
        total += len(text)
    return total_matched / total
