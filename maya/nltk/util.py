import difflib
import math
import re
import string
from difflib import SequenceMatcher

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
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


def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text);


def sent_tokenize(text):
    sent_token = nltk.sent_tokenize(text)
    return sent_token


def stop_word_removal(text, lang="en"):
    stopword = stopwords.words('english')
    word_tokens = nltk.word_tokenize(text)
    removing_stopwords = [word for word in word_tokens if word not in stopword]
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
    cleanr = re.compile('{{((?!{{).)*}}|\[\[(Category|category|Image|fr|nl|de).*\]\]')
    cleantext = re.sub(cleanr, '', raw_html)
    cleanr = re.compile('(==References|==Footnotes)((?:[^\n][\n]?)+)')
    cleantext = re.sub(cleanr, '', cleantext)
    cleanr = re.compile('({{)([\s\S]*?)(}})|\[\[|\]\]|\*\[\[')
    cleantext = re.sub(cleanr, '', cleantext)
    cleanr = re.compile('(==|==)|(--|--)|<.*?>')
    cleantext = re.sub(cleanr, '', cleantext)
    cleanr = re.compile('\[http((?!\]).)*\]')
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
        for words in text.split(' '):
            matches = difflib.SequenceMatcher(None, words, d_text, autojunk=False).get_matching_blocks()
            matches = sorted(matches, key=lambda e: e[2], reverse=True)
            if (int(matches[0][2]) / len(words) > 0.8):
                total_matched += int(matches[0][2])
            total += len(words)
    return round(total_matched / total, 2)


def textPreservedRatioStrict(o_text, d_text):
    """
   Compute the ratio of preserved text between two revisions using Contains method

   Args:
       o_text (list): a list of elements of the form [index of insertion position, text to be inserted]
       d_text (str): text content of the destination revision.

   Result:
       ratio of preserved text (real).
    """
    total = 0
    total_matched = 0
    dest_tokens = sent_tokenize(d_text)
    dest_tokens_words = word_tokenize(d_text)

    for text in o_text:
        for sent_token in sent_tokenize(text):
            if sent_token in dest_tokens:
                total_matched += len(sent_token)
                total += len(sent_token)
            else:
                for word in word_tokenize(sent_token):
                    if word in dest_tokens_words:
                        total_matched += len(word)
                    total += len(word)
    return round(total_matched / total, 2)


def textPreservedRatioContains(o_text, d_text):
    """
   Compute the ratio of preserved text between two revisions using Contains method

   Args:
       o_text (list): a list of elements of the form [index of insertion position, text to be inserted]
       d_text (str): text content of the destination revision.

   Result:
       ratio of preserved text (real).
    """
    total = 0
    total_matched = 0
    dest_tokens = sent_tokenize(d_text)

    for text in o_text:
        for sent_token in sent_tokenize(text):
            if sent_token in dest_tokens:
                total_matched += len(sent_token)
            total += len(sent_token)
    return round(total_matched / total, 2)


def textPreservedRatioBigram(o_text, d_text):
    """
   Compute the ratio of preserved text between two revisions using Bigram method

   Args:
       o_text (list): a list of elements of the form [index of insertion position, text to be inserted]
       d_text (str): text content of the destination revision.

   Result:
       ratio of preserved text (real).
    """
    total = 0
    total_matched = 0
    dest_tokens = sent_tokenize(d_text)
    dest_tokens_words = word_tokenize(d_text)

    for text in o_text:
        for sent_token in sent_tokenize(text):
            if sent_token in dest_tokens:
                total_matched += len(sent_token)
                total += len(sent_token)
            else:
                list_words = word_tokenize(sent_token)
                bigrams = list(nltk.bigrams(list_words))
                sizeList = len(bigrams)
                if sizeList > 1:
                    index = 0

                    while index < sizeList:
                        wordToMatch = list(bigrams[index])

                        if index == 0:
                            if isSubListInList(wordToMatch, dest_tokens_words):
                                total_matched += len(wordToMatch[0])
                        else:
                            wordToMatchL = list(bigrams[index - 1])
                            leftMatch = isSubListInList(wordToMatchL, dest_tokens_words)
                            rightMatch = isSubListInList(wordToMatch, dest_tokens_words)
                            if leftMatch or rightMatch:
                                total_matched += len(wordToMatch[0])

                            # adding the last element
                            if index + 1 == sizeList and rightMatch:
                                total_matched += len(wordToMatch[1])

                        index += 1

                else:
                    if isSubListInList(list_words, dest_tokens_words):
                        total_matched += sum(map(len, list_words))

                total += sum(map(len, list_words))

    return round(total_matched / total, 2)


def textPreservedRatioBigramEnhanced(o_text, d_text):
    """
   Compute the ratio of preserved text between two revisions using Bigram method
   including removing stopwords and matched words

   Args:
       o_text (list): a list of elements of the form [index of insertion position, text to be inserted]
       d_text (str): text content of the destination revision.

   Result:
       ratio of preserved text (real).
    """
    total = 0
    total_matched = 0
    dest_tokens_st = stop_word_removal(d_text)
    delOffset = 0

    if len(dest_tokens_st) > 0 and sum(map(len, o_text)) > 0:
        try:
            for text in o_text:
                for sent_token in sent_tokenize(text):
                    sent_token_st = stop_word_removal(sent_token)
                    if len(sent_token_st) > 0:
                        res = isSubListInListWithIndex(sent_token_st, dest_tokens_st)
                        if res[0]:
                            total_matched += sum(map(len, sent_token_st))
                            total += sum(map(len, sent_token_st))
                            del dest_tokens_st[res[1]:res[1] + len(sent_token_st)]
                        else:
                            bigrams = list(nltk.bigrams(sent_token_st))
                            sizeList = len(bigrams)
                            if sizeList > 1:
                                index = 0
                                indexToRemove = []
                                while index < sizeList:
                                    wordToMatch = list(bigrams[index])

                                    if index == 0:
                                        leftMatch = isSubListInListWithIndex(wordToMatch, dest_tokens_st)
                                        if leftMatch[0]:
                                            total_matched += len(wordToMatch[0])
                                            indexToRemove.append((leftMatch[1], delOffset))
                                    else:
                                        wordToMatchL = list(bigrams[index - 1])
                                        leftMatch = isSubListInListWithIndex(wordToMatchL, dest_tokens_st)
                                        rightMatch = isSubListInListWithIndex(wordToMatch, dest_tokens_st)
                                        if leftMatch[0]:
                                            total_matched += len(wordToMatch[0])
                                            indexToRemove.append((leftMatch[1] + 1, delOffset))
                                        elif rightMatch[0]:
                                            total_matched += len(wordToMatch[0])
                                            indexToRemove.append((rightMatch[1], delOffset))

                                        # adding the last element
                                        if index + 1 == sizeList and rightMatch[0]:
                                            total_matched += len(wordToMatch[1])
                                            indexToRemove.append((rightMatch[1] + 1, delOffset))

                                    index += 1
                                    if index % 2 == 0:
                                        delOffset = deleteFromList(dest_tokens_st, indexToRemove[0:-1], delOffset)
                                        del indexToRemove[0:-1]
                                delOffset = deleteFromList(dest_tokens_st, indexToRemove, delOffset)
                            total += sum(map(len, sent_token_st))
        except Exception as e:
            print("Algo Error: ", e)
            print("Source: ", o_text)
            print("Destination: ", d_text)

        print("Total: ", total, "Matched: ", total_matched)
        return round(total_matched / total, 2)
    else:
        return 0

def deleteFromList(alist, indexes, delOffset):
    """
       deletes from the list at certain indexes

       Args:
           alist (list): a list of elements.
           indexes (array): an array of index to delete at.

       Result:
           the updated offset after delete operation
        """
    for i in indexes:
        if i[0] + i[1] - delOffset >= 0 and i[0] + i[1] - delOffset <= len(alist):
            del alist[i[0] + i[1] - delOffset]
            delOffset = delOffset + 1

    return delOffset


def isSubListInList(sublist, alist):
    """
   Predicates that checks if a list is included in another one

   Args:
       sublist (list): a (sub)-list of elements.
       alist (list): a list in which to look if the sublist is included in.

   Result:
       True if the sublist is included in the list. False otherwise.
    """
    for i in range(len(alist) - len(sublist) + 1):
        if sublist == alist[i: i + len(sublist)]:
            return True
    return False


def isSubListInListWithIndex(sublist, alist):
    """
   Predicates that checks if a list is included in another one

   Args:
       sublist (list): a (sub)-list of elements.
       alist (list): a list in which to look if the sublist is included in.

   Result:
       (True, Index) if the sublist is included in the list. False otherwise.
    """
    for i in range(len(alist) - len(sublist) + 1):
        if sublist == alist[i: i + len(sublist)]:
            return True, i
    return False, -1
