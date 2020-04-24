import difflib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize, pos_tag, ne_chunk
from textblob import TextBlob
import spacy
from textstat.textstat import textstatistics, easy_word_set, textstat
import language_check
import re
import math


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


# Splits the text into sentences, using
# Spacy's sentence segmentation which can
# be found at https://spacy.io/usage/spacy-101
def break_sentences(text):
    nlp = spacy.load('en')
    doc = nlp(text)
    return doc.sents


# Returns Number of Words in the text
def word_count(text):
    sentences = break_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence])
    return words


# Returns the number of sentences in the text
def sentence_count(text):
    sentences = break_sentences(text)
    return len(list(sentences))


# Returns average sentence length
def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length


# Textstat is a python package, to calculate statistics from
# text to determine readability,
# complexity and grade level of a particular corpus.
# Package can be found at https://pypi.python.org/pypi/textstat
def syllables_count(word):
    return textstatistics().syllable_count(word)


# Returns the average number of syllables per
# word in the text
def avg_syllables_per_word(text):
    syllable = syllables_count(text)
    words = word_count(text)
    ASPW = float(syllable) / float(words)
    return legacy_round(ASPW, 1)


# Return total Difficult Words in a text
def difficult_words(text):
    # Find all words in the text
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]

        # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as
    # a list of common words
    diff_words_set = set()

    for word in words:
        syllable_count = syllables_count(word)
        if word not in easy_word_set and syllable_count >= 2:
            diff_words_set.add(word)

    return len(diff_words_set)


# A word is polysyllablic if it has more than 3 syllables
# this functions returns the number of all such words
# present in the text
def poly_syllable_count(text):
    count = 0
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [token for token in sentence]

    for word in words:
        syllable_count = syllables_count(word)
        if syllable_count >= 3:
            count += 1
    return count


def flesch_reading_ease(text):
    """
        Implements Flesch Formula:
        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)
        Here,
          ASL = average sentence length (number of words
                divided by number of sentences)
          ASW = average word length in syllables (number of syllables
                divided by number of words)
    """
    FRE = 206.835 - float(1.015 * avg_sentence_length(text)) - \
          float(84.6 * avg_syllables_per_word(text))
    return legacy_round(FRE, 2)


def gunning_fog(text):
    per_diff_words = (difficult_words(text) / word_count(text) * 100) + 5
    grade = 0.4 * (avg_sentence_length(text) + per_diff_words)
    return [grade, textstat.smog_index(text)]


def smog_index(text):
    """
        Implements SMOG Formula / Grading
        SMOG grading = 3 + ?polysyllable count.
        Here,
           polysyllable count = number of words of more
          than two syllables in a sample of 30 sentences.
    """

    if sentence_count(text) >= 3:
        poly_syllab = poly_syllable_count(text)
        SMOG = (1.043 * (30 * (poly_syllab / sentence_count(text))) ** 0.5) \
               + 3.1291
        return legacy_round(SMOG, 1)
    else:
        return 0


def dale_chall_readability_score(text):
    """
        Implements Dale Challe Formula:
        Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365
        Here,
            PDW = Percentage of difficult words.
            ASL = Average sentence length
    """
    words = word_count(text)
    # Number of words not termed as difficult words
    count = word_count - difficult_words(text)
    if words > 0:
        # Percentage of words not on difficult word list

        per = float(count) / float(words) * 100

    # diff_words stores percentage of difficult words
    diff_words = 100 - per

    raw_score = (0.1579 * diff_words) + \
                (0.0496 * avg_sentence_length(text))

    # If Percentage of Difficult Words is greater than 5 %, then;
    # Adjusted Score = Raw Score + 3.6365,
    # otherwise Adjusted Score = Raw Score

    if diff_words > 5:
        raw_score += 3.6365

    # return legacy_round(score, 2)


def check_grammar_error_rate(text):
    tool = language_check.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches) / len(word_tokenize(text))


def check_grammar_error_rate(tool, text, count):
    matches = tool.check(text)
    try:
        return int(len(matches) / count * 100)
    except:
        return 0


def cleanhtml(raw_html):
    cleanr = re.compile('{{((?!{{).)*}}|\[\[(.*Category).*\]\]')
    cleantext = re.sub(cleanr, '', raw_html)
    cleanr = re.compile('({{)([\s\S]*?)(}})|\[\[|\]\]|(==.*==)|<.*?>')
    cleantext = re.sub(cleanr, '', cleantext)
    cleanr = re.compile('http\S+')
    cleantext = re.sub(cleanr, '', cleantext)
    return cleantext


def ignoreTag(tag):
    left = re.compile(r'<%s\b.*?>' % tag, re.IGNORECASE | re.DOTALL)  # both <ref> and <reference>
    right = re.compile(r'</\s*%s>' % tag, re.IGNORECASE)
    options.ignored_tag_patterns.append((left, right))


def unescape(text):
    """
    Removes HTML or XML character references and entities from a text string.
    :param text The HTML (or XML) source text.
    :return The plain text, as a Unicode string, if necessary.
    """

    def fixup(m):
        text = m.group(0)
        code = m.group(1)
        try:
            if text[1] == "#":  # character reference
                if text[2] == "x":
                    return chr(int(code[1:], 16))
                else:
                    return chr(int(code))
            else:  # named entity
                return chr(name2codepoint[code])
        except:
            return text  # leave as is

    return re.sub("&#?(\w+);", fixup, text)


def clean(self, text):
    selfClosingTags = ('br', 'hr', 'nobr', 'ref', 'references', 'nowiki')

    placeholder_tags = {'math': 'formula', 'code': 'codice'}

    # Match HTML comments
    # The buggy template {{Template:T}} has a comment terminating with just "->"
    comment = re.compile(r'<!--.*?-->', re.DOTALL)

    # Match selfClosing HTML tags
    selfClosing_tag_patterns = [
        re.compile(r'<\s*%s\b[^>]*/\s*>' % tag, re.DOTALL | re.IGNORECASE) for tag in selfClosingTags
    ]

    discardElements = [
        'gallery', 'timeline', 'noinclude', 'pre',
        'table', 'tr', 'td', 'th', 'caption', 'div',
        'form', 'input', 'select', 'option', 'textarea',
        'ul', 'li', 'ol', 'dl', 'dt', 'dd', 'menu', 'dir',
        'ref', 'references', 'img', 'imagemap', 'source', 'small',
        'sub', 'sup', 'indicator'
    ]

    # Match HTML placeholder tags
    placeholder_tag_patterns = [
        (re.compile(r'<\s*%s(\s*| [^>]+?)>.*?<\s*/\s*%s\s*>' % (tag, tag), re.DOTALL | re.IGNORECASE),
         repl) for tag, repl in placeholder_tags.items()
    ]
    ignored_tag_patterns = []
    # Match preformatted lines
    preformatted = re.compile(r'^ .*?$')

    # Match external links (space separates second optional parameter)
    externalLink = re.compile(r'\[\w+[^ ]*? (.*?)]')
    externalLinkNoAnchor = re.compile(r'\[\w+[&\]]*\]')

    # Matches bold/italic
    bold_italic = re.compile(r"'''''(.*?)'''''")
    bold = re.compile(r"'''(.*?)'''")
    italic_quote = re.compile(r"''\"([^\"]*?)\"''")
    italic = re.compile(r"''(.*?)''")
    quote_quote = re.compile(r'""([^"]*?)""')

    # Matches space
    spaces = re.compile(r' {2,}')

    # Matches dots
    dots = re.compile(r'\.{4,}')

    """
    Removes irrelevant parts from :param: text.
    """

    # Collect spans
    spans = []
    # Drop HTML comments
    for m in comment.finditer(text):
        spans.append((m.start(), m.end()))

    # Drop self-closing tags
    for pattern in selfClosing_tag_patterns:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))

    # Drop ignored tags
    for left, right in ignored_tag_patterns:
        for m in left.finditer(text):
            spans.append((m.start(), m.end()))
        for m in right.finditer(text):
            spans.append((m.start(), m.end()))

    # Bulk remove all spans
    text = dropSpans(spans, text)

    # Drop discarded elements
    for tag in discardElements:
        text = dropNested(text, r'<\s*%s\b[^>/]*>' % tag, r'<\s*/\s*%s>' % tag)

    # if not options.toHTML:
    #     # Turn into text what is left (&amp;nbsp;) and <syntaxhighlight>
    text = unescape(text)

    # Expand placeholders
    for pattern, placeholder in placeholder_tag_patterns:
        index = 1
        for match in pattern.finditer(text):
            text = text.replace(match.group(), '%s_%d' % (placeholder, index))
            index += 1

    text = text.replace('<<', '«').replace('>>', '»')

    #############################################

    # Cleanup text
    text = text.replace('\t', ' ')
    text = spaces.sub(' ', text)
    text = dots.sub('...', text)
    text = re.sub(' (,:\.\)\]»)', r'\1', text)
    text = re.sub('(\[\(«) ', r'\1', text)
    text = re.sub(r'\n\W+?\n', '\n', text, flags=re.U)  # lines with only punctuations
    text = text.replace(',,', ',').replace(',.', '.')
    # if options.keep_tables:
    # the following regular expressions are used to remove the wikiml chartacters around table strucutures
    # yet keep the content. The order here is imporant so we remove certain markup like {| and then
    # then the future html attributes such as 'style'. Finally we drop the remaining '|-' that delimits cells.
    text = re.sub(r'!(?:\s)?style=\"[a-z]+:(?:\d+)%;\"', r'', text)
    text = re.sub(r'!(?:\s)?style="[a-z]+:(?:\d+)%;[a-z]+:(?:#)?(?:[0-9a-z]+)?"', r'', text)
    text = text.replace('|-', '')
    text = text.replace('|', '')
    # if options.toHTML:
    #     text = html.escape(text)
    return text


def dropSpans(spans, text):
    """
    Drop from text the blocks identified in :param spans:, possibly nested.
    """
    spans.sort()
    res = ''
    offset = 0
    for s, e in spans:
        if offset <= s:  # handle nesting
            if offset < s:
                res += text[offset:s]
            offset = e
    res += text[offset:]
    return res


def dropNested(text, openDelim, closeDelim):
    """
    A matching function for nested expressions, e.g. namespaces and tables.
    """
    openRE = re.compile(openDelim, re.IGNORECASE)
    closeRE = re.compile(closeDelim, re.IGNORECASE)
    # partition text in separate blocks { } { }
    spans = []  # pairs (s, e) for each partition
    nest = 0  # nesting level
    start = openRE.search(text, 0)
    if not start:
        return text
    end = closeRE.search(text, start.end())
    next = start
    while end:
        next = openRE.search(text, next.end())
        if not next:  # termination
            while nest:  # close all pending
                nest -= 1
                end0 = closeRE.search(text, end.end())
                if end0:
                    end = end0
                else:
                    break
            spans.append((start.start(), end.end()))
            break
        while end.end() < next.start():
            # { } {
            if nest:
                nest -= 1
                # try closing more
                last = end.end()
                end = closeRE.search(text, end.end())
                if not end:  # unbalanced
                    if spans:
                        span = (spans[0][0], last)
                    else:
                        span = (start.start(), last)
                    spans = [span]
                    break
            else:
                spans.append((start.start(), end.end()))
                # advance start, find next close
                start = next
                end = closeRE.search(text, next.end())
                break  # { }
        if next != start:
            # { { }
            nest += 1
    # collect text outside partitions
    return dropSpans(spans, text)


def read_file(path):
    file = open(path, mode='r')
    # read all lines at once
    all_of_it = file.read()
    # close the file
    file.close()
    return all_of_it


def findDiff(parent_rev, current_rev):
    output = difflib.ndiff(parent_rev.splitlines(1), current_rev.splitlines(1))
    output_list = [li for li in output if li[0] != ' ']

    output2 = difflib.ndiff(parent_rev, current_rev)
    output_list2 = []
    for i, s in enumerate(output2):
        if s[0] == '+':
            output_list2.append([s[0], s[-1], i])

    output_list2_r = []
    pre_char_i = -1
    word = ""
    w_index = -1
    for s in (output_list2):
        if (pre_char_i == -1):
            word = (s[1] + s[0])
            w_index = s[2]
            pre_char_i = s[2]
        elif pre_char_i + 1 != s[2]:
            output_list2_r.append([w_index, word])
            w_index = s[2]
            word = (s[1] + s[0])
            pre_char_i = s[2]
        else:
            word = word + ((s[1] + s[0]))
            pre_char_i = s[2]

    output3 = difflib.ndiff(parent_rev, current_rev)
    output_list3 = []
    for i, s in enumerate(output3):
        output_list3.append([s[0], s[-1], i])

    return output_list


def findDiffRevised(parent_rev, current_rev):
    output2 = difflib.ndiff(parent_rev, current_rev)
    output_list2 = []
    for i, s in enumerate(output2):
        if s[0] == '+':
            output_list2.append([s[0], s[-1], i])

    output_list2_r = []
    pre_char_i = -1
    word = ""
    w_index = -1
    for s in (output_list2):
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

    return output_list2_r


def findDiffRevised2(parent_rev, current_rev):
    output2 = difflib.ndiff(parent_rev.split('\n'), current_rev.split('\n'))
    output_list2 = []
    for i, s in enumerate(output2):
        if s[0] == '+':
            output_list2.append([s[0], s[-1], i])

    output_list2_r = []
    pre_char_i = -1
    word = ""
    w_index = -1
    for s in (output_list2):
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

    return output_list2_r


def textPreservedRatio(o_text, d_text, total):
    # val = []
    d_arr = d_text.split('\n')
    # for txt in o_text:
    #     val.append(difflib.get_close_matches(txt[1], d_arr, n=1, cutoff=0.3))
    # print(val)

    # o_text.append([3, '| birth_place  dd  = [[San Francisco, California]]'])
    percent_match = []
    acquired = 0
    for tt in o_text:
        seq = list((e, difflib.SequenceMatcher(None, tt[1], e, autojunk=False).get_matching_blocks()[0]) for e in d_arr)
        seq = [[k, _] for k, _ in sorted(seq, key=lambda e: e[-1].size, reverse=True)]
        percent_match.append([seq[0][0], seq[0][1][2], len(tt[1])])
        acquired += seq[0][1][2]

    return acquired / total
    # similar_sentence = []
    # length = len(d_arr)
    # for t in o_text:
    #     max_prob = 0.1
    #     for i in range(length):
    #             match_ratio = difflib.SequenceMatcher(None, t[1], d_arr[i]).ratio()
    #             if match_ratio > max_prob:
    #                 max_prob = match_ratio
    #                 similar_sentence.append([max_prob, match_ratio, t[1], d_arr[i]])
    # print(similar_sentence)
