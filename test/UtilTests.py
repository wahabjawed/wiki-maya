import unittest

from maya.nltk import util


class TestUtil(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_detokenize(self):
        str = "A quick brown fox jump over a lazy dog"
        strAfterProcess = "A quick brown fox jump lazy dog"

        output = util.stop_word_removal(str)
        output = util.detokenize(output)
        self.assertEqual(strAfterProcess, output)

    def test_shouldSentenceTokenWork(self):

        o_text = ['I am fine. how about you? the weather is hot']
        right_tokens = ['I am fine.', 'how about you?', 'the weather is hot']

        for text in o_text:
            sent_tokens = util.sent_tokenize(text)
            for (token, right_token) in zip(sent_tokens, right_tokens):
                self.assertEqual(token, right_token)

    def test_shouldContainMethodWork(self):

        o_text = ['I am fine, how about you?', 'One needs to hydrate regularly.']
        d_text = 'I am fine, how about you? this weather is really hot and humid. ' \
                 'One needs to hydrate regularly to survive.'

        d_sent_token = util.sent_tokenize(d_text)

        sent_tokens = util.sent_tokenize(o_text[0])
        for sent_token in sent_tokens:
            self.assertTrue(sent_token in d_sent_token)

        sent_tokens = util.sent_tokenize(o_text[1])
        for sent_token in sent_tokens:
            self.assertFalse(sent_token in d_sent_token)


    def test_shouldContainMethodWithWordWork(self):

        o_text = ['One needs to hydrate regularly.', 'intersteller was nice movie']
        d_text = 'I am fine, how about you? this weather is really hot and humid. ' \
                 'One needs to hydrate regularly to survive.'

        d_sent_token = util.sent_tokenize(d_text)
        d_word_token = util.word_tokenize(d_text)

        sent_tokens = util.sent_tokenize(o_text[0])
        for sent_token in sent_tokens:
            self.assertFalse(sent_token in d_sent_token)
            for word in util.word_tokenize(sent_token):
                self.assertTrue(word in d_word_token)

        sent_tokens = util.sent_tokenize(o_text[1])
        for sent_token in sent_tokens:
            self.assertFalse(sent_token in d_sent_token)
            for word in util.word_tokenize(sent_token):
                self.assertFalse(word in d_word_token)

    def test_containMethod(self):

        o_text = ['One needs to hydrate regularly.', 'intersteller was nice movie']
        d_text = 'I am fine, how about you? this weather is really hot and humid. ' \
                 'One needs to hydrate regularly to survive.'

        ratio = util.textPreservedRatioContains(o_text, d_text)

        self.assertEqual(ratio, 0.0)

        o_text = ['One needs to hydrate regularly to survive.', 'this weather is']

        ratio = util.textPreservedRatioContains(o_text, d_text)

        self.assertEqual(ratio, 0.74)


    def test_containMethodWithWords(self):

        o_text = ['Cant See that movie', 'intersteller was nice movie']
        d_text = 'I am fine, how about you? this weather is really hot and humid. ' \
                 'One needs to hydrate regularly to survive.'

        ratio = util.textPreservedRatioStrict(o_text, d_text)

        self.assertEqual(ratio, 0.0)

        o_text = ['One needs to hydrate regularly to survive.', 'this weather is']

        ratio = util.textPreservedRatioStrict(o_text, d_text)

        self.assertEqual(ratio, 1)

    def test_bigramMethod(self):

        o_text = ['Cant See this weather', 'intersteller was nice movie']
        d_text = 'I am fine, how about you? this weather is really hot and humid. ' \
                 'One needs to hydrate regularly to survive.'

        ratio = util.textPreservedRatioBigram(o_text, d_text)

        self.assertEqual(ratio, 0.26)

        o_text = ['One needs to hydrate regularly to survive.', 'this weather really']

        ratio = util.textPreservedRatioBigram(o_text, d_text)

        self.assertEqual(ratio, 0.9)


        o_text = ['One needs', 'two three']

        ratio = util.textPreservedRatioBigram(o_text, d_text)

        self.assertEqual(ratio, 0.5)

    def test_bigramMethodEnhanced(self):


        o_text = ['(2001) ', ' is that terrapin nests and were found last year,', ', Hampshire)', ' very', '<em>', '</em>']
        d_text ="The '''''', ''Trachemys scripta elegans'' is native to the southern [[nited States]], and has become common in the UKIt is a medium-ration to a tortoise, ranging in sizeKeared terrapins are not native,,(2001)  is that terrapin nests and eggs were found last year,t[[snapping turtles]]6, Hampshire) very'''', tterrapins are members of the group [[]], referring to reptiles with a shell, which contains"

        ratio = util.textPreservedRatioBigramEnhanced(o_text, d_text)

        self.assertEqual(ratio, 0.83)

        o_text = ['Cant See this weather. I am fine', 'intersteller was nice movie']
        d_text = 'I am fine, how about you? this weather is really hot and humid. ' \
                 'One needs to hydrate regularly to survive.'

        ratio = util.textPreservedRatioBigramEnhanced(o_text, d_text)

        self.assertEqual(ratio, 0.12)

        o_text = ['One needs to hydrate regularly to survive.', 'this weather really']

        ratio = util.textPreservedRatioBigramEnhanced(o_text, d_text)

        self.assertEqual(ratio, 1.0)


        o_text = ['One needs', 'two three']

        ratio = util.textPreservedRatioBigramEnhanced(o_text, d_text)

        self.assertEqual(ratio, 0.5)

        o_text = ['One needs to regularly to survive.']

        ratio = util.textPreservedRatioBigramEnhanced(o_text, d_text)

        self.assertEqual(ratio, 1.0)


    def test_removeWikiSyntax(self):

        stringWiki = "'''3rd/4th Cavalry Regiment''' is an armoured [[regiment]] of the [[Australian Army]], and is third in seniority in the [[Royal Australian Armoured Corps]]. The regiment was formed in [[1981]] through the amalgamation of the '''3rd Cavalry Regiment''' and the '''4th Cavalry Regiment'''." \
'==3rd Cavalry==' \
"The 3rd Cavalry Regiment was formed in [[1966]] as '''1st Armoured Personnel Carrier Squadron''' by the redesignation of a troop from the [[4th/19th Prince of Wales's Light Horse]], before being designated the 3rd Cavalry the same year. This unit saw extensive service throughout Australia's presence in [[Vietnam War|Vietnam]] as part of the [[Australian Task Force]]. At the time, the regiment consisted of a single squadron. However, part of A Squadron was redesignated first as 1 APC Squadron, then as B Squadron in [[1967]]-[[1968|68]]. The two squadrons swapped titles at various times until [[1971]], when A Squadron was transferred to the [[2nd Cavalry Regiment]]."\
'==4th Cavalry==' \
'The 4th Cavalry Regiment was formed in [[1971]] by the redesignation of B Squadron, [[2nd Cavalry Regiment|2nd Cavalry]] as A Squadron. This was joined in [[1975]] by a newly raised Regimental Headquarters and B Squadron. The service of the regiment was short lived, as in 1981 it was decided to amalgamate it with the 3rd Cavalry Regiment to form the 3rd/4th Cavalry Regiment.' \
'==Current Role==' \
'The 3rd/4th Cavalry Regiment currently consists of a single squadron, B Squadron. It serves as an [[armoured personnel carrier]] squadron, equipped with the [[M113]] vehicle, in the light armoured role.'\
\
"*'''Battle Honours'''"\
"**(inherited battle honours of 3rd Cavalry Regiment)"\
"**Long Tan, Bien Hoa, Coral-Balmoral, Hat Dich, Binh Ba, Vietnam 1965-72"\
\
"[[Category:Australian regiments]]"

        stringClean = ""
        stringWikiP = util.cleanhtml(stringWiki)

        self.assertEqual(stringClean, stringWikiP)


    # def test_bigramMethodEnhancedClean(self):
    #
    #
    #     o_text = ['(2001) ', ' is that terrapin nests and were found last year,', ', Hampshire)', ' very', '<em>', '</em>']
    #
    #     for i in range(len(o_text)):
    #         o_text[i] = util.cleanhtml(o_text[i])
    #
    #     d_text = util.cleanhtml("The '''''', ''Trachemys scripta elegans'' is native to the southern [[nited States]], and has become common in the UKIt is a medium-ration to a tortoise, ranging in sizeKeared terrapins are not native,,(2001)  is that terrapin nests and eggs were found last year,t[[snapping turtles]]6, Hampshire) very'''', tterrapins are members of the group [[]], referring to reptiles with a shell, which contains")
    #
    #     ratio = util.textPreservedRatioBigramEnhanced(o_text, d_text)
    #
    #     self.assertEqual(ratio, 1.0)
    #
    #     o_text = ['Cant See this weather. I am fine', 'intersteller was nice movie']
    #     d_text = util.cleanhtml('I am fine, how about you? this weather is really hot and humid. ' \
    #              'One needs to hydrate regularly to survive.')
    #
    #     for i in range(len(o_text)):
    #         o_text[i] = util.cleanhtml(o_text[i])
    #
    #     ratio = util.textPreservedRatioBigramEnhanced(o_text, d_text)
    #
    #     self.assertEqual(ratio, 0.12)
    #
    #     o_text = ['One needs to hydrate regularly to survive.', 'this weather really']
    #
    #     for i in range(len(o_text)):
    #         o_text[i] = util.cleanhtml(o_text[i])
    #
    #     ratio = util.textPreservedRatioBigramEnhanced(o_text, d_text)
    #
    #     self.assertEqual(ratio, 1.0)
    #
    #     o_text = ['One needs', 'two three']
    #
    #     for i in range(len(o_text)):
    #         o_text[i] = util.cleanhtml(o_text[i])
    #
    #     ratio = util.textPreservedRatioBigramEnhanced(o_text, d_text)
    #
    #     self.assertEqual(ratio, 0.5)
    #
    #     o_text = ['One needs to regularly to survive.']
    #
    #     for i in range(len(o_text)):
    #         o_text[i] = util.cleanhtml(o_text[i])
    #
    #     ratio = util.textPreservedRatioBigramEnhanced(o_text, d_text)
    #
    #     self.assertEqual(ratio, 1.0)

    def test_ExtractOriginalContribution(self):
        source = "abc ghi mno"
        destination = "abc def ghi jkl mno"

        ratio = util.findDiffRevised(source, destination)
        self.assertEqual(2,len(ratio))

    def test_DiffOfContributions(self):
        parent_rev = [
            "I think the article could  widfdfdth a review.\nFrom memory dfdfdidn't one of our pilots get some dirty US looks for canceling a mission when he decided he couldn't reliably isolate the intended target, as per his Aust. orders accuracy in avoiding civilians had top priority.",
            "Thanks Cunch. I a guess you are right."]
        current_rev = util.read_file('../script/rev_user/22272908')

        ratio = util.textPreservedRatio([parent_rev[1]], current_rev)

        self.assertEqual(0.77, ratio)


if __name__ == '__main__':
    unittest.main()
