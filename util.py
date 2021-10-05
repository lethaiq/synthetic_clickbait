from sklearn.externals import joblib
import jsonlines
import pandas as pd
import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.probability import FreqDist
# import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import json
from sklearn.model_selection import learning_curve
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
import math
from sklearn.metrics import accuracy_score
from sklearn import tree
import sklearn
from nltk.translate.ribes_score import position_of_ngram
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from nltk.data import load
# from readability_score.calculators.fleschkincaid import *
import re
from similar_text import similar_text
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import numpy as np
import sys
from similarity import *
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
# from readability_score.calculators.dalechall import *

from sklearn.ensemble import RandomForestRegressor
import os

tagdict = load('help/tagsets/upenn_tagset.pickle')
SEMTIMENT_THRESHOLD = 0.7


def remove_duplicates(data_input, y):
    ids = range(data_input.shape[0])
    data = data_input.copy()
    le = preprocessing.LabelEncoder()
    data_int = le.fit_transform(data)
    count = np.bincount(data_int)
    dups = le.inverse_transform(np.where(count > 1))[0]
    to_keep = []
    added_dup = []
    for i in range(len(data_input)):
        if data_input[i] not in dups:
            to_keep.append(i)
        if data_input[i] in dups and data_input[i] not in added_dup:
            to_keep.append(i)
            added_dup.append(data_input[i])
    data = data[to_keep]
    print("after remove {} duplicates, remains: {}".format(dups.shape[0], data.shape[0]))
    labels = y.copy()
    labels = labels[to_keep]
    return data, labels

# checked


def loadClassifier(_path):
    clf = joblib.load(_path)
    return clf

# checked


def generatePosSequence(s):
    tokens = nltk.wordpunct_tokenize(s)
    tags = nltk.pos_tag(tokens)
    return tags

# stemming and generating tokens


def generateTokens(s):
    stemmer = SnowballStemmer("english")
    # wnl = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(s)
#     tokens = [stemmer.stem(wnl.lemmatize(token)) for token in tokens]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

# checked


def generateFdist(sentTokens):
    fdist = FreqDist()
    tokens = []
    for sentToken in sentTokens:
        for token in sentToken:
            fdist[token] += 1
            if token not in tokens:
                tokens.append(token)
    return fdist, tokens

# checked


def getParseTree(cp, sentPOSs):
    sentPOSs = [(a[0], a[1]) for a in sentPOSs]
    # print(sentPOSs)
    if (len(sentPOSs) > 0):
        return cp.parse(sentPOSs)
    else:
        return None


# def findPosPattern(grammar, POSs, patternName):
#     rt = []
#     cp = nltk.RegexpParser(grammar)
#     for sentPOSs in POSs:
#         # print('Parsing:{}'.format(sentPOSs))
#         # tree = cp.parse(sentPOSs)
#         tree = getParseTree(cp, sentPOSs)
#         if (tree != None):
#             for subtree in tree.subtrees():
#                 if subtree.label() == patternName:
#                     # print(sentPOSs)
#                     # print('==>{}'.format(subtree))
#                     rt.append(subtree)

#     return np.array(rt)

# checked
def isPosPattern(grammar, sentPOSs, patternName):
    cp = nltk.RegexpParser(grammar)
    # tree = cp.parse(sentPOSs)
    tree = getParseTree(cp, sentPOSs)
    if (tree != None):
        for subtree in tree.subtrees():
            if subtree.label() == patternName:
                return True
    return False

# checked


def countPosPattern(grammar, sentPOSs, patternName):
    cp = nltk.RegexpParser(grammar)
    # tree = cp.parse(sentPOSs)
    tree = getParseTree(cp, sentPOSs)
    count = 0
    if (tree != None):
        for subtree in tree.subtrees():
            if subtree.label() == patternName:
                count += 1
    return count


# checked
def findPosPatternIncludeWord(grammar, sentPOSs, patternName, words):
    rt = []
    cp = nltk.RegexpParser(grammar)

    # tree = cp.parse(sentPOSs)
    tree = getParseTree(cp, sentPOSs)
    if (tree != None):
        for subtree in tree.subtrees():
            if subtree.label() == patternName:
                #                 subtree.pprint
                check = 0
                for POS in subtree.pos():
                    for word in words:
                        if word in POS[0]:
                            check += 1
                if (check > 0):
                    rt.append(subtree)

    return np.array(rt)


# checked
def generatePosUnigramFeatures(ids, POSs, patterns, patternNames):
    df = pd.DataFrame()
    df['id'] = ids

    feat = {}
    for patternName in patternNames:
        feat[patternName] = []

    for sentPOSs in POSs:
        for i in range(len(patterns)):
            patternName = patternNames[i]
            grammar = r"""{}:{}""".format(
                patternName, "{<" + patterns[i] + ">}")
            count = countPosPattern(grammar, sentPOSs, patternName)
            feat[patternName].append(count)

    for patternName in patternNames:
        df[patternName] = feat[patternName]

    return df


def generateTopPosNgram(POSs, m, n):
    patterns = []
    patternNames = []
    patternsFreq = findTopPosNgrams(POSs, m, n)

    for k, v in patternsFreq:
        patterns.append(list(k))
        patternNames.append('-'.join(list(k)))

    return patterns, patternNames


def generatePosNgramsFeatures(ids, POSs, patterns, patternNames, m, n):
    df = pd.DataFrame()
    df['id'] = ids

#     patterns, patternNames = generateTopPosNgram(POSs, m, n)

    feat = {}
    for patternName in patternNames:
        feat[patternName] = []

    for sentPOSs in POSs:
        for i in range(len(patterns)):
            patternName = patternNames[i]
            pattern = patterns[i]
            patternx = ["<" + p + ">" for p in pattern]
            grammar = r"""{}:{}""".format(
                patternName, "{" + "".join(patternx) + "}")
            count = countPosPattern(grammar, sentPOSs, patternName)
            feat[patternName].append(count)

    # print(len(feat))
    for patternName in patternNames:
        df["POS_" + str(n) + "_gram_" + patternName] = feat[patternName]

    return df

# Find top k POS N-grams: n = 2 or n = 3


def findTopPosNgrams(POSs, m, n):
    fdistPOS = FreqDist()
    for sentPOSs in POSs:
        onlyPOSs = [k[1] for k in sentPOSs]

        if (n == 2):
            onlyPOSs = nltk.bigrams(onlyPOSs)
        elif (n == 3):
            onlyPOSs = nltk.trigrams(onlyPOSs)

        for onlyPOS in onlyPOSs:
            if ':' not in onlyPOS and '#' not in onlyPOS and '@' not in onlyPOS and '?' not in onlyPOS:
                fdistPOS[onlyPOS] += 1

    POSngrams = fdistPOS.most_common(m)
    return POSngrams


def ngramFreqClass(title, n):
    sentTokens = list(map(generateTokens, title))

    fdist = FreqDist()
    for sent in sentTokens:
        if (n == 2):
            ngrams = nltk.bigrams(sent)
        elif (n == 3):
            ngrams = nltk.trigrams(sent)
        elif (n == 1):
            ngrams = sent

        for ngram in ngrams:
            fdist[ngram] += 1

    return fdist


def ngramFreq(title):
    sentTokens = list(map(generateTokens, title))

    fdist = FreqDist()
    for sent in sentTokens:
        bigrams = nltk.bigrams(sent)
        for bigram in bigrams:
            fdist[bigram] += 1

    return fdist


def generateNgramFeatures(ids, fdistNgram, sentTokens, k, n):
    #     fdistPos = ngramFreq(title, labels, 0)
    #     fdistNeg = ngramFreq(title, labels, 1)
    #     fdistAll = ngramFreq(title)
    df = pd.DataFrame()
    df['id'] = ids

    ngrams = fdistNgram.most_common(k)
    for (ngram, freq) in ngrams:
        feat = []
        ngramS = ' '.join(list(ngram))
        for sentToken in sentTokens:
            feat.append(int(isNgramExist(ngramS, sentToken)))
        df[str(n) + '-gram_' + '_'.join(list(ngram))] = feat

    return df


def findNgram(ngram, sentToken):
    tokens = generateTokens(ngram)
    pos = position_of_ngram(tuple(tokens), sentToken)
    return pos


def isNgramExist(ngram, sentToken):
    pos = findNgram(ngram, sentToken)
    return pos is not None


def countAverageWordLength(title):
    feat = []
    for text in title:
        tokens = nltk.word_tokenize(text)
        avg = sum(map(len, tokens))/len(tokens) if len(tokens) > 0 else 0
        feat.append(avg)
    return feat


def lengthLongestWord(title):
    feat = []
    for text in title:
        tokens = nltk.word_tokenize(text)
        if len(tokens) > 0:
            length = len(sorted(tokens, key=lambda x: len(x))[-1])
        else:
            length = 0
        feat.append(length)
    return feat


def countWordLength(title):
    feat = []
    for text in title:
        var = re.split(r'\W+', text)
        wordlen = len(var)-1
        feat.append(wordlen)
    return feat


def generateReadability(title):
    feat = []
    #df = pd.DataFrame()

    for text in title:
        fk = FleschKincaid(text)
        feat.append(fk.min_age)
    #df['readability_min_age'] = feat

    return feat


def calculateOverAllSimilarity(title, targetTitles):
    feat = []
    for i in range(len(title)):
        text = title[i]
        target = targetTitles[i]
#         print(text, target)
        simi_value = similarity(text, target, True)
        feat.append(simi_value)
    return feat


def whetherStart5W1H(title):
    feat = []
    for text in title:
        if re.search('^what|why|when|who|which|how', text, re.I) == None:
            feat.append(0)
        else:
            feat.append(1)
    return feat


def whetherStartNumber(title):
    feat = []
    for text in title:
        if re.search('^[0-9]+', text, flags=0) == None:
            feat.append(0)
        else:
            feat.append(1)
    return feat


def ratioStopWords(title):
    feat = []
    stopworddic = set(stopwords.words('english'))
    for text in title:
        var = re.split(r'\W+', text)
        a = len(var)
        var = [i for i in var if i in stopworddic]
        b = len(var)
        c = 1.0*b / a
        feat.append(c)
    return feat


def similarity_text(title, data_df):
    feat = []
    for ind in range(len(title)):
        postText = title[ind]
        targetTitle = data_df["targetTitle"].values[ind]
        similarity = similar_text(postText, targetTitle)
        feat.append(similarity)
    return feat


def calculateOverAllSimilarity(title, targetTitles):
    feat = []
    for i in range(len(title)):
        text = title[i]
        target = targetTitles[i]
#         print(text, target)
        simi_value = similarity(text, target, True)
        feat.append(simi_value)
    return feat


def whetherStart5W1H(title):
    feat = []
    for text in title:
        if re.search('^what|why|when|who|which|how', text, re.I) == None:
            feat.append(0)
        else:
            feat.append(1)
    return feat


def whetherStartNumber(title):
    feat = []
    for text in title:
        if re.search('^[0-9]+', text, flags=0) == None:
            feat.append(0)
        else:
            feat.append(1)
    return feat


def ratioStopWords(title):
    feat = []
    stopworddic = set(stopwords.words('english'))
    for text in title:
        var = re.split(r'\W+', text)
        a = len(var)
        var = [i for i in var if i in stopworddic]
        b = len(var)
        c = 1.0*b / a
        feat.append(c)
    return feat


def similarity_Paragraphs(title, data_df):
    feat = []
    for ind in range(len(title)):
        postText = title[ind]
        targetParagraphs = data_df["targetParagraphs"].values[ind]
        target = " ".join(targetParagraphs)
        similarity = similar_text(postText, target)
        feat.append(similarity)
    return feat


def matchKeywords(title, data_df):
    feat = []
    featureV = 0

    for ind in range(len(title)):
        text = title[ind]
        s = data_df["targetKeywords"].values[ind].split(',')
        for item in s:
            it = item.rstrip("\\")
            it = re.sub('\*|^|$|\?|\+|\.|\||\[|\]|\{|\}|\(|\)', '', it)
            try:
                match = re.search(it, text, re.I)
            except:
                # print(it)
                # print(text)
                match = re.search(it, text, re.I)
            if match == None:
                pass
            else:
                featureV += 1
        feat.append(featureV)
        featureV = 0
    return feat


def generateAllNgramFeaturesDF(ids, POSs, sentTokens, k):
    # loading dictionary
    with open(os.path.join(os.getcwd(), 'nlp_model/df_pos_bigram_50.pkl'), 'rb') as f:
        patterns_pos_2 = pickle.load(f)
    with open(os.path.join(os.getcwd(), 'nlp_model/df_pos_bigram_50_name.pkl'), 'rb') as f:
        patternNames_pos_2 = pickle.load(f)
    with open(os.path.join(os.getcwd(), 'nlp_model/df_pos_trigram_50.pkl'), 'rb') as f:
        patterns_pos_3 = pickle.load(f)
    with open(os.path.join(os.getcwd(), 'nlp_model/df_pos_trigram_50_name.pkl'), 'rb') as f:
        patternNames_pos_3 = pickle.load(f)
    with open(os.path.join(os.getcwd(), 'nlp_model/df_unigram_50.pkl'), 'rb') as f:
        fdist_word_1 = pickle.load(f)
    with open(os.path.join(os.getcwd(), 'nlp_model/df_bigram_50.pkl'), 'rb') as f:
        fdist_word_2 = pickle.load(f)
    with open(os.path.join(os.getcwd(), 'nlp_model/df_trigram_50.pkl'), 'rb') as f:
        fdist_word_3 = pickle.load(f)

    # unigram POS
    tags = list(tagdict.keys())
    tags.remove('(')
    tags.remove(')')
    tags.remove(':')
    df_pos_unigram = generatePosUnigramFeatures(ids, POSs, tags, tags)

    # bigram POS
    df_pos_bigram_50 = generatePosNgramsFeatures(
        ids, POSs, patterns_pos_2, patternNames_pos_2, k, 2)
    # trigram POS
    df_pos_trigram_50 = generatePosNgramsFeatures(
        ids, POSs, patterns_pos_3, patternNames_pos_3, k, 3)
    # unigram WORD
    df_unigram_50 = generateNgramFeatures(ids, fdist_word_1, sentTokens, k, 1)
    # bigram WORD
    df_bigram_50 = generateNgramFeatures(ids, fdist_word_2, sentTokens, k, 2)
    # trigram WORD
    df_trigram_50 = generateNgramFeatures(ids, fdist_word_3, sentTokens, k, 3)

    return df_pos_unigram, df_pos_bigram_50, df_pos_trigram_50, df_unigram_50, df_bigram_50, df_trigram_50


def generateFeatureDF(POSs, title, sentTokens):
    df = pd.DataFrame()

    # Sentiment score feature
    # print('Generating SENTIMENT SCORE FEATURE')
    feat_Sentiment_HIGH = []
    feat_Sentiment = []
    sid = SentimentIntensityAnalyzer()
    for text in title:
        feat_Sentiment_HIGH.append(
            int(math.fabs(sid.polarity_scores(text)['compound']) > 0.8))
        feat_Sentiment.append(math.fabs(sid.polarity_scores(text)['compound']))

    df['sentiment_score_high'] = feat_Sentiment_HIGH
    df['sentiment_score'] = feat_Sentiment
    # END of sentiment popularity score feature

    # IF POS Patterns Exist Feature
    # print('Generating IF POS PATTERN FEATURES')
    patterns = {'EXIST_POS_NUMBER_NP_THAT': r"""CHUNK: {<CD><JJ.*>?<N.*><WDT><VB.*|VB>}""",
                'EXIST_POS_NUMBER_NP_VB': r"""CHUNK: {<CD><JJ.*>?<N.*><PRP.*><VB.*|VB>}"""}

    feat_EXIST_POS = {}
    for key in patterns:
        feat_EXIST_POS[key] = []

    for sentPOSs in POSs:
        for key in patterns:
            check = isPosPattern(patterns[key], sentPOSs, 'CHUNK')
            feat_EXIST_POS[key].append(int(check))

    for key in patterns:
        df[key] = feat_EXIST_POS[key]
    # End of IF POS Pattern Exist Features

    # COUNT POS Patterns
    # print('Generating COUNT OF POS PATTERN FEATURES')
    patterns = {'POS_pattern_COUNT_NUM_SHORTTEN': r"""CHUNK: {<''><VBP|MD>}""",
                'POS_pattern_COUNT_DT': r"""CHUNK: {<DT>}""",
                'POS_pattern_COUNT_WRB': r"""CHUNK: {<WRB>}""",
                'POS_pattern_COUNT_PRP_Dollar': r"""CHUNK: {<PRP$>}""",
                'POS_pattern_COUNT_MD': r"""CHUNK: {<MD>}""",
                'POS_pattern_COUNT_WDT': r"""CHUNK: {<WDT>}""",
                'POS_pattern_COUNT_PRP': r"""CHUNK: {<PRP>}""",
                'POS_pattern_COUNT_RB': r"""CHUNK: {<RB>}""",
                'POS_pattern_COUNT_WRB': r"""CHUNK: {<WRB>}""",
                'POS_pattern_COUNT_WP': r"""CHUNK: {<WP>}"""}

    feat_COUNT_POS = {}
    for key in patterns:
        feat_COUNT_POS[key] = []

    for sentPOSs in POSs:
        for key in patterns:
            count = countPosPattern(patterns[key], sentPOSs, 'CHUNK')
            feat_COUNT_POS[key].append(count)

    for key in patterns:
        df[key] = feat_COUNT_POS[key]
    # End COUNT POS Patterns Features

    # COUNT OF POS Pattern with Conditions
    # print('Generating COUNT OF POS Pattern with Conditions')
    feat_POS_pattern4 = []
    for sentPOSs in POSs:
        found = findPosPatternIncludeWord(r"""CHUNK: {<DT><NN.*>}""", sentPOSs, 'CHUNK',
                                          ['this', 'these', 'This', 'These'])
        count = len(found)
        feat_POS_pattern4.append(count)
    df['POS_pattern_COUNT_this-these_NN'] = feat_POS_pattern4
    # End of Count of POS pattern with Conditions

    # IF DICT Exist Feature
    # print('Generating TOKEN DICTIONARY FEATURE')
    ngramsToCheck = ['@', 'http', '?', '#', '!',
                     '! ?', '. . .', '* * *', '! !', '! ! !']
    ngramsNames = ['AT', 'WEB', 'QM', 'OC', "EX",
                   'EX-QM', 'TRIPLE-DOT', 'TRIPLE-AS', 'DOUBLE-EX', 'TRIPLE-EX']
    for i in range(len(ngramsToCheck)):
        ngramToCheck = ngramsToCheck[i]
        feat_exist_ = []
        for sentToken in sentTokens:
            if (isNgramExist(ngramToCheck, sentToken)):
                feat_exist_.append(1)
            else:
                feat_exist_.append(0)
        df['CONTAINS_' + ngramsNames[i]] = feat_exist_
     # END OF IF DICT Exist Feature

    # print('Generating LENGTH/NUMERIC FEATURES')
    # number of tokens
    feat_NUM_TOKENS = []
    for sentPOSs in POSs:
        feat_NUM_TOKENS.append(len(sentPOSs))
    df['NUM_TOKENS'] = feat_NUM_TOKENS

    # average word length
    averageWordLengthFeat = np.array(countAverageWordLength(title))
    df['AVG_WORD_LENGTH'] = averageWordLengthFeat

    # length of longest word
    lengthLongest = lengthLongestWord(title)
    df['LEN_LONGEST_WORD'] = lengthLongest

    # match between postText and keywords
    # featureV = matchKeywords(title, data_df)
    # df['count_match_keywords'] = featureV

    # similartity between postText and targetTitle
    # similarity = similarity_text(title, data_df)
    # df['similarity_post_Title'] = similarity

    # similarity between postText and targetParagraphs
    # similarity2 = similarity_Paragraphs(title, data_df)
    # df['similarity_paragraphs'] = similarity2

    # readability
    # readability = generateReadability(ids,[" ".join(a) for a in data_df['targetParagraphs']])
#     print(len(readability))
#     print(df.shape)
    # df['readability']=readability

    # readability2 = generateReadability(title)
    # df['readability2']=readability2

    # count word length
    wordlen = countWordLength(title)
    df['wordLength'] = wordlen

    # wordlenTarget = countWordLength([" ".join(a) for a in data_df['targetParagraphs']])
    # df['targetWordLength'] = wordlenTarget

    # ratio of stop wprds in postText
    ratioStop = ratioStopWords(title)
    df['ratioStop'] = ratioStop

    # whether postText start with number
    jd = whetherStartNumber(title)
    df['startNumber'] = jd

    # whether start with 5W1H
    whether5W1H = whetherStart5W1H(title)
    df['whether5W1H'] = whether5W1H

    # match the internet slangs
    countSlangs = matchInternetSlangs(title)
    df['countSlangs'] = countSlangs

    # #calculate the value of formality of targetParagraph
    # formality = calculateFormality(title, data_df)
    # df['formality'] = formality

    # calculate overall similarity of postText and targetTitle
#     overallSimilarity = calculateOverAllSimilarity(title, data_df['targetTitle'].get_values())
#     df['overallSimilarity'] = overallSimilarity
    return df


def matchInternetSlangs(allPostText):
    feat = []
    file = open('./nlp_model/internetSlangList.txt', 'r')
    for text in allPostText:
        featureV = 0
        for line in file.readlines():
            str = line
            str = ' '+str[0:len(str)-1]+' '
            str = str.replace('\\', '')
            str = re.sub('\*|^|$|\?|\+|\.|\||\[|\]|\{|\}|\(|\)', '', str)
            if re.search(str, text, re.I) == None:
                featureV += 1
        feat.append(featureV)
    return feat


def factor_NLP(title):
    POSs = np.array(list(map(generatePosSequence, title)))
    title = np.array(title)
    sentTokens = np.array(list(map(generateTokens, title)))
    tokensDist, tokens = generateFdist(sentTokens)
    return POSs, tokensDist, sentTokens, tokens


def generateFeatures(titles, contents=None):  # return Pandas dataframe
    ids = range(len(titles))
    titles = np.array(titles)
    POSs, tokensDist, sentTokens, tokens = factor_NLP(titles)
    X = generateFeatureDF(POSs, titles, sentTokens)
    df_pos_unigram, df_pos_bigram_50,\
        df_pos_trigram_50, df_unigram_50,\
        df_bigram_50, df_trigram_50 = generateAllNgramFeaturesDF(
            ids, POSs, sentTokens, 50)
    X = pd.concat([X, df_pos_unigram, df_pos_bigram_50,
                   df_pos_trigram_50, df_unigram_50,
                   df_bigram_50, df_trigram_50], axis=1).drop(['id'], axis=1)
    # print('Generated features with shape {}'.format(X.shape))
    return X


def make_unbalance(Xs, y, target_class, ratio):
    i_target = np.where(np.array(y) == target_class)[0]
    np.random.shuffle(i_target)
    i_target = i_target[:int(len(i_target)*ratio)]
    i_else = np.where(y != target_class)[0]
    i = np.concatenate((i_target, i_else))
    return [x[i] for x in Xs], y[i]


def process_generated_data(file, path_filter_bot=None):
    with open(file, 'r') as f:
        comp = np.array(f.readlines())
    comp = np.array([c.strip().replace('\n', '') for c in comp[comp != '\n']])
    comp, _ = remove_duplicates(comp)

    if path_filter_bot:
        with open(path_filter_bot, 'r') as f:
            filter_bots = f.readlines()
        filter_bots = [c.replace('\n', '')
                       for c in filter_bots]  # this is important
        to_keep = [i for i in range(len(comp)) if comp[i] not in filter_bots]
        comp = comp[to_keep]

    return comp
