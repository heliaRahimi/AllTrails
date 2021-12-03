import itertools, string, operator, re, unicodedata, nltk
from collections import Counter

import pandas as pd
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from operator import itemgetter
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from bs4 import BeautifulSoup
import re
from textblob import TextBlob


def create_contraction_map():
    c_dict = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "I would",
        "i'd've": "I would have",
        "i'll": "I will",
        "i'll've": "I will have",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there had",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'alls": "you alls",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you you will",
        "you'll've": "you you will have",
        "you're": "you are",
        "you've": "you have",
    }
    return c_dict


c_dict = create_contraction_map()

c_re = re.compile("(%s)" % "|".join(c_dict.keys()))
add_stop = ["", " ", "say", "s", "u", "ap", "afp", "...", "n", "\\"]
stop_words = ENGLISH_STOP_WORDS.union(add_stop)
tokenizer = TweetTokenizer()
pattern = r"(?u)\b\w\w+\b"
duplicate_words = ["trail", "hike", "view", "walk", "fall"]
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer(language="english")

punc = list(set(string.punctuation))


def casual_tokenizer(
    text,
):  # Splits words on white spaces (leaves contractions intact) and splits out trailing punctuation
    tokens = tokenizer.tokenize(text)
    return tokens


# Function to replace the nltk pos tags with the corresponding wordnet pos tag to use the wordnet lemmatizer
def get_word_net_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def lemma_wordnet(tagged_text):
    final = []
    for word, tag in tagged_text:
        wordnet_tag = get_word_net_pos(tag)
        if wordnet_tag is None:
            final.append(stemmer.stem(lemmatizer.lemmatize(word)))
        else:
            final.append(stemmer.stem(lemmatizer.lemmatize(word, pos=wordnet_tag)))
    return final


def expandContractions(text, c_re=c_re):
    def replace(match):
        return c_dict[match.group(0)]

    return c_re.sub(replace, text)


def remove_html(text):
    soup = BeautifulSoup(text, "html5lib")
    tags_del = soup.get_text()
    uni = unicodedata.normalize("NFKD", tags_del)
    bracket_del = re.sub(r"\[.*?\]", "  ", uni)
    apostrphe = re.sub("’", "'", bracket_del)
    string = apostrphe.replace("\r", "  ")
    string = string.replace("\n", "  ")
    extra_space = re.sub(" +", " ", string)
    return extra_space


def process_text(text):
    soup = BeautifulSoup(text, "html.parser")
    tags_del = soup.get_text()
    no_html = re.sub("<[^>]*>", "", tags_del)
    tokenized = casual_tokenizer(no_html)
    lower = [item.lower() for item in tokenized]
    decontract = [expandContractions(item, c_re=c_re) for item in lower]
    tagged = nltk.pos_tag(decontract)
    lemma = lemma_wordnet(tagged)
    no_num = [re.sub("[0-9]+", "", each) for each in lemma]
    no_punc = [w for w in no_num if w not in punc]
    no_stop = [w for w in no_punc if w not in stop_words]
    final = [w for w in no_stop if w not in duplicate_words]
    return final


# Create a function to get the subjectivity
def subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Create a function to get the polarity
def polarity(text):
    return TextBlob(text).sentiment.polarity


def sent_Analysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"


# Calculate frequency of each word
def word_freq(clean_text_list, top_n):
    """
    Word Frequency
    """
    # flat = [item for sublist in clean_text_list for item in sublist]
    flat = [wrd for sub in clean_text_list for wrd in sub.split()]
    with_counts = Counter(flat)
    top = with_counts.most_common(top_n)
    word = [each[0] for each in top]
    num = [each[1] for each in top]
    return pd.DataFrame([word, num]).T


# Top 20 most frequent words for all the descrpitions


## Function for labelling the data to postive or negative reviews
