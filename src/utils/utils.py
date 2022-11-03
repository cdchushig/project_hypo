import spacy
import nltk
import re
import pandas as pd

nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm')
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}

STOPWORDS_CLINICAL = ['disease', 'nos']


def lemma(comment):
    """
    Lemmatize comments using spacy lemmatizer.
    :param comment: a comment
    :return: lemmatized comment
    """
    lemmatized = nlp(comment)
    lemmatized_final = ' '.join([word.lemma_ for word in lemmatized if word.lemma_ != '\'s'])
    return lemmatized_final


def ascii_rm(comment):
    """
    Parse comments and keep only ascii characters
    :param comment: a comment
    :return: comment with only ascii characters
    """
    comment = comment.encode('ascii', errors='ignore')
    return comment


def clean_all(df, col_name):
    """
    Combine all functions used to clean and lemmatize the comments.
    :param df: data frame with comments
    :param col_name: column name in data frame containing comments
    :return: data frame with comments column lemmatized
    """

    # encode for only ascii characters
    # df[col_name] = df[col_name].map(ascii_rm)

    # lowercase texts
    df[col_name] = df[col_name].map(lambda x: x.lower())

    # lemmatize words
    df[col_name] = df[col_name].astype(str).map(lemma)

    # remove punctuation
    df[col_name] = df[col_name].map(remove_punctuations)

    # remove stopwords
    df[col_name] = df[col_name].apply(remove_stopwords)
    df[col_name] = df[col_name].apply(change_diabetes)

    # filter only english comments/non blank comments
    # df['language'] = df[col_name].map(identify_language)
    # df = df.loc[df['language'] == 'english']
    # df = df.drop('language', axis=1)
    # df = df[df[col_name] != ""]

    return df


def change_diabetes(text):
    text = re.sub(r'(type I diabetes mellitus)', 't1dm', text)
    text = re.sub(r'(chronic kidney)', 'ckd', text)
    text = re.sub(r'(proliferative diabetic retinopathy)', 'pdr', text)
    text = re.sub(r'(coronary artery)', 'cad', text)
    text = re.sub(r'(erectile dysfunction)', 'ed', text)
    text = re.sub(r'(diabetic peripheral neuropathy)', 'dpn', text)
    text = re.sub(r'(diabetic neuropathy)', 'dn', text)
    text = re.sub(r'(sleep apnea)', 'sa', text)
    text = re.sub(r'(gastroesophageal reflux)', 'gr', text)
    text = re.sub(r'(hear loss)', 'hl', text)
    return text


def remove_stopwords(text):
    text = re.split("\W+", text)
    text = [word for word in text if word not in STOPWORDS_CLINICAL]
    text = ' '.join(text)
    return text


def identify_language(text):
    """
    Determines what language the comment is written in and filters only English comments.
    :param text: comment
    :return: language of comment
    """
    words = set(nltk.wordpunct_tokenize(text.lower()))
    return max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key = lambda x: x[1])[0]


def remove_punctuations(comment):
    """
    Remove punctuations from comments.
    :param comment: a comment
    :return: comment without punctuations
    """
    regex = re.compile('[' + re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~')+'0-9\\r\\t\\n]')
    # regex = re.compile("[^0-9a-zA-Z,]+")
    nopunct = regex.sub(" ", comment)
    nopunct_words = nopunct.split(' ')
    filter_words = [word.strip() for word in nopunct_words if word != '']
    words = ' '.join(filter_words)
    return words
