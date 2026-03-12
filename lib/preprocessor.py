from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import gensim.downloader as api
from nltk.util import ngrams
import contractions
import numpy as np
import string
import nltk

# from typing import Literal


nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")
nltk.download("punkt_tab")
nltk.download("stopwords")
# python3 -->
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

class Preprocessor:
    # spell = SpellChecker()
    stemmer      = PorterStemmer()
    stemmer_plus = SnowballStemmer(language="english")
    lemmatizer   = WordNetLemmatizer()

    stop_words   = set(stopwords.words("english"))

    punct_translator = str.maketrans("", "", string.punctuation)
    digit_translator = str.maketrans("", "", string.digits)

    @staticmethod
    def get_wordnet_pos(tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    @classmethod
    def clean(cls, tweets: list, show_trans):
        cleaned_tweets = []

        for text in tweets:
            text = contractions.fix(text)
            text = text.lower()
            text = text.translate(cls.punct_translator)
            text = text.translate(cls.digit_translator)
            text = text.strip()  # remove leading/trailing spaces
            text = " ".join(text.split())  # remove duplicate spaces

            cleaned_tweets.append(text)

        if show_trans:
            print("after cleaning")
            print(cleaned_tweets[:20])

        return cleaned_tweets

    @classmethod
    def tokenize(cls, cleaned_tweets: list, n_grams, show_trans):
        tokenized_tweets = []

        for text in cleaned_tweets:
            tokens = word_tokenize(text)
            # t_grams = list(ngrams(tokens, n_grams))
            tokenized_tweets.append(tokens)

        if show_trans:
            print("after tokenization")
            print(tokenized_tweets[:20])
        return tokenized_tweets

    @classmethod
    def stopwords(cls, tokenized_sentences: list, show_trans):
        # The use of 'cls.stop_words' correctly references the class attribute.
        filtered_tokens = []

        for tokens in tokenized_sentences:
            # This list comprehension is fast because cls.stop_words is a set.
            filtered = [word for word in tokens if word not in cls.stop_words]
            filtered_tokens.append(filtered)

        if show_trans:
            print("after stop words")
            print(filtered_tokens[:20])

        return filtered_tokens

    @classmethod
    def lemmatize(cls, tokenized_sentences: list, show_trans):
        standardized_tokens = []

        for tokens in tokenized_sentences:
            lemmas = []
            pos_tags = nltk.pos_tag(tokens)

            for word, tag in pos_tags:
                wn_tag = cls.get_wordnet_pos(tag)
                lemma = cls.lemmatizer.lemmatize(word, wn_tag)
                lemmas.append(lemma)

            standardized_tokens.append(lemmas)

        if show_trans:
            print("after lemmatization")
            print(standardized_tokens[:20])

        return standardized_tokens

    # @classmethod
    # def spelling(cls, tokenized_tweets: list):
    #     corrected_tokens = []

    #     for tokens in tokenized_tweets:
    #         corrected_text = []
    #         for word in tokens:
    #             corrected_word = cls.spell.correction(word)
    #             corrected_tokens.append(
    #                 corrected_word if corrected_word is not None else ""
    #             )

    #         corrected_tokens.append(corrected_text)

    #     return corrected_tokens

    @classmethod
    def stemming(cls, tokenized_tweets: list, show_trans):
        stems_tokens = []

        for tokens in tokenized_tweets:
            stems = [cls.stemmer.stem(w) for w in tokens]
            stems_tokens.append(stems)

        if show_trans:
            print("after stemming")
            print(stems_tokens[:20])
        return stems_tokens

    @classmethod
    def stemming_plus(cls, tokenized_tweets: list, show_trans):
        stems_tokens = []

        for tokens in tokenized_tweets:
            stems = [cls.stemmer_plus.stem(w) for w in tokens]
            stems_tokens.append(stems)

        if show_trans:
            print("after stemming")
            print(stems_tokens[:20])

        return stems_tokens

    @staticmethod
    def identity(x):
        return x

    @classmethod
    def wordcounts(self, standardized_tokens: list, vectorization):
        self.vectorizer = CountVectorizer(
            binary=(vectorization == "binary"),
            tokenizer=self.identity,
            preprocessor=self.identity,
            token_pattern=None,
        )
        vector = self.vectorizer.fit_transform(standardized_tokens)
        return np.array(vector.toarray())

    @classmethod
    def tf_idf(self, standardized_tokens: list):
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.identity, preprocessor=self.identity, token_pattern=None
        )
        vector = self.vectorizer.fit_transform(standardized_tokens)
        return np.array(vector.toarray())

    @classmethod
    def sentence_embedding(self, sentence, model):
        vectors = [model[w] for w in sentence if w in model]
        if not vectors:
            return np.zeros(25)
        return np.mean(vectors, axis=0)

    @classmethod
    def word2vec(self, tokenized_sentences: list):
        model = api.load("glove-twitter-25")

        X = np.array(
            [
                self.sentence_embedding(sentence, model)
                for sentence in tokenized_sentences
            ]
        )
        return X

    @classmethod
    def vectorize(self, standardized_tokens: list, vectorization="tf-idf"):
        if vectorization in ("bow", "binary"):
            return self.wordcounts(standardized_tokens, vectorization)
        if vectorization == "tf-idf":
            return self.tf_idf(standardized_tokens)

        return self.word2vec(standardized_tokens)


    @classmethod
    def processing_methods(self, key, tokenized_sentences, show_trans):

        methods_dict = {
            'lemmatize': self.lemmatize,
            'stem': self.stemming,
            'stem+': self.stemming_plus,
        }
        return methods_dict[key](tokenized_sentences, show_trans)
    
    @classmethod
    def process(self, raws: list, processing_params: dict, n_grams=1, show_trans=True):

        cleaned = self.clean(raws, show_trans)
        tokenized = self.tokenize(cleaned, n_grams, show_trans)

        if  'stopwords' in processing_params and processing_params['stopwords']:
            tokenized = self.stopwords(tokenized, show_trans)

        if 'method' in processing_params and processing_params['method']:
            tokenized = self.processing_methods(processing_params['method'], tokenized, show_trans)

        return self.vectorize(tokenized, processing_params['vectorization'])

