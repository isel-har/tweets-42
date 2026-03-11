# from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim.downloader as api
from nltk.corpus import wordnet
import contractions
import numpy as np
import string
import nltk


nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")
nltk.download("punkt_tab")
nltk.download("stopwords")
# python3 -->
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')


class Preprocessor:
    # spell = SpellChecker()
    stemmer = PorterStemmer()
    stemmer_plus = SnowballStemmer(language="english")
    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words("english"))

    punct_translator = str.maketrans("", "", string.punctuation)
    digit_translator = str.maketrans("", "", string.digits)

    # vectorization_methods = ['tf-idf', 'bow', 'binary','word2vec']

    # def __init__(self, vectorization='tf-idf', vector_size=100):
    #     self.vectorization = vectorization.lower()
    #     self.processed  = False
    #     self.vectorizer = None
    #     self.vector_size = vector_size

    # ________________________________________________________

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
    def clean(cls, tweets: list):
        cleaned_tweets = []

        for text in tweets:
            text = contractions.fix(text)
            text = text.lower()
            text = text.translate(cls.punct_translator)
            text = text.translate(cls.digit_translator)
            text = text.strip()  # remove leading/trailing spaces
            text = " ".join(text.split())  # remove duplicate spaces

            cleaned_tweets.append(text)

        return cleaned_tweets

    @classmethod
    def tokenize(cls, cleaned_tweets: list):
        tokenized_tweets = []

        for text in cleaned_tweets:
            tokens = word_tokenize(text)
            tokenized_tweets.append(tokens)

        return tokenized_tweets

    @classmethod
    def stopwords(cls, tokenized_sentences: list):
        # The use of 'cls.stop_words' correctly references the class attribute.
        filtered_tokens = []

        for tokens in tokenized_sentences:
            # This list comprehension is fast because cls.stop_words is a set.
            filtered = [word for word in tokens if word not in cls.stop_words]
            filtered_tokens.append(filtered)

        return filtered_tokens

    @classmethod
    def lemmatize(cls, tokenized_sentences: list):
        standardized_tokens = []

        for tokens in tokenized_sentences:
            lemmas = []
            pos_tags = nltk.pos_tag(tokens)

            for word, tag in pos_tags:
                wn_tag = cls.get_wordnet_pos(tag)
                lemma = cls.lemmatizer.lemmatize(word, wn_tag)
                lemmas.append(lemma)

            standardized_tokens.append(lemmas)

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
    def stemming(cls, tokenized_tweets: list):
        stems_tokens = []

        for tokens in tokenized_tweets:
            stems = [cls.stemmer.stem(w) for w in tokens]
            stems_tokens.append(stems)
        return stems_tokens

    @classmethod
    def stemming_plus(cls, tokenized_tweets: list):
        stems_tokens = []

        for tokens in tokenized_tweets:
            stems = [cls.stemmer_plus.stem(w) for w in tokens]
            stems_tokens.append(stems)
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
    def processing_methods(self, key, tokenized_sentences):

        methods_dict = {
            'lemmatize': self.lemmatize,
            'stem': self.stemming,
            'stem+': self.stemming_plus,
        }
        return methods_dict[key](tokenized_sentences)
    
    @classmethod
    def process(self, raw_sentences : list, processing_params: dict):

        cleaned_sentences = self.clean(raw_sentences)
        tokenized_sentences = self.tokenize(cleaned_sentences)

        if  'stopwords' in processing_params and processing_params['stopwords']:
            tokenized_sentences = self.stopwords(tokenized_sentences)

        if 'method' in processing_params and processing_params['method']:
            tokenized_sentences = self.processing_methods(processing_params['method'], tokenized_sentences)

        return self.vectorize(tokenized_sentences, processing_params['vectorization'])


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# class Cleaner()


class NLProcessor:


    def __init__(self):
        ...
    
    def fit(self, X):
        ...


    def transform(self, X):
        ...