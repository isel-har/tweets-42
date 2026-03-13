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


import pkg_resources
from symspellpy import SymSpell, Verbosity
# from typing import Literal


nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")
nltk.download("punkt_tab")
nltk.download("stopwords")
# python3 -->
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

class Preprocessor:

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


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
    def tokenize(cls, cleaned_tweets: list, show_trans):
        tokenized_tweets = []

        for text in cleaned_tweets:
            tokens = word_tokenize(text)
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

    @classmethod
    def spelling(cls, tokenized_tweets: list, show_trans: bool):
        corrected_tweets = []

        for tokens in tokenized_tweets:
            corrected_text = []
            for word in tokens:
                # SymSpell lookup returns a list of suggestions
                suggestions = cls.sym_spell.lookup(
                    word, Verbosity.CLOSEST, max_edit_distance=2
                )
                
                # If suggestions exist, take the top one; otherwise, keep original word
                corrected_word = suggestions[0].term if suggestions else word
                corrected_text.append(corrected_word)

            corrected_tweets.append(corrected_text)
            
        if show_trans:
            print(corrected_tweets)
            
        return corrected_tweets

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
        vectorizer = CountVectorizer(
            binary=(vectorization == "binary"),
            tokenizer=self.identity,
            preprocessor=self.identity,
            token_pattern=None,
        )
        vector = vectorizer.fit_transform(standardized_tokens)
        return np.array(vector.toarray()), vectorizer


    @classmethod
    def tf_idf(self, standardized_tokens: list):
        vectorizer = TfidfVectorizer(
            tokenizer=self.identity, preprocessor=self.identity, token_pattern=None
        )
        vector = vectorizer.fit_transform(standardized_tokens)

        return np.array(vector.toarray()), vectorizer


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
        return X, None

    @classmethod
    def vectorize(self, standardized_tokens: list, vectorization="tf-idf"):
        if vectorization in ("bow", "binary"):
            return self.wordcounts(standardized_tokens, vectorization)
        if vectorization == "tf-idf":
            return self.tf_idf(standardized_tokens)

        return self.word2vec(standardized_tokens)


    @classmethod
    def processing_methods(self, key, tokenized_sentences, show_trans):

        if key =='lem+misspelling':
            return self.spelling(
                self.lemmatize(tokenized_sentences, show_trans),
                show_trans
            )
        
        methods_dict = {
            'lemmatize': self.lemmatize,
            'stem': self.stemming,
            'stem+': self.stemming_plus,
            'misspelling':self.spelling,
        }
        return methods_dict[key](tokenized_sentences, show_trans)
    
    @classmethod
    def process(self, raws: list,
        processing_params: dict,
        misspellings=False,
        stopwords=False,
        show_trans=True,
        vectorizer=None):

        cleaned = self.clean(raws, show_trans)
        tokenized = self.tokenize(cleaned, show_trans)

        if stopwords:
            tokenized = self.stopwords(tokenized, show_trans)

        if 'method' in processing_params and processing_params['method']:
            tokenized = self.processing_methods(processing_params['method'], tokenized, show_trans)

        if vectorizer is None:
            return self.vectorize(tokenized, processing_params['vectorization'])

        vectors = np.array(vectorizer.transform(tokenized).toarray())
        return vectors, vectorizer
