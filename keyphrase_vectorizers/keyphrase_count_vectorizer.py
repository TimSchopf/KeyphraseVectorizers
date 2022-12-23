"""
.. _spaCy pipeline: https://spacy.io/models
.. _stopwords available in NLTK: https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/stopwords.zip
.. _POS-tags: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
.. _regex pattern: https://docs.python.org/3/library/re.html#regular-expression-syntax
.. _spaCy part-of-speech tags: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
.. _spaCy pipeline components: https://spacy.io/usage/processing-pipelines#built-in
"""

import warnings
from typing import List, Union

import numpy as np
import psutil
import spacy
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.deprecation import deprecated

from keyphrase_vectorizers.keyphrase_vectorizer_mixin import _KeyphraseVectorizerMixin


class KeyphraseCountVectorizer(_KeyphraseVectorizerMixin, BaseEstimator):
    """
    KeyphraseCountVectorizer

    KeyphraseCountVectorizer converts a collection of text documents to a matrix of document-token counts.
    The tokens are keyphrases that are extracted from the text documents based on their part-of-speech tags.
    The matrix rows indicate the documents and columns indicate the unique keyphrases. Each cell represents the count.
    The part-of-speech pattern of keyphrases can be defined by the ``pos_pattern`` parameter.
    By default, keyphrases are extracted, that have 0 or more adjectives, followed by 1 or more nouns.
    A list of extracted keyphrases matching the defined part-of-speech pattern can be returned after fitting via :class:`get_feature_names_out()`.

    Attention:
        If the vectorizer is used for languages other than English, the ``spacy_pipeline`` and ``stop_words`` parameters
        must be customized accordingly.
        Additionally, the ``pos_pattern`` parameter has to be customized as the `spaCy part-of-speech tags`_  differ between languages.
        Without customizing, the words will be tagged with wrong part-of-speech tags and no stopwords will be considered.

    Parameters
    ----------
    spacy_pipeline : Union[str, spacy.Language], default='en_core_web_sm'
            A spacy.Language object or the name of the `spaCy pipeline`_, used to tag the parts-of-speech in the text. Standard is the 'en' pipeline.

    pos_pattern :  str, default='<J.*>*<N.*>+'
        The `regex pattern`_ of `POS-tags`_ used to extract a sequence of POS-tagged tokens from the text.
        Standard is to only select keyphrases that have 0 or more adjectives, followed by 1 or more nouns.

    stop_words : Union[str, List[str]], default='english'
            Language of stopwords to remove from the document, e.g. 'english'.
            Supported options are `stopwords available in NLTK`_.
            Removes unwanted stopwords from keyphrases if 'stop_words' is not None.
            If given a list of custom stopwords, removes them instead.

    lowercase : bool, default=True
        Whether the returned keyphrases should be converted to lowercase.

    workers : int, default=1
            How many workers to use for spaCy part-of-speech tagging.
            If set to -1, use all available worker threads of the machine.
            SpaCy uses the specified number of cores to tag documents with part-of-speech.
            Depending on the platform, starting many processes with multiprocessing can add a lot of overhead.
            In particular, the default start method spawn used in macOS/OS X (as of Python 3.8) and in Windows can be slow.
            Therefore, carefully consider whether this option is really necessary.

    spacy_exclude : List[str], default=None
            A list of `spaCy pipeline components`_ that should be excluded during the POS-tagging.
            Removing not needed pipeline components can sometimes make a big difference and improve loading and inference speed.

    custom_pos_tagger: callable, default=None
            A callable function which expects a list of strings in a 'raw_documents' parameter and returns a list of (word token, POS-tag) tuples.
            If this parameter is not None, the custom tagger function is used to tag words with parts-of-speech, while the spaCy pipeline is ignored.

    max_df : int, default=None
        During fitting ignore keyphrases that have a document frequency strictly higher than the given threshold.

    min_df : int, default=None
        During fitting ignore keyphrases that have a document frequency strictly lower than the given threshold.
        This value is also called cut-off in the literature.

    binary : bool, default=False
        If True, all non zero counts are set to 1.
        This is useful for discrete probabilistic models that model binary events rather than integer counts.

    dtype : type, default=np.int64
        Type of the matrix returned by fit_transform() or transform().
    """

    def __init__(self, spacy_pipeline: Union[str, spacy.Language] = 'en_core_web_sm', pos_pattern: str = '<J.*>*<N.*>+',
                 stop_words: Union[str, List[str]] = 'english', lowercase: bool = True, workers: int = 1,
                 spacy_exclude: List[str] = None, custom_pos_tagger: callable = None,
                 max_df: int = None, min_df: int = None, binary: bool = False, dtype: np.dtype = np.int64):

        # triggers a parameter validation
        if not isinstance(min_df, int) and min_df is not None:
            raise ValueError(
                "'min_df' parameter must be of type int"
            )
        # triggers a parameter validation
        if min_df == 0:
            raise ValueError(
                "'min_df' parameter must be > 0"
            )

        # triggers a parameter validation
        if not isinstance(max_df, int) and max_df is not None:
            raise ValueError(
                "'max_df' parameter must be of type int"
            )

        # triggers a parameter validation
        if max_df == 0:
            raise ValueError(
                "'max_df' parameter must be > 0"
            )

        # triggers a parameter validation
        if max_df and min_df and max_df <= min_df:
            raise ValueError(
                "'max_df' must be > 'min_df'"
            )

        # triggers a parameter validation
        if not isinstance(workers, int):
            raise ValueError(
                "'workers' parameter must be of type int"
            )

        if (workers < -1) or (workers > psutil.cpu_count(logical=True)) or (workers == 0):
            raise ValueError(
                "'workers' parameter value cannot be 0 and must be between -1 and " + str(
                    psutil.cpu_count(logical=True))
            )

        self.spacy_pipeline = spacy_pipeline
        self.pos_pattern = pos_pattern
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.workers = workers
        self.spacy_exclude = spacy_exclude
        self.custom_pos_tagger = custom_pos_tagger
        self.max_df = max_df
        self.min_df = min_df
        self.binary = binary
        self.dtype = dtype

    def fit(self, raw_documents: List[str]) -> object:
        """
        Learn the keyphrases that match the defined part-of-speech pattern from the list of raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings.

        Returns
        -------
        self : object
            Fitted vectorizer.
        """

        self.keyphrases = self._get_pos_keyphrases(document_list=raw_documents,
                                                   stop_words=self.stop_words,
                                                   spacy_pipeline=self.spacy_pipeline,
                                                   pos_pattern=self.pos_pattern,
                                                   lowercase=self.lowercase, workers=self.workers,
                                                   spacy_exclude=self.spacy_exclude,
                                                   custom_pos_tagger=self.custom_pos_tagger)

        # remove keyphrases that have more than 8 words, as they are probably no real keyphrases
        # additionally this prevents memory issues during transformation to a document-keyphrase matrix
        self.keyphrases = [keyphrase for keyphrase in self.keyphrases if len(keyphrase.split()) <= 8]

        # compute document frequencies of keyphrases
        if self.max_df or self.min_df:
            document_keyphrase_counts = CountVectorizer(vocabulary=self.keyphrases, ngram_range=(
                min([len(keyphrase.split()) for keyphrase in self.keyphrases]),
                max([len(keyphrase.split()) for keyphrase in self.keyphrases])),
                                                        lowercase=self.lowercase, binary=self.binary,
                                                        dtype=self.dtype).transform(
                raw_documents=raw_documents).toarray()

            document_frequencies = self._document_frequency(document_keyphrase_counts)

        # remove keyphrases with document frequencies < min_df and document frequencies > max_df
        if self.max_df:
            self.keyphrases = [keyphrase for index, keyphrase in enumerate(self.keyphrases) if
                               (document_frequencies[index] <= self.max_df)]
        if self.min_df:
            self.keyphrases = [keyphrase for index, keyphrase in enumerate(self.keyphrases) if
                               (document_frequencies[index] >= self.min_df)]

        # set n-gram range to zero if no keyphrases could be extracted
        if self.keyphrases:
            self.max_n_gram_length = max([len(keyphrase.split()) for keyphrase in self.keyphrases])
            self.min_n_gram_length = min([len(keyphrase.split()) for keyphrase in self.keyphrases])
        else:
            raise ValueError(
                "Empty keyphrases. Perhaps the documents do not contain keyphrases that match the 'pos_pattern' parameter, only contain stop words, or you set the 'min_df'/'max_df' parameters too strict.")

        return self

    def fit_transform(self, raw_documents: List[str]) -> List[List[int]]:
        """
        Learn the keyphrases that match the defined part-of-speech pattern from the list of raw documents
        and return the document-keyphrase matrix.
        This is equivalent to fit followed by transform, but more efficiently implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings.

        Returns
        -------
        X : array of shape (n_samples, n_features)
            Document-keyphrase matrix.
        """

        # fit
        KeyphraseCountVectorizer.fit(self=self, raw_documents=raw_documents)

        # transform
        return CountVectorizer(vocabulary=self.keyphrases, ngram_range=(self.min_n_gram_length, self.max_n_gram_length),
                               lowercase=self.lowercase, binary=self.binary, dtype=self.dtype).fit_transform(
            raw_documents=raw_documents)

    def transform(self, raw_documents: List[str]) -> List[List[int]]:
        """
        Transform documents to document-keyphrase matrix.
        Extract token counts out of raw text documents using the keyphrases
        fitted with fit.

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-keyphrase matrix.
        """

        # triggers a parameter validation
        if not hasattr(self, 'keyphrases'):
            raise NotFittedError("Keyphrases not fitted.")

        return CountVectorizer(vocabulary=self.keyphrases, ngram_range=(self.min_n_gram_length, self.max_n_gram_length),
                               lowercase=self.lowercase, binary=self.binary, dtype=self.dtype).transform(
            raw_documents=raw_documents)

    def inverse_transform(self, X: List[List[int]]) -> List[List[str]]:
        """
        Return keyphrases per document with nonzero entries in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document-keyphrase matrix.

        Returns
        -------
        X_inv : list of arrays of shape (n_samples,)
            List of arrays of keyphrase.
        """

        # triggers a parameter validation
        if not hasattr(self, 'keyphrases'):
            raise NotFittedError("Keyphrases not fitted.")

        return CountVectorizer(vocabulary=self.keyphrases, ngram_range=(self.min_n_gram_length, self.max_n_gram_length),
                               lowercase=self.lowercase, binary=self.binary, dtype=self.dtype).inverse_transform(X=X)

    @deprecated(
        "get_feature_names() is deprecated in scikit-learn 1.0 and will be removed "
        "with scikit-learn 1.2. Please use get_feature_names_out() instead."
    )
    def get_feature_names(self) -> List[str]:
        """
        Array mapping from feature integer indices to feature name.

        Returns
        -------
        feature_names : list
            A list of fitted keyphrases.
        """

        # triggers a parameter validation
        if not hasattr(self, 'keyphrases'):
            raise NotFittedError("Keyphrases not fitted.")

        # raise DeprecationWarning when function is removed from scikit-learn
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return CountVectorizer(vocabulary=self.keyphrases,
                                       ngram_range=(self.min_n_gram_length, self.max_n_gram_length),
                                       lowercase=self.lowercase, binary=self.binary,
                                       dtype=self.dtype).get_feature_names()
        except AttributeError:
            raise DeprecationWarning("get_feature_names() is deprecated. Please use 'get_feature_names_out()' instead.")

    def get_feature_names_out(self) -> np.array(str):
        """
        Get fitted keyphrases for transformation.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed keyphrases.
        """

        # triggers a parameter validation
        if not hasattr(self, 'keyphrases'):
            raise NotFittedError("Keyphrases not fitted.")

        return CountVectorizer(vocabulary=self.keyphrases, ngram_range=(self.min_n_gram_length, self.max_n_gram_length),
                               lowercase=self.lowercase, binary=self.binary, dtype=self.dtype).get_feature_names_out()
