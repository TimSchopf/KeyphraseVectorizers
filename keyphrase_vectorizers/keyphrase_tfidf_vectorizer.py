"""
.. _spaCy pipeline: https://spacy.io/models
.. _stopwords available in NLTK: https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/stopwords.zip
.. _POS-tags: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
.. _regex pattern: https://docs.python.org/3/library/re.html#regular-expression-syntax
.. _spaCy part-of-speech tags: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
"""

import warnings

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils.validation import FLOAT_DTYPES

from keyphrase_vectorizers.keyphrase_count_vectorizer import KeyphraseCountVectorizer


class KeyphraseTfidfVectorizer(KeyphraseCountVectorizer):
    """
    KeyphraseTfidfVectorizer

    KeyphraseTfidfVectorizer converts a collection of text documents to a normalized tf or tf-idf document-token matrix.
    The tokens are keyphrases that are extracted from the text documents based on their part-of-speech tags.
    The matrix rows indicate the documents and columns indicate the unique keyphrases.
    Each cell represents the tf or tf-idf value, depending on the parameter settings.
    The part-of-speech pattern of keyphrases can be defined by the ``pos_pattern`` parameter.
    By default, keyphrases are extracted, that have 0 or more adjectives, followed by 1 or more nouns.
    A list of extracted keyphrases matching the defined part-of-speech pattern can be returned after fitting via :class:`get_feature_names_out()`.

    Attention:
        If the vectorizer is used for languages other than English, the ``spacy_pipeline`` and ``stop_words`` parameters
        must be customized accordingly.
        Additionally, the ``pos_pattern`` parameter has to be customized as the `spaCy part-of-speech tags`_  differ between languages.
        Without customizing, the words will be tagged with wrong part-of-speech tags and no stopwords will be considered.

    Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency.
    This is a common term weighting scheme in information retrieval,
    that has also found good use in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document
    is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training corpus.

    The formula that is used to compute the tf-idf for a term t of a document d in a document set is
    tf-idf(t, d) = tf(t, d) * idf(t), and the idf is computed as idf(t) = log [ n / df(t) ] + 1 (if ``smooth_idf=False``),
    where n is the total number of documents in the document set and df(t) is the document frequency of t;
    the document frequency is the number of documents in the document set that contain the term t.
    The effect of adding "1" to the idf in the equation above is that terms with zero idf, i.e., terms
    that occur in all documents in a training set, will not be entirely ignored.
    (Note that the idf formula above differs from the standard textbook
    notation that defines the idf as idf(t) = log [ n / (df(t) + 1) ]).

    If ``smooth_idf=True`` (the default), the constant "1" is added to the numerator and denominator of the idf as
    if an extra document was seen containing every term in the collection exactly once, which prevents
    zero divisions: idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1.

    Furthermore, the formulas used to compute tf and idf depend on parameter settings that correspond to
    the SMART notation used in IR as follows:

    Tf is "n" (natural) by default, "l" (logarithmic) when ``sublinear_tf=True``.
    Idf is "t" when use_idf is given, "n" (none) otherwise.
    Normalization is "c" (cosine) when ``norm='l2'``, "n" (none) when ``norm=None``.

    Parameters
    ----------
    spacy_pipeline : str, default='en_core_web_sm'
            The name of the `spaCy pipeline`_, used to tag the parts-of-speech in the text. Standard is the 'en' pipeline.

    pos_pattern :  str, default='<J.*>*<N.*>+'
        The `regex pattern`_ of `POS-tags`_ used to extract a sequence of POS-tagged tokens from the text.
        Standard is to only select keyphrases that have 0 or more adjectives, followed by 1 or more nouns.

    stop_words : str, default='english'
            Language of stopwords to remove from the document, e.g.'english.
            Supported options are `stopwords available in NLTK`_.
            Removes unwanted stopwords from keyphrases if 'stop_words' is not None.

    lowercase : bool, default=True
        Whether the returned keyphrases should be converted to lowercase.

    binary : bool, default=False
        If True, all non zero counts are set to 1.
        This is useful for discrete probabilistic models that model binary events rather than integer counts.

    dtype : type, default=np.int64
        Type of the matrix returned by fit_transform() or transform().

    norm : {'l1', 'l2'}, default='l2'
        Each output row will have unit norm, either:
        - 'l2': Sum of squares of vector elements is 1. The cosine similarity between two vectors is their dot product when l2 norm has been applied.
        - 'l1': Sum of absolute values of vector elements is 1.

    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting. If False, idf(t) = 1.

    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : bool, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    """

    def __init__(self, spacy_pipeline='en_core_web_sm', pos_pattern='<J.*>*<N.*>+', stop_words='english',
                 lowercase=True, binary=False, dtype=np.float64, norm="l2", use_idf=True, smooth_idf=True,
                 sublinear_tf=False):

        self.spacy_pipeline = spacy_pipeline
        self.pos_pattern = pos_pattern
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.binary = binary
        self.dtype = dtype
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

        self._tfidf = TfidfTransformer(norm=self.norm, use_idf=self.use_idf, smooth_idf=self.smooth_idf,
                                       sublinear_tf=self.sublinear_tf)

        super().__init__(spacy_pipeline=self.spacy_pipeline, pos_pattern=self.pos_pattern, stop_words=self.stop_words,
                         lowercase=self.lowercase, binary=self.binary, dtype=self.dtype)

    def _check_params(self):
        """
        Validate dtype parameter.
        """

        if self.dtype not in FLOAT_DTYPES:
            warnings.warn(
                "Only {} 'dtype' should be used. {} 'dtype' will "
                "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
                UserWarning,
            )

    def fit(self, raw_documents):
        """Learn the keyphrases that match the defined part-of-speech pattern and idf from the list of raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings.

        Returns
        -------
        self : object
            Fitted vectorizer.
        """

        self._check_params()
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents):
        """
        Learn the keyphrases that match the defined part-of-speech pattern and idf from the list of raw documents.
        Then return document-keyphrase matrix.
        This is equivalent to fit followed by transform, but more efficiently implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings.

        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-keyphrase matrix.
        """

        self._check_params()
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents):
        """
        Transform documents to document-keyphrase matrix.
        Uses the keyphrases and document frequencies (df) learned by fit (or fit_transform).

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings.

        Returns
        -------
        X : sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-keyphrase matrix.
        """

        # triggers a parameter validation
        if not hasattr(self, 'keyphrases'):
            raise NotFittedError("Keyphrases not fitted.")

        X = super().transform(raw_documents)
        return self._tfidf.transform(X, copy=False)
