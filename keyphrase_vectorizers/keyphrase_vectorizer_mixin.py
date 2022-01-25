"""
.. _spaCy pipeline: https://spacy.io/models
.. _stopwords available in NLTK: https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/stopwords.zip
.. _POS-tags: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
.. _regex pattern: https://docs.python.org/3/library/re.html#regular-expression-syntax
"""

import spacy
from nltk import RegexpParser
from nltk.corpus import stopwords


class _KeyphraseVectorizerMixin():
    """
    _KeyphraseVectorizerMixin

    Provides common code for text vectorizers.
    """

    def _remove_suffixes(self, text: str, suffixes: list) -> str:
        """
        Removes pre-defined suffixes from a given text string.

        Parameters
        ----------
        text : str
            Text string where suffixes should be removed.

        suffixes : list
            List of strings that should be removed from the end of the text.

        Returns
        -------
        text : Text string with removed suffixes.
        """

        for suffix in suffixes:
            if text.lower().endswith(suffix.lower()):
                return text[:-len(suffix)].strip()
        return text

    def _remove_prefixes(self, text: str, prefixes: list) -> str:
        """
        Removes pre-defined prefixes from a given text string.

        Parameters
        ----------
        text : str
            Text string where prefixes should be removed.

        prefixes :  list
            List of strings that should be removed from the beginning of the text.

        Returns
        -------
        text : Text string with removed prefixes.
        """

        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                return text[len(prefix):].strip()
        return text

    def _get_pos_keyphrases(self, document: str, stop_words: str, spacy_pipeline: str, pos_pattern: str,
                            lowercase=True) -> list:
        """
        Select keyphrases with part-of-speech tagging from a text document.

        Parameters
        ----------
        document :  str
            Text document from which to extract the keyphrases.

        stop_words : str
            Language of stopwords to remove from the document, e.g.'english.
            Supported options are `stopwords available in NLTK`_.
            Removes unwanted stopwords from keyphrases if 'stop_words' is not None.

        spacy_pipeline : str
            The name of the `spaCy pipeline`_, used to tag the parts-of-speech in the text.

        pos_pattern : str
            The `regex pattern`_ of `POS-tags`_ used to extract a sequence of POS-tagged tokens from the text.

        lowercase : bool, default=True
            Whether the returned keyphrases should be converted to lowercase.

        Returns
        -------
        keyphrases : List of unique keyphrases of varying length, extracted from the text document with the defined 'pos_pattern'.
        """

        # triggers a parameter validation
        if not isinstance(document, str):
            raise ValueError(
                "Given document is not a string."
            )

        # triggers a parameter validation
        if not isinstance(stop_words, str):
            raise ValueError(
                "'stop_words' parameter needs to be a string. E.g. 'english'"
            )

        # triggers a parameter validation
        if not isinstance(spacy_pipeline, str):
            raise ValueError(
                "'spacy_pipeline' parameter needs to be a spaCy pipeline string. E.g. 'en_core_web_sm'"
            )

        # triggers a parameter validation
        if not isinstance(pos_pattern, str):
            raise ValueError(
                "'pos_pattern' parameter needs to be a regex string. E.g. '<J.*>*<N.*>+'"
            )

        stop_words_list = []
        if stop_words:
            stop_words_list = set(stopwords.words(stop_words))

        # add spaCy POS tags for document
        try:
            nlp = spacy.load(spacy_pipeline)
        except:
            spacy.cli.download(spacy_pipeline)
            nlp = spacy.load(spacy_pipeline)
        tagged_doc = nlp(document)
        tagged_pos_doc = []
        for sentence in tagged_doc.sents:
            pos_tagged_sentence = []
            for word in sentence:
                pos_tagged_sentence.append((word.text, word.tag_))
            tagged_pos_doc.append(pos_tagged_sentence)

        # extract keyphrases that match the NLTK RegexpParser filter
        cp = RegexpParser('CHUNK: {(' + pos_pattern + ')}')
        keyphrases = []
        prefix_list = [stop_word + ' ' for stop_word in stop_words_list]
        suffix_list = [' ' + stop_word for stop_word in stop_words_list]
        for sentence in tagged_pos_doc:
            tree = cp.parse(sentence)
            for subtree in tree.subtrees():
                if subtree.label() == 'CHUNK':
                    # join candidate keyphrase from single words
                    keyphrase = ' '.join([i[0] for i in subtree.leaves()])

                    # convert keyphrase to lowercase
                    if lowercase:
                        keyphrase = keyphrase.lower()

                    # remove stopword suffixes
                    keyphrase = self._remove_suffixes(keyphrase, suffix_list)

                    # remove stopword prefixes
                    keyphrase = self._remove_prefixes(keyphrase, prefix_list)

                    # remove whitespace from the beginning and end of keyphrases
                    keyphrase = keyphrase.strip()

                    # do not include single keywords that are actually stopwords
                    if keyphrase.lower() not in stop_words_list:
                        keyphrases.append(keyphrase)

        # remove potential empty keyphrases
        keyphrases = [keyphrase for keyphrase in keyphrases if keyphrase != '']

        return list(set(keyphrases))

    def _get_pos_keyphrases_of_multiple_docs(self, document_list: str, stop_words: str, spacy_pipeline: str,
                                             pos_pattern: str, lowercase=True) -> list:
        """
        Select keyphrases with part-of-speech tagging from a list of text documents.

        Parameters
        ----------
        document_list : list of str
            List of text documents from which to extract the keyphrases.

        stop_words : str
            Language of stopwords to remove from the document, e.g.'english.
            Supported options are `stopwords available in NLTK`_.
            Removes unwanted stopwords from keyphrases if 'stop_words' is not None.

        spacy_pipeline : str
            The name of the `spaCy pipeline`_, used to tag the parts-of-speech in the text.

        pos_pattern : str
            The `regex pattern`_ of `POS-tags`_ used to extract a sequence of POS-tagged tokens from the text.

        lowercase : bool, default=True
            Whether the returned keyphrases should be converted to lowercase.

        Returns
        -------
        keyphrases : List of unique keyphrases of varying length, extracted from the given text documents with the given 'pos_pattern'.
        """

        # triggers a parameter validation
        if isinstance(document_list, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        # triggers a parameter validation
        if not hasattr(document_list, '__iter__'):
            raise ValueError(
                "Iterable over raw text documents expected."
            )

        keyphrases = [
            self._get_pos_keyphrases(document=doc, stop_words=stop_words, spacy_pipeline=spacy_pipeline,
                                     pos_pattern=pos_pattern, lowercase=lowercase) for doc in document_list]
        keyphrases = [keyphrase for sub_keyphrase_list in keyphrases for keyphrase in
                      sub_keyphrase_list]
        return list(set(keyphrases))
