"""
.. _spaCy pipeline: https://spacy.io/models
.. _stopwords available in NLTK: https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/stopwords.zip
.. _POS-tags: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
.. _regex pattern: https://docs.python.org/3/library/re.html#regular-expression-syntax
"""

import logging
import os
from typing import List

import nltk
import numpy as np
import psutil
import scipy.sparse as sp
import spacy


class _KeyphraseVectorizerMixin():
    """
    _KeyphraseVectorizerMixin

    Provides common code for text vectorizers.
    """

    def _document_frequency(self, document_keyphrase_count_matrix: List[List[int]]) -> np.array:
        """
        Count the number of non-zero values for each feature in sparse a matrix.

        Parameters
        ----------
        document_keyphrase_count_matrix : list of integer lists
                The document-keyphrase count matrix to transform to document frequencies

        Returns
        -------
        document_frequencies : np.array
            Numpy array of document frequencies for keyphrases
        """
        document_keyphrase_count_matrix = sp.csr_matrix(document_keyphrase_count_matrix)
        document_frequencies = np.bincount(document_keyphrase_count_matrix.indices,
                                           minlength=document_keyphrase_count_matrix.shape[1])

        return document_frequencies

    def _remove_suffixes(self, text: str, suffixes: List[str]) -> str:
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

    def _remove_prefixes(self, text: str, prefixes: List[str]) -> str:
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

    def _cumulative_length_joiner(self, text_list: List[str], max_text_length: int) -> List[str]:
        """
        Joins strings from list of strings to single string until maximum char length is reached.
        Then join the next strings from list to a single string and so on.

        Parameters
        ----------
        text_list : list of strings
            List of strings to join.

        max_text_length : int
            Maximun character length of the joined strings.

        Returns
        -------
        list_of_joined_srings_with_max_length : List of joined text strings with max char length of 'max_text_length.
        """

        # triggers a parameter validation
        if isinstance(text_list, str):
            raise ValueError(
                "Iterable over raw texts expected, string object received."
            )

        # triggers a parameter validation
        if not hasattr(text_list, '__iter__'):
            raise ValueError(
                "Iterable over texts expected."
            )

        text_list_len = len(text_list) - 1
        list_of_joined_srings_with_max_length = []
        one_string = ''
        for index, text in enumerate(text_list):
            # Add the text to the substring if it doesn't make it to large
            if len(one_string) + len(text) < max_text_length:
                one_string += ' ' + text
                if index == text_list_len:
                    list_of_joined_srings_with_max_length.append(one_string)

            # Substring too large, so add to the list and reset
            else:
                list_of_joined_srings_with_max_length.append(one_string)
                one_string = text
                if index == text_list_len:
                    list_of_joined_srings_with_max_length.append(one_string)
        return list_of_joined_srings_with_max_length

    def _split_long_document(self, text: str, max_text_length: int) -> List[str]:
        """
        Split single string in list of strings with a maximum character length.

        Parameters
        ----------
        text : str
            Text string that should be split.

        max_text_length : int
            Maximun character length of the strings.

        Returns
        -------
        splitted_document : List of text strings.
        """
        # triggers a parameter validation
        if not isinstance(text, str):
            raise ValueError(
                "'text' parameter needs to be a string."
            )

        # triggers a parameter validation
        if not isinstance(max_text_length, int):
            raise ValueError(
                "'max_text_length' parameter needs to be a int"
            )

        text = text.replace("? ", "?<stop>")
        text = text.replace("! ", "!<stop>")
        if "<stop>" in text:
            splitted_document = text.split("<stop>")
            splitted_document = splitted_document[:-1]
            splitted_document = [s.strip() for s in splitted_document]
            splitted_document = [
                self._cumulative_length_joiner(text_list=doc.split(" "), max_text_length=max_text_length) if len(
                    doc) > max_text_length else [doc] for doc in splitted_document]
            return [text for doc in splitted_document for text in doc]
        else:
            splitted_document = text.split(" ")
            splitted_document = self._cumulative_length_joiner(text_list=splitted_document,
                                                               max_text_length=max_text_length)
            return splitted_document

    def _get_pos_keyphrases(self, document_list: List[str], stop_words: str, spacy_pipeline: str, pos_pattern: str, pos_tagger: any = None,
                            lowercase: bool = True, workers: int = 1) -> List[str]:
        """
        Select keyphrases with part-of-speech tagging from a text document.
        Parameters
        ----------
        document_list : list of str
            List of text documents from which to extract the keyphrases.

        stop_words : str
            Language of stopwords to remove from the document, e.g. 'english'.
            Supported options are `stopwords available in NLTK`_.
            Removes unwanted stopwords from keyphrases if 'stop_words' is not None.

        spacy_pipeline : str
            The name of the `spaCy pipeline`_, used to tag the parts-of-speech in the text.

        pos_pattern : str
            The `regex pattern`_ of `POS-tags`_ used to extract a sequence of POS-tagged tokens from the text.

        lowercase : bool, default=True
            Whether the returned keyphrases should be converted to lowercase.

        workers :int, default=1
            How many workers to use for spaCy part-of-speech tagging.
            If set to -1, use all available worker threads of the machine.
            spaCy uses the specified number of cores to tag documents with part-of-speech.
            Depending on the platform, starting many processes with multiprocessing can add a lot of overhead.
            In particular, the default start method spawn used in macOS/OS X (as of Python 3.8) and in Windows can be slow.
            Therefore, carefully consider whether this option is really necessary.

        Returns
        -------
        keyphrases : List of unique keyphrases of varying length, extracted from the text document with the defined 'pos_pattern'.
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

        stop_words_list = []
        if stop_words:
            try:
                stop_words_list = set(nltk.corpus.stopwords.words(stop_words))
            except LookupError:
                logger = logging.getLogger('KeyphraseVectorizer')
                logger.setLevel(logging.WARNING)
                sh = logging.StreamHandler()
                sh.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logger.addHandler(sh)
                logger.setLevel(logging.DEBUG)
                logger.info(
                    'It looks like you do not have downloaded a list of stopwords yet. It is attempted to download the stopwords now.')
                nltk.download('stopwords')
                stop_words_list = set(nltk.corpus.stopwords.words(stop_words))

        # add spaCy POS tags for documents
        spacy_exclude = ['parser', 'ner', 'entity_linker', 'entity_ruler', 'textcat', 'textcat_multilabel',
                         'lemmatizer', 'morphologizer', 'senter', 'sentencizer', 'tok2vec', 'transformer']
        try:
            nlp = spacy.load(spacy_pipeline,
                             exclude=spacy_exclude)

        except OSError:
            # set logger
            logger = logging.getLogger('KeyphraseVectorizer')
            logger.setLevel(logging.WARNING)
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(sh)
            logger.setLevel(logging.DEBUG)
            logger.info(
                'It looks like the selected spaCy pipeline is not downloaded yet. It is attempted to download the spaCy pipeline now.')
            spacy.cli.download(spacy_pipeline)
            nlp = spacy.load(spacy_pipeline,
                             exclude=spacy_exclude)

        # add rule based sentence boundary detection
        nlp.add_pipe('sentencizer')

        if pos_tagger != None:
          pos_tagger_component = Language.component("pos_tagger", func=pos_tagger)
          nlp.add_pipe("pos_tagger", name="pos_tagger", first=True)

        keyphrases_list = []
        if workers != 1:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # split large documents in smaller chunks, so that spacy can process them without memory issues
        docs_list = []
        # set maximal character length of documents for spaCy processing
        max_doc_length = 1000000
        for document in document_list:
            if len(document) > max_doc_length:
                docs_list.extend(self._split_long_document(text=document, max_text_length=max_doc_length))
            else:
                docs_list.append(document)
        document_list = docs_list
        del docs_list

        # increase max length of documents that spaCy can parse
        # (should only be done if parser and ner are not used due to memory issues)
        nlp.max_length = max([len(doc) for doc in document_list]) + 100

        cp = nltk.RegexpParser('CHUNK: {(' + pos_pattern + ')}')
        for tagged_doc in nlp.pipe(document_list, n_process=workers):
            tagged_pos_doc = []
            for sentence in tagged_doc.sents:
                pos_tagged_sentence = []
                for word in sentence:
                    pos_tagged_sentence.append((word.text, word.tag_))
                tagged_pos_doc.append(pos_tagged_sentence)

            # extract keyphrases that match the NLTK RegexpParser filter
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

            keyphrases_list.append(list(set(keyphrases)))

        return list(set([keyphrase for sub_keyphrase_list in keyphrases_list for keyphrase in sub_keyphrase_list]))
