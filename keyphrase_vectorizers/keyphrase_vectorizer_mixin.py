"""
.. _spaCy pipeline: https://spacy.io/models
.. _stopwords available in NLTK: https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/stopwords.zip
.. _POS-tags: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
.. _regex pattern: https://docs.python.org/3/library/re.html#regular-expression-syntax
.. _spaCy pipeline components: https://spacy.io/usage/processing-pipelines#built-in
"""

import logging
import os
from typing import List, Union

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

    def build_tokenizer(self) -> callable:
        """
        Return a function that splits a string into a sequence of tokens.

        Returns
        -------
        tokenizer: callable
              A function to split a string into a sequence of tokens.
        """

        return self._tokenize

    def _tokenize_simple(self, text: str) -> List[str]:
        """
        Simple tokenizer that just splits strings by whitespace.

        Parameters
        ----------
        text : str
                The text to tokenize.

        Returns
        -------
        tokens: List[str]
              A list of tokens.
        """

        tokens = text.split()
        return tokens

    def _tokenize(self, text: str) -> List[str]:
        """
        Custom word tokenizer for sklearn vectorizer that uses a spaCy pipeline for tokenization.

        Parameters
        ----------
        text : str
                The text to tokenize.

        Returns
        -------
        tokens: List[str]
              A list of tokens.
        """

        processed_documents, _ = self._get_pos_keyphrases(document_list=[text],
                                                          stop_words=self.stop_words,
                                                          spacy_pipeline=self.spacy_pipeline,
                                                          pos_pattern=self.pos_pattern,
                                                          lowercase=self.lowercase, workers=self.workers,
                                                          spacy_exclude=['tok2vec', 'tagger', 'parser',
                                                                         'attribute_ruler', 'lemmatizer', 'ner',
                                                                         'textcat'],
                                                          custom_pos_tagger=self.custom_pos_tagger,
                                                          extract_keyphrases=False)

        return self._tokenize_simple(processed_documents[0])

    def _document_frequency(self, document_keyphrase_count_matrix: List[List[int]]) -> np.array:
        """
        Count the number of non-zero values for each feature in sparse a matrix.

        Parameters
        ----------
        document_keyphrase_count_matrix : List[List[int]]
                The document-keyphrase count matrix to transform to document frequencies.

        Returns
        -------
        document_frequencies : np.array
            Numpy array of document frequencies for keyphrases.
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
            Maximum character length of the joined strings.

        Returns
        -------
        list_of_joined_strings_with_max_length : List of joined text strings with max char length of 'max_text_length'.
        """

        if isinstance(text_list, str):
            raise ValueError("Iterable over raw texts expected, string object received.")

        if not isinstance(max_text_length, int) or max_text_length <= 0:
            raise ValueError("max_text_length must be a positive integer.")

        joined_strings = []
        current_string = ""

        for text in text_list:
            if not text:
                continue

            # If the next text exceeds the max length, start a new string
            if len(current_string) + len(text) + 1 > max_text_length:  # +1 for space character
                # Append the current string to the result list
                if current_string:
                    joined_strings.append(current_string.strip())
                # Start a new string with the current text
                current_string = text
            else:
                # Add the text to the current string
                if current_string:
                    current_string += ' ' + text
                else:
                    current_string = text

        # Append the last string to the result list
        if current_string:
            joined_strings.append(current_string.strip())

        return joined_strings

    def _split_long_document(self, text: str, max_text_length: int) -> List[str]:
        """
        Split single string in list of strings with a maximum character length.

        Parameters
        ----------
        text : str
            Text string that should be split.

        max_text_length : int
            Maximum character length of the strings.

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
            splitted_document = [s.strip() for s in splitted_document if s.strip()]  # Filter out empty strings
            splitted_document = [
                self._cumulative_length_joiner(text_list=doc.split(" "), max_text_length=max_text_length) if len(
                    doc) > max_text_length else [doc] for doc in splitted_document]
            return [text for doc in splitted_document for text in doc]
        else:
            # No punctuation marks found, process the entire text
            splitted_document = text.split(" ")
            splitted_document = self._cumulative_length_joiner(text_list=splitted_document,
                                                               max_text_length=max_text_length)
            return splitted_document

    def _get_pos_keyphrases(self, document_list: List[str], stop_words: Union[str, List[str]], spacy_pipeline: Union[str, spacy.Language],
                            pos_pattern: str, spacy_exclude: List[str], custom_pos_tagger: callable,
                            lowercase: bool = True, workers: int = 1, extract_keyphrases: bool = True) -> List[str]:
        """
        Select keyphrases with part-of-speech tagging from a text document.
        Parameters
        ----------
        document_list : list of str
            List of text documents from which to extract the keyphrases.

        stop_words : Union[str, List[str]]
            Language of stopwords to remove from the document, e.g. 'english'.
            Supported options are `stopwords available in NLTK`_.
            Removes unwanted stopwords from keyphrases if 'stop_words' is not None.
            If given a list of custom stopwords, removes them instead.

        spacy_pipeline : Union[str, spacy.Language]
            A spacy.Language object or the name of the `spaCy pipeline`_, used to tag the parts-of-speech in the text.

        pos_pattern : str
            The `regex pattern`_ of `POS-tags`_ used to extract a sequence of POS-tagged tokens from the text.

        spacy_exclude : List[str]
            A list of `spaCy pipeline components`_ that should be excluded during the POS-tagging.
            Removing not needed pipeline components can sometimes make a big difference and improve loading and inference speed.

        custom_pos_tagger : callable
            A callable function which expects a list of strings in a 'raw_documents' parameter and returns a list of (word token, POS-tag) tuples.
            If this parameter is not None, the custom tagger function is used to tag words with parts-of-speech, while the spaCy pipeline is ignored.

        lowercase : bool, default=True
            Whether the returned keyphrases should be converted to lowercase.

        workers : int, default=1
            How many workers to use for spaCy part-of-speech tagging.
            If set to -1, use all available worker threads of the machine.
            spaCy uses the specified number of cores to tag documents with part-of-speech.
            Depending on the platform, starting many processes with multiprocessing can add a lot of overhead.
            In particular, the default start method spawn used in macOS/OS X (as of Python 3.8) and in Windows can be slow.
            Therefore, carefully consider whether this option is really necessary.

        extract_keyphrases : bool, default=True
            Whether to run the keyphrase extraction step or just return an empty list.

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
        if not isinstance(stop_words, str) and (stop_words is not None) and (not hasattr(stop_words, '__iter__')):
            raise ValueError(
                "'stop_words' parameter needs to be a string, e.g. 'english' or 'None' or a list of strings."
            )

        # triggers a parameter validation
        if not isinstance(spacy_pipeline, (str, spacy.Language)):
            raise ValueError(
                "'spacy_pipeline' parameter needs to be a spacy.Language object or a spaCy pipeline string. E.g. 'en_core_web_sm'"
            )

        # triggers a parameter validation
        if not isinstance(pos_pattern, str):
            raise ValueError(
                "'pos_pattern' parameter needs to be a regex string. E.g. '<J.*>*<N.*>+'"
            )

        # triggers a parameter validation
        if ((not hasattr(spacy_exclude, '__iter__')) and (spacy_exclude is not None)) or (
                isinstance(spacy_exclude, str)):
            raise ValueError(
                "'spacy_exclude' parameter needs to be a list of 'spaCy pipeline components' strings."
            )

        # triggers a parameter validation
        if not callable(custom_pos_tagger) and (custom_pos_tagger is not None):
            raise ValueError(
                "'custom_pos_tagger' must be a callable function that gets a list of strings in a 'raw_documents' parameter and returns a list of (word, POS-tag) tuples."
            )

        # triggers a parameter validation
        if not isinstance(workers, int):
            raise ValueError(
                "'workers' parameter must be of type int."
            )

        if (workers < -1) or (workers > psutil.cpu_count(logical=True)) or (workers == 0):
            raise ValueError(
                "'workers' parameter value cannot be 0 and must be between -1 and " + str(
                    psutil.cpu_count(logical=True))
            )


        stop_words_list = set()
        if isinstance(stop_words, str):
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

        elif hasattr(stop_words, '__iter__'):
            stop_words_list = set(stop_words)

        # add spaCy PoS tags for documents
        if not custom_pos_tagger:
            if isinstance(spacy_pipeline, spacy.Language):
                nlp = spacy_pipeline
            else:
                if not spacy_exclude:
                    spacy_exclude = []
                try:
                    if extract_keyphrases:
                        nlp = spacy.load(spacy_pipeline, exclude=spacy_exclude)
                    else:
                        # only use tokenizer if no keywords are extracted
                        nlp = spacy.blank(spacy_pipeline.split("_")[0])

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

        if workers != 1:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # add document delimiter, so we can identify the original document split later
        doc_delimiter = "thisisadocumentdelimiternotakeyphrasepleaseignore"
        document_list = [doc_delimiter + " " + doc for doc in document_list]

        # split large documents in smaller chunks, so that spacy can process them without memory issues
        docs_list = []
        # set maximal character length of documents for spaCy processing
        max_doc_length = 500
        for document in document_list:
            if len(document) > max_doc_length:
                docs_list.extend(self._split_long_document(text=document, max_text_length=max_doc_length))
            else:
                docs_list.append(document)
        document_list = docs_list
        del docs_list

        # increase max length of documents that spaCy can parse
        # (should only be done if parser and ner are not used due to memory issues)
        if not custom_pos_tagger:
            nlp.max_length = max([len(doc) for doc in document_list]) + 100

        if not custom_pos_tagger:
            pos_tuples = []
            for tagged_doc in nlp.pipe(document_list, n_process=workers):
                pos_tuples.extend([(word.text, word.tag_) for word in tagged_doc if word.text])
        else:
            pos_tuples = custom_pos_tagger(raw_documents=document_list)

        # get the original documents after they were processed by a tokenizer and a POS tagger
        processed_docs = []
        for tup in pos_tuples:
            token = tup[0]
            if lowercase:
                token = token.lower()
            if token not in stop_words_list:
                processed_docs.append(token)
        processed_docs = ' '.join(processed_docs)

        # add delimiter to stop_words_list to ignore it during keyphrase extraction
        stop_words_list.add(doc_delimiter)

        # split processed documents by delimiter
        processed_docs = [doc.strip() for doc in processed_docs.split(doc_delimiter)][1:]

        if extract_keyphrases:
            # extract keyphrases that match the NLTK RegexpParser filter
            keyphrases = []
            # prefix_list = [stop_word + ' ' for stop_word in stop_words_list]
            # suffix_list = [' ' + stop_word for stop_word in stop_words_list]
            cp = nltk.RegexpParser('CHUNK: {(' + pos_pattern + ')}')
            tree = cp.parse(pos_tuples)
            for subtree in tree.subtrees(filter=lambda tuple: tuple.label() == 'CHUNK'):
                # join candidate keyphrase from single words
                keyphrase = ' '.join([i[0] for i in subtree.leaves() if i[0] not in stop_words_list])

                # convert keyphrase to lowercase
                if lowercase:
                    keyphrase = keyphrase.lower()

                # remove stopword suffixes
                # keyphrase = self._remove_suffixes(keyphrase, suffix_list)

                # remove stopword prefixes
                # keyphrase = self._remove_prefixes(keyphrase, prefix_list)

                # remove whitespace from the beginning and end of keyphrases
                keyphrase = keyphrase.strip()

                # do not include single keywords that are actually stopwords
                if keyphrase.lower() not in stop_words_list:
                    keyphrases.append(keyphrase)

            # remove potential empty keyphrases
            keyphrases = [keyphrase for keyphrase in keyphrases if keyphrase != '']

        else:
            keyphrases = []

        return processed_docs, list(set(keyphrases))