from typing import List

import flair
import spacy
from bertopic import BERTopic
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter
from keybert import KeyBERT
from sklearn.datasets import fetch_20newsgroups

import tests.utils as utils
from keyphrase_vectorizers import KeyphraseCountVectorizer, KeyphraseTfidfVectorizer

english_docs = utils.get_english_test_docs()
german_docs = utils.get_german_test_docs()
french_docs = utils.get_french_docs()


def test_default_count_vectorizer():
    sorted_english_test_keyphrases = utils.get_english_test_keyphrases()
    sorted_count_matrix = utils.get_sorted_english_count_matrix()

    vectorizer = KeyphraseCountVectorizer()
    vectorizer.fit(english_docs)
    keyphrases = vectorizer.get_feature_names_out()
    document_keyphrase_matrix = vectorizer.transform(english_docs).toarray()

    assert [sorted(count_list) for count_list in
            KeyphraseCountVectorizer().fit_transform(english_docs).toarray()] == sorted_count_matrix
    assert [sorted(count_list) for count_list in document_keyphrase_matrix] == sorted_count_matrix
    assert sorted(keyphrases) == sorted_english_test_keyphrases


def test_spacy_language_argument():
    sorted_english_test_keyphrases = utils.get_english_test_keyphrases()
    sorted_count_matrix = utils.get_sorted_english_count_matrix()

    nlp = spacy.load("en_core_web_sm")

    vectorizer = KeyphraseCountVectorizer(spacy_pipeline=nlp)
    vectorizer.fit(english_docs)
    keyphrases = vectorizer.get_feature_names_out()
    document_keyphrase_matrix = vectorizer.transform(english_docs).toarray()

    assert [sorted(count_list) for count_list in
            KeyphraseCountVectorizer().fit_transform(english_docs).toarray()] == sorted_count_matrix
    assert [sorted(count_list) for count_list in document_keyphrase_matrix] == sorted_count_matrix
    assert sorted(keyphrases) == sorted_english_test_keyphrases


def test_german_count_vectorizer():
    sorted_german_test_keyphrases = utils.get_german_test_keyphrases()

    vectorizer = KeyphraseCountVectorizer(spacy_pipeline='de_core_news_sm', pos_pattern='<ADJ.*>*<N.*>+',
                                          stop_words='german')
    keyphrases = vectorizer.fit(german_docs).get_feature_names_out()
    assert sorted(keyphrases) == sorted_german_test_keyphrases


def test_default_tfidf_vectorizer():
    sorted_english_test_keyphrases = utils.get_english_test_keyphrases()
    sorted_english_tfidf_matrix = utils.get_sorted_english_tfidf_matrix()

    vectorizer = KeyphraseTfidfVectorizer()
    vectorizer.fit(english_docs)
    keyphrases = vectorizer.get_feature_names_out()
    document_keyphrase_matrix = vectorizer.transform(english_docs).toarray()
    document_keyphrase_matrix = [[round(element, 10) for element in tfidf_list] for tfidf_list in
                                 document_keyphrase_matrix]

    assert [sorted(tfidf_list) for tfidf_list in document_keyphrase_matrix] == sorted_english_tfidf_matrix
    assert sorted(keyphrases) == sorted_english_test_keyphrases


def test_keybert_integration():
    english_keybert_keyphrases = utils.get_english_keybert_keyphrases()
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    keyphrases = kw_model.extract_keywords(docs=english_docs, vectorizer=KeyphraseCountVectorizer())
    keyphrases = [[element[0] for element in keyphrases_list] for keyphrases_list in keyphrases]

    assert keyphrases == english_keybert_keyphrases


def test_french_trf_spacy_pipeline():
    sorted_french_test_keyphrases = utils.get_french_test_keyphrases()
    sorted_french_count_matrix = utils.get_sorted_french_count_matrix()

    vectorizer = KeyphraseCountVectorizer(spacy_pipeline='fr_dep_news_trf', spacy_exclude=[])
    vectorizer.fit(french_docs)
    keyphrases = vectorizer.get_feature_names_out()
    document_keyphrase_matrix = vectorizer.transform(french_docs).toarray()

    assert [sorted(count_list) for count_list in
            KeyphraseCountVectorizer(spacy_pipeline='fr_dep_news_trf', spacy_exclude=[]).fit_transform(
                french_docs).toarray()] == sorted_french_count_matrix
    assert [sorted(count_list) for count_list in document_keyphrase_matrix] == sorted_french_count_matrix
    assert sorted(keyphrases) == sorted_french_test_keyphrases


def test_custom_tagger():
    sorted_english_test_keyphrases = utils.get_sorted_english_keyphrases_custom_flair_tagger()

    tagger = SequenceTagger.load('pos')
    splitter = SegtokSentenceSplitter()

    # define custom pos tagger function using flair
    def custom_pos_tagger(raw_documents: List[str], tagger: flair.models.SequenceTagger = tagger,
                          splitter: flair.tokenization.SegtokSentenceSplitter = splitter) -> List[tuple]:
        """
        Important:

        The mandatory 'raw_documents' parameter can NOT be named differently and has to expect a list of strings.
        Furthermore the function has to return a list of (word token, POS-tag) tuples.
        """
        # split texts into sentences
        sentences = []
        for doc in raw_documents:
            sentences.extend(splitter.split(doc))

        # predict POS tags
        tagger.predict(sentences)

        # iterate through sentences to get word tokens and predicted POS-tags
        pos_tags = []
        words = []
        for sentence in sentences:
            pos_tags.extend([label.value for label in sentence.get_labels('pos')])
            words.extend([word.text for word in sentence])

        return list(zip(words, pos_tags))

    vectorizer = KeyphraseCountVectorizer(custom_pos_tagger=custom_pos_tagger)
    vectorizer.fit(english_docs)
    keyphrases = vectorizer.get_feature_names_out()

    assert sorted(keyphrases) == sorted_english_test_keyphrases


def test_online_vectorizer():
    first_doc_count_matrix = utils.get_sorted_english_first_doc_count_matrix()
    second_doc_count_matrix = utils.get_sorted_english_second_doc_count_matrix()
    first_doc_test_keyphrases = utils.get_english_first_doc_test_keyphrases()
    english_keyphrases = utils.get_english_test_keyphrases()
    frequencies_after_min_df = utils.get_frequencies_after_min_df()
    frequent_keyphrases_after_min_df = utils.get_frequent_keyphrases_after_min_df()
    frequencies_after_bow = utils.get_frequencies_after_bow()

    # intitial vectorizer fit
    vectorizer = KeyphraseCountVectorizer(decay=0.5, delete_min_df=3)

    assert [sorted(count_list) for count_list in
            vectorizer.fit_transform([english_docs[0]]).toarray()] == first_doc_count_matrix
    assert sorted(vectorizer.get_feature_names_out()) == first_doc_test_keyphrases

    # learn additional keyphrases from new documents with partial fit
    vectorizer.partial_fit([english_docs[1]])

    assert [sorted(count_list) for count_list in
            vectorizer.transform([english_docs[1]]).toarray()] == second_doc_count_matrix
    assert sorted(vectorizer.get_feature_names_out()) == english_keyphrases

    # update list of learned keyphrases according to 'delete_min_df'
    vectorizer.update_bow([english_docs[1]])
    assert (vectorizer.transform([english_docs[1]]).toarray() == frequencies_after_min_df).all()

    # check updated list of learned keyphrases (only the ones that appear more than 'delete_min_df' remain)
    assert sorted(vectorizer.get_feature_names_out()) == frequent_keyphrases_after_min_df

    # update again and check the impact of 'decay' on the learned document-keyphrase matrix
    vectorizer.update_bow([english_docs[1]])
    assert (vectorizer.X_.toarray() == frequencies_after_bow).all()


def test_bertopic():
    data = fetch_20newsgroups(subset='train')
    texts = data.data[:100]
    topic_model = BERTopic(vectorizer_model=KeyphraseCountVectorizer())
    topics, probs = topic_model.fit_transform(documents=texts)
    new_topics = topic_model.reduce_outliers(texts, topics)
    topic_model.update_topics(texts, topics=new_topics)
