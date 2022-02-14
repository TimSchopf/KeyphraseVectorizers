from keybert import KeyBERT

import tests.utils as utils
from keyphrase_vectorizers import KeyphraseCountVectorizer, KeyphraseTfidfVectorizer

english_docs = utils.get_english_test_docs()
german_docs = utils.get_german_test_docs()


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
