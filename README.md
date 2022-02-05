[![PyPI - Python](https://img.shields.io/badge/python-%3E%3D3.7-blue)](https://pypi.org/project/keyphrase-vectorizers/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-green.svg)](https://github.com/TimSchopf/Keyphrase_Vectorizers/blob/master/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/keyphrase-vectorizers.svg)](https://pypi.org/project/keyphrase-vectorizers/)
[![Documentation Status](https://readthedocs.org/projects/keyphrase-vectorizers/badge/?version=latest)](https://keyphrase-vectorizers.readthedocs.io/en/latest/?badge=latest)

KeyphraseVectorizers
===================== 

Set of vectorizers that extract keyphrases with part-of-speech patterns from a collection of text documents and convert
them into a document-keyphrase matrix. A document-keyphrase matrix is a mathematical matrix that describes the frequency
of keyphrases that occur in a collection of documents. The matrix rows indicate the text documents and columns indicate
the unique keyphrases.

The package contains wrappers of the
[sklearn.feature_extraction.text.CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html?highlight=countvectorizer#sklearn.feature_extraction.text.CountVectorizer "scikit-learn CountVectorizer")
and
[sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer "scikit-learn TfidfVectorizer")
classes. Instead of using n-gram tokens of a pre-defined range, these classes extract keyphrases from text documents
using part-of-speech tags to compute document-keyphrase matrices.

Benefits
--------

* Extract grammatically accurate keyphases based on their part-of-speech tags.
* No need to specify n-gram ranges.
* Get document-keyphrase matrices.
* Multiple language support.
* User-defined part-of-speech patterns for keyphrase extraction possible.

<a name="toc"/></a>

Table of Contents
-----------------

<!--ts-->

1. [How does it work?](#how-does-it-work)
2. [Installation](#installation)
3. [Usage](#usage)
   1. [KeyphraseCountVectorizer](#KeyphraseCountVectorizer)
      1. [English language](#english-language)
      2. [Other languages](#other-languages)
   2. [KeyphraseTfidfVectorizer](#KeyphraseTfidfVectorizer)
   3. [Keyphrase extraction with KeyBERT](#keyphrase-extraction-with-keybert)

<!--te-->

<a name="#how-does-it-work"/></a>

How does it work?
-----------------

First, the document texts are annotated with [spaCy](https://spacy.io "spaCy homepage") part-of-speech tags. A list of
all possible spaCy part-of-speech tags for different languages is
linked [here](https://github.com/explosion/spaCy/blob/master/spacy/glossary.py "spaCy POS tags"). The annotation
requires passing the [spaCy pipeline](https://spacy.io/models "available spaCy pipelines") of the corresponding language
to the vectorizer with the `spacy_pipeline` parameter.

Second, words are extracted from the document texts whose part-of-speech tags match the regex pattern defined in
the `pos_pattern`
parameter. The keyphrases are a list of unique words extracted from text documents by this method.

Finally, the vectorizers calculate document-keyphrase matrices.

<a name="#installation"/></a>

Installation
------------

```
pip install keyphrase-vectorizers
```

<a name="#usage"/></a>

Usage
-----
For detailed information visit
the [API Guide](https://keyphrase-vectorizers.readthedocs.io/en/latest/index.html "Keyphrase_Vectorizers API Guide").

<a name="#KeyphraseCountVectorizer"/></a>

### KeyphraseCountVectorizer

[Back to Table of Contents](#toc)

<a name="#english-language"/></a>

#### English language

```python
from keyphrase_vectorizers import KeyphraseCountVectorizer

docs = ["""Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal). 
         A supervised learning algorithm analyzes the training data and produces an inferred function, 
         which can be used for mapping new examples. An optimal scenario will allow for the 
         algorithm to correctly determine the class labels for unseen instances. This requires 
         the learning algorithm to generalize from the training data to unseen situations in a 
         'reasonable' way (see inductive bias).""", 
             
        """Keywords are defined as phrases that capture the main topics discussed in a document. 
        As they offer a brief yet precise summary of document content, they can be utilized for various applications. 
        In an information retrieval environment, they serve as an indication of document relevance for users, as the list 
        of keywords can quickly help to determine whether a given document is relevant to their interest. 
        As keywords reflect a document's main topics, they can be utilized to classify documents into groups 
        by measuring the overlap between the keywords assigned to them. Keywords are also used proactively 
        in information retrieval."""]
        
# Init default vectorizer.
vectorizer = KeyphraseCountVectorizer()

# Print parameters
print(vectorizer.get_params())
>>> {'binary': False, 'dtype': <class 'numpy.int64'>, 'lowercase': True, 'max_df': None, 'min_df': None, 'multiprocessing': False, 'pos_pattern': '<J.*>*<N.*>+', 'spacy_pipeline': 'en_core_web_sm', 'stop_words': 'english'}
```

By default, the vectorizer is initialized for the English language. That means, an English `spacy_pipeline` is
specified, English `stop_words` are removed, and the `pos_pattern` extracts keywords that have 0 or more adjectives,
followed by 1 or more nouns using the English spaCy part-of-speech tags.

```python
# After initializing the vectorizer, it can be fitted
# to learn the keyphrases from the text documents.
vectorizer.fit(docs)
```

```python
# After learning the keyphrases, they can be returned.
keyphrases = vectorizer.get_feature_names_out()

print(keyphrases)
>>> ['output' 'training data' 'task' 'way' 'input object' 'documents'
 'unseen instances' 'vector' 'interest' 'learning algorithm'
 'unseen situations' 'training examples' 'machine' 'given document'
 'document' 'document relevance' 'output pairs' 'document content'
 'class labels' 'new examples' 'pair' 'main topics' 'phrases' 'overlap'
 'algorithm' 'various applications' 'information retrieval' 'users' 'list'
 'example input' 'supervised learning' 'optimal scenario'
 'precise summary' 'keywords' 'input' 'supervised learning algorithm'
 'example' 'supervisory signal' 'indication' 'set'
 'information retrieval environment' 'output value' 'inductive bias'
 'groups' 'function']
```

```python
# After fitting, the vectorizer can transform the documents 
# to a document-keyphrase matrix.
# Matrix rows indicate the documents and columns indicate the unique keyphrases.
# Each cell represents the count.
document_keyphrase_matrix = vectorizer.transform(docs).toarray()

print(document_keyphrase_matrix)
>>> [[3 3 1 1 1 0 1 1 0 2 1 1 1 0 0 0 1 0 1 1 1 0 0 0 3 0 0 0 0 1 3 1 0 0 3 1
  2 1 0 1 0 1 1 0 3]
 [0 0 0 0 0 1 0 0 1 0 0 0 0 1 5 1 0 1 0 0 0 2 1 1 0 1 2 1 1 0 0 0 1 5 0 0
  0 0 1 0 1 0 0 1 0]]
```

```python
# Fit and transform can also be executed in one step, 
# which is more efficient. 
document_keyphrase_matrix = vectorizer.fit_transform(docs).toarray()

print(document_keyphrase_matrix)
>>> [[3 3 1 1 1 0 1 1 0 2 1 1 1 0 0 0 1 0 1 1 1 0 0 0 3 0 0 0 0 1 3 1 0 0 3 1
  2 1 0 1 0 1 1 0 3]
 [0 0 0 0 0 1 0 0 1 0 0 0 0 1 5 1 0 1 0 0 0 2 1 1 0 1 2 1 1 0 0 0 1 5 0 0
  0 0 1 0 1 0 0 1 0]]
```

<a name="#other-languages"/></a>

#### Other languages

[Back to Table of Contents](#toc)

```python
german_docs = ["""Goethe stammte aus einer angesehenen bürgerlichen Familie. 
                Sein Großvater mütterlicherseits war als Stadtschultheiß höchster Justizbeamter der Stadt Frankfurt, 
                sein Vater Doktor der Rechte und Kaiserlicher Rat. Er und seine Schwester Cornelia erfuhren eine aufwendige 
                Ausbildung durch Hauslehrer. Dem Wunsch seines Vaters folgend, studierte Goethe in Leipzig und Straßburg 
                Rechtswissenschaft und war danach als Advokat in Wetzlar und Frankfurt tätig. 
                Gleichzeitig folgte er seiner Neigung zur Dichtkunst.""",
              
               """Friedrich Schiller wurde als zweites Kind des Offiziers, Wundarztes und Leiters der Hofgärtnerei in 
               Marbach am Neckar Johann Kaspar Schiller und dessen Ehefrau Elisabetha Dorothea Schiller, geb. Kodweiß, 
               die Tochter eines Wirtes und Bäckers war, 1759 in Marbach am Neckar geboren
               """]
# Init vectorizer for the german language
vectorizer = KeyphraseCountVectorizer(spacy_pipeline='de_core_news_sm', pos_pattern='<ADJ.*>*<N.*>+', stop_words='german')
```

The German `spacy_pipeline` is specified and German `stop_words` are removed. Because the German spaCy part-of-speech
tags differ from the English ones, the `pos_pattern` parameter is also customized. The regex pattern `<ADJ.*>*<N.*>+`
extracts keywords that have 0 or more adjectives, followed by 1 or more nouns using the German spaCy part-of-speech
tags.

<a name="#KeyphraseTfidfVectorizer"/></a>

### KeyphraseTfidfVectorizer

[Back to Table of Contents](#toc)

The `KeyphraseTfidfVectorizer` has the same function calls and features as the `KeyphraseCountVectorizer`. The only
difference is, that document-keyphrase matrix cells represent tf or tf-idf values, depending on the parameter settings,
instead of counts.

```python
from keyphrase_vectorizers import KeyphraseTfidfVectorizer

docs = ["""Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal). 
         A supervised learning algorithm analyzes the training data and produces an inferred function, 
         which can be used for mapping new examples. An optimal scenario will allow for the 
         algorithm to correctly determine the class labels for unseen instances. This requires 
         the learning algorithm to generalize from the training data to unseen situations in a 
         'reasonable' way (see inductive bias).""", 
             
        """Keywords are defined as phrases that capture the main topics discussed in a document. 
        As they offer a brief yet precise summary of document content, they can be utilized for various applications. 
        In an information retrieval environment, they serve as an indication of document relevance for users, as the list 
        of keywords can quickly help to determine whether a given document is relevant to their interest. 
        As keywords reflect a document's main topics, they can be utilized to classify documents into groups 
        by measuring the overlap between the keywords assigned to them. Keywords are also used proactively 
        in information retrieval."""]
        
# Init default vectorizer for the English language that computes tf-idf values
vectorizer = KeyphraseTfidfVectorizer()

# Print parameters
print(vectorizer.get_params())
{'binary': False, 'dtype': <class 'numpy.float64'>, 'lowercase': True, 'max_df': None, 'min_df': None, 'multiprocessing': False, 'norm': 'l2', 'pos_pattern': '<J.*>*<N.*>+', 'smooth_idf': True, 'spacy_pipeline': 'en_core_web_sm', 'stop_words': 'english', 'sublinear_tf': False, 'use_idf': True}
```

To calculate tf values instead, set `use_idf=False`.

```python
# Fit and transform to document-keyphrase matrix.
document_keyphrase_matrix = vectorizer.fit_transform(docs).toarray()

print(document_keyphrase_matrix)
>>> [[0.11111111 0.22222222 0.11111111 0.         0.         0.
  0.11111111 0.         0.11111111 0.11111111 0.33333333 0.
  0.         0.         0.11111111 0.         0.         0.11111111
  0.         0.33333333 0.         0.22222222 0.         0.11111111
  0.11111111 0.11111111 0.11111111 0.11111111 0.33333333 0.11111111
  0.11111111 0.33333333 0.11111111 0.         0.33333333 0.
  0.         0.         0.11111111 0.         0.11111111 0.11111111
  0.         0.33333333 0.11111111]
 [0.         0.         0.         0.11785113 0.11785113 0.11785113
  0.         0.11785113 0.         0.         0.         0.11785113
  0.11785113 0.11785113 0.         0.11785113 0.23570226 0.
  0.23570226 0.         0.58925565 0.         0.11785113 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.58925565 0.         0.11785113
  0.11785113 0.11785113 0.         0.11785113 0.         0.
  0.11785113 0.         0.        ]]
```

```python
# Return keyphrases
keyphrases = vectorizer.get_feature_names_out()

print(keyphrases)
>>> ['optimal scenario' 'example' 'input object' 'groups' 'list'
 'precise summary' 'inductive bias' 'phrases' 'training examples'
 'output value' 'function' 'given document' 'documents'
 'information retrieval environment' 'new examples' 'interest'
 'main topics' 'unseen situations' 'information retrieval' 'input'
 'keywords' 'learning algorithm' 'indication' 'set' 'example input'
 'vector' 'machine' 'supervised learning algorithm' 'algorithm' 'pair'
 'task' 'training data' 'way' 'document' 'supervised learning' 'users'
 'document relevance' 'document content' 'supervisory signal' 'overlap'
 'class labels' 'unseen instances' 'various applications' 'output'
 'output pairs']
```

<a name="#keyphrase-extraction-with-keybert"/></a>

### Keyphrase extraction with [KeyBERT](https://github.com/MaartenGr/KeyBERT "KeyBERT repository")

[Back to Table of Contents](#toc)

The keyphrase vectorizers can be used together with KeyBERT to extract grammatically correct keyphrases that are most
similar to a document. Thereby, the vectorizer first extracts candidate keyphrases from the text documents, which are
subsequently ranked by KeyBERT based on their document similarity. The top-n most similar keyphrases can then be
considered as document keywords.

The advantage of using KeyphraseVectorizers in addition to KeyBERT is that it allows users to get grammatically correct
keyphrases instead of simple n-grams of pre-defined lengths. In KeyBERT, users can specify the `keyphrase_ngram_range`
to define the length of the retrieved keyphrases. However, this raises two issues. First, users usually do not know the
optimal n-gram range and therefore have to spend some time experimenting until they find a suitable n-gram range.
Second, even after finding a good n-gram range, the returned keyphrases are sometimes still grammatically not quite
correct or are slightly off-key. Unfortunately, this limits the quality of the returned keyphrases.

To adress this issue, we can use the vectorizers of this package to first extract candidate keyphrases that consist of
zero or more adjectives, followed by one or multiple nouns in a pre-processing step instead of simple n-grams.
[Wan and Xiao](https://www.aaai.org/Papers/AAAI/2008/AAAI08-136.pdf) successfully used this noun phrase approach for
keyphrase extraction during their research in 2008. The extracted candidate keyphrases are subsequently passed to
KeyBERT for embedding generation and similarity calculation. To use both packages for keyphrase extraction, we need to
pass KeyBERT a keyphrase vectorizer with the `vectorizer` parameter. Since the length of keyphrases now depends on
part-of-speech tags, there is no need to define an n-gram length anymore.

#### Example:

KeyBERT can be installed via `pip install keybert`.

```python
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT

docs = ["""Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal). 
         A supervised learning algorithm analyzes the training data and produces an inferred function, 
         which can be used for mapping new examples. An optimal scenario will allow for the 
         algorithm to correctly determine the class labels for unseen instances. This requires 
         the learning algorithm to generalize from the training data to unseen situations in a 
         'reasonable' way (see inductive bias).""", 
             
        """Keywords are defined as phrases that capture the main topics discussed in a document. 
        As they offer a brief yet precise summary of document content, they can be utilized for various applications. 
        In an information retrieval environment, they serve as an indication of document relevance for users, as the list 
        of keywords can quickly help to determine whether a given document is relevant to their interest. 
        As keywords reflect a document's main topics, they can be utilized to classify documents into groups 
        by measuring the overlap between the keywords assigned to them. Keywords are also used proactively 
        in information retrieval."""]

kw_model = KeyBERT()
```

Instead of deciding on a suitable n-gram range which could be e.g.(1,2)...

```python
>>> kw_model.extract_keywords(docs=docs, keyphrase_ngram_range=(1,2))
[[('labeled training', 0.6013),
  ('examples supervised', 0.6112),
  ('signal supervised', 0.6152),
  ('supervised', 0.6676),
  ('supervised learning', 0.6779)],
 [('keywords assigned', 0.6354),
  ('keywords used', 0.6373),
  ('list keywords', 0.6375),
  ('keywords quickly', 0.6376),
  ('keywords defined', 0.6997)]]
```

we can now just let the keyphrase vectorizer decide on suitable keyphrases, without limitations to a maximum or minimum
n-gram range. We only have to pass a keyphrase vectorizer as parameter to KeyBERT:

```python
>>> kw_model.extract_keywords(docs=docs, vectorizer=KeyphraseCountVectorizer())
[[('training examples', 0.4668),
  ('training data', 0.5271),
  ('learning algorithm', 0.5632),
  ('supervised learning', 0.6779),
  ('supervised learning algorithm', 0.6992)],
 [('given document', 0.4143),
  ('information retrieval environment', 0.5166),
  ('information retrieval', 0.5792),
  ('keywords', 0.6046),
  ('document relevance', 0.633)]]
```

This allows us to make sure that we do not cut off important words caused by defining our n-gram range too short. For
example, we would not have found the keyphrase "supervised learning algorithm" with keyphrase_ngram_range=(1,2).
Furthermore, we avoid to get keyphrases that are slightly off-key like "labeled training", "signal supervised" or
"keywords quickly".
