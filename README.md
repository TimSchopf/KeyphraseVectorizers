[![Documentation Status](https://readthedocs.org/projects/keyphrase-vectorizers/badge/?version=latest)](https://keyphrase-vectorizers.readthedocs.io/en/latest/?badge=latest)

Keyphrase_Vectorizers
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
      1. [English language](#KeyphraseCountVectorizer)
      2. [Other languages](#otherlanguages)
   2. [KeyphraseTfidfVectorizer](#KeyphraseTfidfVectorizer)

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

#### English language

```python
from keyphrase_vectorizers.keyphrase_count_vectorizer import KeyphraseCountVectorizer

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
>>> {'binary': False, 'dtype': <class 'numpy.int64'>, 'lowercase': True, 'pos_pattern': '<J.*>*<N.*>+', 'spacy_pipeline': 'en_core_web_sm', 'stop_words': 'english'}
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

<a name="#otherlanguages"/></a>
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
from keyphrase_vectorizers.keyphrase_tfidf_vectorizer import KeyphraseTfidfVectorizer

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
>>> {'binary': False, 'dtype': <class 'numpy.float64'>, 'lowercase': True, 'norm': 'l2', 'pos_pattern': '<J.*>*<N.*>+', 'smooth_idf': True, 'spacy_pipeline': 'en_core_web_sm', 'stop_words': 'english', 'sublinear_tf': False, 'use_idf': True}
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