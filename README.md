[![Documentation Status](https://readthedocs.org/projects/keyphrase-vectorizers/badge/?version=latest)](https://keyphrase-vectorizers.readthedocs.io/en/latest/?badge=latest)

Keyphrase_Vectorizers
===================== 

Set of vectorizers that extract keyphrases with part-of-speech patterns from a collection of text documents and convert
them into a document-keyphrase matrix.

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
* User-defined part-of-speech patterns for keyphrase extraction supported.

<a name="toc"/></a>

## Table of Contents

<!--ts-->

1. [How does it work?](#how-does-it-work)

<!--te-->

<a name="#how-does-it-work"/></a>

How does it work?
-----------------
[Back to ToC](#toc)

First, the document texts are annotated with spaCy part-of-speech tags. This requires passing
the [spaCy pipeline](https://spacy.io/models "available spaCy pipelines") of the corresponding language to the
vectorizer with the `spacy_pipeline` parameter. A list of all possible spaCy part-of-speech tags for different languages
is linked [here](https://github.com/explosion/spaCy/blob/master/spacy/glossary.py "spaCy POS tags"). Second, words are
extracted from the document texts whose part-of-speech tags match the regex pattern defined in the `pos_pattern`
parameter. The keyphrases are a list of unique words extracted form text documents by this method. Finally, the
vectorizers calculate document-keyphrase matrices from this. A document-keyphrase matrix is a mathematical matrix that
describes the frequency of keyphrases that occur in a collection of documents.
