def get_english_test_docs():
    english_docs = ["""Supervised learning is the machine learning task of learning a function that
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

    return english_docs


def get_german_test_docs():
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
    return german_docs


def get_french_docs():
    french_docs = ["Les voitures autonomes déplacent la responsabilité de l'assurance vers les constructeurs"]

    return french_docs


def get_english_test_keyphrases():
    sorted_english_test_keyphrases = ['algorithm', 'class labels', 'document', 'document content', 'document relevance',
                                      'documents', 'example', 'example input', 'function', 'groups', 'indication',
                                      'inductive bias', 'information retrieval', 'information retrieval environment',
                                      'input', 'input object', 'interest', 'keywords', 'list', 'machine', 'main topics',
                                      'new examples', 'optimal scenario', 'output', 'output pairs', 'output value',
                                      'overlap', 'pair', 'phrases', 'precise summary', 'set', 'supervised learning',
                                      'supervised learning algorithm', 'supervisory signal', 'task', 'training data',
                                      'training examples', 'unseen instances', 'unseen situations', 'users',
                                      'various applications', 'vector', 'way']

    return sorted_english_test_keyphrases


def get_sorted_english_keyphrases_custom_flair_tagger():
    sorted_english_custom_tagger_keyphrases = ['algorithm', 'class labels', 'document', 'document content',
                                               'document relevance',
                                               'documents', 'example', 'example input-output pairs', 'function',
                                               'groups',
                                               'indication', 'inductive bias', 'inferred function',
                                               'information retrieval', 'information retrieval environment', 'input',
                                               'input object', 'interest', 'keywords', 'learning', 'learning algorithm',
                                               'list', 'machine', 'main topics', 'new examples',
                                               'optimal scenario', 'output', 'output value', 'overlap', 'pair',
                                               'phrases', 'precise summary', 'set', 'supervised learning',
                                               'supervised learning algorithm', 'supervisory signal', 'task',
                                               'training data', 'training examples', 'unseen instances',
                                               'unseen situations', 'users', 'various applications', 'vector', 'way']

    return sorted_english_custom_tagger_keyphrases


def get_german_test_keyphrases():
    sorted_german_test_keyphrases = ['advokat', 'angesehenen bürgerlichen familie', 'ausbildung', 'bäckers',
                                     'dichtkunst',
                                     'ehefrau elisabetha dorothea schiller', 'frankfurt', 'friedrich schiller',
                                     'geb. kodweiß',
                                     'goethe', 'großvater', 'hauslehrer', 'hofgärtnerei', 'höchster justizbeamter',
                                     'kaiserlicher rat', 'leipzig', 'leiters', 'marbach', 'neckar',
                                     'neckar johann kaspar schiller', 'neigung', 'offiziers', 'rechte',
                                     'rechtswissenschaft',
                                     'schwester cornelia', 'stadt frankfurt', 'stadtschultheiß', 'straßburg', 'tochter',
                                     'vater doktor', 'vaters', 'wetzlar', 'wirtes', 'wundarztes', 'wunsch',
                                     'zweites kind']
    return sorted_german_test_keyphrases


def get_french_test_keyphrases():
    sorted_french_test_keyphrases = ['assurance', 'constructeurs', 'responsabilité', 'voitures']

    return sorted_french_test_keyphrases


def get_sorted_english_count_matrix():
    sorted_english_count_matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
         3, 3, 3, 3, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 2, 2, 5, 5]]

    return sorted_english_count_matrix


def get_sorted_french_count_matrix():
    sorted_french_coun_matrix = [[1, 1, 1, 1]]

    return sorted_french_coun_matrix


def get_sorted_english_tfidf_matrix():
    sorted_english_tfidf_matrix = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1139605765,
         0.1139605765, 0.1139605765, 0.1139605765, 0.1139605765, 0.1139605765, 0.1139605765, 0.1139605765, 0.1139605765,
         0.1139605765, 0.1139605765, 0.1139605765, 0.1139605765, 0.1139605765, 0.1139605765, 0.1139605765, 0.1139605765,
         0.1139605765, 0.1139605765, 0.2279211529, 0.3418817294, 0.3418817294, 0.3418817294, 0.3418817294, 0.3418817294,
         0.3418817294],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.1186781658, 0.1186781658, 0.1186781658, 0.1186781658, 0.1186781658, 0.1186781658,
         0.1186781658, 0.1186781658, 0.1186781658, 0.1186781658, 0.1186781658, 0.1186781658, 0.1186781658, 0.2373563316,
         0.2373563316, 0.5933908291, 0.5933908291]]

    return sorted_english_tfidf_matrix


def get_english_keybert_keyphrases():
    english_keybert_keyphrases = [
        ['supervised learning algorithm', 'supervised learning', 'training data', 'training examples', 'class labels'],
        ['document relevance', 'keywords', 'information retrieval', 'information retrieval environment',
         'document content']]

    return english_keybert_keyphrases
