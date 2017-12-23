import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    #temp_dict = {}
    probabilities = []
    guesses = []


    test_sequences = list(test_set.get_all_Xlengths().values())
    #print(test_sequences)
    #print('------------------')
    #print(list(test_set.get_all_Xlengths()))
    for test_X, test_Xlength in test_sequences:
        best_guess = None
        best_score = float("-Inf")
        temp_dict = {}
        for word, model in models.items():
            try:
                logL = model.score(test_X,test_Xlength)
                temp_dict[word] = logL
            except:
                temp_dict[word] = float("-inf")
                #pass
            if logL > best_score:
                best_score = logL
                best_guess = word
        probabilities.append(temp_dict)
        guesses.append(best_guess)

    # print ('probabilities:')
    # print(probabilities)
    # print('guess:')
    # print(guesses)
    return probabilities, guesses



    # TODO implement the recognizer

    #raise NotImplementedError
