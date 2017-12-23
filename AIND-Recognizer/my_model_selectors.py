import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        #print('---------------------------------')
        #dict_without_word = {k: v for k, v in all_word_Xlengths.items() if k not in [this_word]}
        #print(dict_without_word.keys())
        #print('---------------------------------')
        #print('input parameters : words - {} , hwords - {}',all_word_sequences, all_word_Xlengths)
        #print(this_word)
        #print(all_word_Xlengths[this_word])
        # print(list(all_word_Xlengths.items())[0])
        # print('---------------------------------')
        # #print(list(all_word_Xlengths.items())[1])
        # print(list(all_word_Xlengths.keys())[0])
        # print('---------------------------------')
        # print(list(all_word_Xlengths.values())[0])
        # X_list = [x[0] for x in all_word_Xlengths[this_word]]
        # print(X_list)
        # X_length_list = [x[1] for x in all_word_Xlengths[this_word]]
        # print(X_length_list)


        #print('self x is {}'.format(self.X))

        #print('self lengths is {}'.format(self.lengths))
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        #BIC = −2 log L + p log N
        #also lower BIC the better
        #referred https://discussions.udacity.com/t/parameter-in-bic-selector/394318/3 for parameter understanding
        # and calculation
        model_score = float('INF')
        selector_model = None
        for i in range(self.min_n_components, self.max_n_components + 1):
            try:
                    temp_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                             random_state=self.random_state, verbose=False).fit(self.X,
                                                                                                self.lengths)
                    logL = temp_model.score(self.X, self.lengths)
                    logN = np.log(len(self.X))
                    p = i ** 2 + 2*i*temp_model.n_features - 1
                    BIC = -2 * logL + p* logN
                    if BIC < model_score:
                        selector_model = temp_model
                        model_score = BIC
            except:
                pass

        return selector_model

        # TODO implement model selection based on BIC scores
        #raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model_score = float('-INF')
        selector_model = None
        for i in range(self.min_n_components, self.max_n_components + 1):
            try:
                temp_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                         random_state=self.random_state, verbose=False).fit(self.X,
                                                                                            self.lengths)
                logL = temp_model.score(self.X, self.lengths)
                #print('logl is {}'.format(logL))

                #print('---------------------------------')
                dict_without_word = {k: v for k, v in self.hwords.items() if k not in [self.this_word]}
                #print(dict_without_word.keys())
                sum_score = 0
                for key, value in dict_without_word.items():
                    sum_score+= temp_model.score(value[0],value[1])

                DIC = logL - sum_score
                if DIC > model_score:
                    selector_model = temp_model
                    model_score = DIC
            except:
                pass

        return selector_model


        # TODO implement model selection based on DIC scores
        #raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        #getting zero division error when running recognizer
        #https://discussions.udacity.com/t/divide-by-zero-recognizer-part-3/333516/7
        #seems to have when len(self.sequences) == 1
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model_score = float('-INF')
        selector_model = None
        n_component = 0
        for i in range(self.min_n_components,self.max_n_components+1):
            try:
                #print(len(self.sequences))
                split_method = KFold(n_splits=min(3,len(self.sequences)))
                split_method.get_n_splits()
                total_score = 0
                folds = 0
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    X_train_data, length_train_data = combine_sequences(cv_train_idx,self.sequences)
                    X_test_data, length_test_data = combine_sequences(cv_test_idx,self.sequences)
                    temp_model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train_data, length_train_data)
                    total_score += temp_model.score(X_test_data,length_test_data)
                    folds += 1

                avg_score = total_score/folds
                if avg_score > model_score:
                    selector_model = temp_model
                    model_score = avg_score
                    n_component = i
            except:
                #print('inside exception')
                return self.base_model(self.n_constant)

        return GaussianHMM(n_components=n_component, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)


        # TODO implement model selection using CV
        #raise NotImplementedError

