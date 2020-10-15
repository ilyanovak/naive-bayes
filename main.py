import pandas as pd
import numpy as np

class NaiveBayes:

    def __init__(self):
        pass

    def _get_data(self):
        return self.data

    def _get_labels(self):
        return self.labels

    def _get_describe_attributes(self):
        return self.describe_attributes

    def _get_describe_labels(self):
        return self.describe_labels

    def _get_prior_probability(self, label):
        '''
        Calculates prior probability of specifiedlabel
        Arguments:
            label: Index value of label whose prior probability to return
        Returns prior probability of specified label
        '''
        return self._get_describe_labels()[label]

    def _get_likelihood(self, label, attribute, x):
        '''
        Calculate likelihood (probability) of attribute
        Arguments:
            label: Index value of label whose attribute's gaussian distribution to return
            attribute: Index value of attribute whose gaussian distribution to return
            x: Real value whose likelihood to estimate
        Returns: A likelihood estimate of the value using a gaussian distribution
        '''

        mean = self._get_describe_attributes()[attribute]['mean'][label]
        std = self._get_describe_attributes()[attribute]['std'][label]
        coefficient = 1 / (std * np.sqrt(2 * np.pi))
        numerator = -1 * (x - mean) ** 2
        denominator = 2 * std ** 2
        euler = np.exp(numerator / denominator)
        F_x = coefficient * euler

        return F_x

    def fit(self, X, y):
        '''
        Fits training data on target data using a Naive Bayes algorithm
        Arguments:
            X: Training data with array-like shape with n samples and m attributes
            Y: Target data with array-like shape with n samples
        '''

        # Concatinate and convert X and Y into dataframe.
        # Last column is target values. The rest are training values.
        self.data = pd.concat(
            [pd.DataFrame(X), pd.DataFrame(Y)], axis=1, ignore_index=True)

        # List of unique labels in Y target data
        self.labels = self._get_data()[self._get_data().columns[-1]].unique()

        # Dataframe that describes mean and standard deviation of each attribute for each label
        # Dataframe with describe[i][mean/std][j] where i is attribute index value and j is label index value
        self.describe_attributes = self._get_data().groupby(by=self._get_data()[self._get_data().columns[-1]]).describe()\
            .drop(['count', 'min', '25%', '50%', '75%', 'max'], axis=1, level=1)

        # Series that describes percentage of each label in the target values
        self.describe_labels = self._get_data(
        )[self._get_data().columns[-1]].value_counts(normalize=True)

    def predict(self, X):
        pass
