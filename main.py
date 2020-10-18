import pandas as pd
import numpy as np
import plotly.graph_objects as go


class NaiveBayesGauss:

    def __init__(self):
        self.predict_prob = None

    def _get_data(self):
        return self.data

    def _get_labels(self):
        return self.labels

    def _get_describe_attributes(self):
        return self.describe_attributes

    def _get_describe_labels(self):
        return self.describe_labels

    def _get_prior(self, label):
        '''
        Calculates prior (probability) of specified label
        Arguments:
            label: Index value of label whose prior probability to calculate
        Returns prior probability of specified label
        '''
        return self._get_describe_labels()[label]

    def _get_likelihood(self, label, attribute, x):
        '''
        Calculate likelihood (probability) of attribute
        Arguments:
            label: Index value of label whose attribute's gaussian distribution to calculate
            attribute: Index value of attribute whose gaussian distribution to calculate
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

    def _get_posterior(self, label, X):
        '''
        Calculate posterior (probability) of label for X values
        Arguments:
            label: Index value of label whose posterior to calculate
            X: List of real values whose posterior to estimate
        Returns: A posterior estimate of the X values
        '''

        posterior = self._get_prior(label)
        for i in range(len(X)):
            posterior *= self._get_likelihood(label, i, X[i])

        return posterior

    def predict_prob(self):
        return self.predict_prob

    def fit(self, X, Y):
        '''
        Fits training data on target data using a Naive Bayes algorithm
        Arguments:
            X: Training data with array-like shape with n samples and m attributes
            Y: Target data with array-like shape with n samples
        '''

        # Verify X and Y are Series, DataFrame or Numpy Array objects
        if not isinstance(X, (pd.Series, pd.DataFrame, np.ndarray)) or not isinstance(Y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                f'X and Y must be Series, DataFrame or Numpy Array object but X is {type(X)} and Y is {type(Y)}')
        # Verify X and Y have identical number of rows
        elif X.shape[0] != Y.shape[0]:
            raise ValueError(
                f'X and Y must have identical number of rows but X has shape {X.shape} and Y has shape {Y.shape}')

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
        '''
        Predicts class based on specified attributes
            X: 1-D list of attribute values used to predict class
        '''

        # Verify X is a Series, DataFrame or Numpy Array object
        if not isinstance(X, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                f'X must be a Series, DataFrame or Numpy Array object but X is {type(X)}')
        # Verify X and training data have identical number of rows
        elif X.shape != (model._get_data().shape[1] - 1, 1) and X.shape != (model._get_data().shape[1] - 1, ):
            raise ValueError(
                f'X must have number of rows identical to number of columns of training data but X has shape {X.shape} and training data has shape {self._get_data().iloc[:,0:-1].shape}')

        predict_prob = {}
        max_label = 0
        max_posterior = 0

        # Iterate through each label in fitted data
        for label in self._get_labels():
            # Calculate label's posterior and assign probability value to predict_prob with label as key
            predict_prob[label] = self._get_posterior(label, X)
            # Select highest posterior
            if predict_prob[label] > max_posterior:
                max_posterior = predict_prob[label]
                max_label = label

        # Save predict_prob for future reference
        self.predict_prob = predict_prob

        return max_label

    def plot(self, X, Y, attributes=[0, 1], h=0.1):
        '''
        Plots heat map of naive bayes probabilities for data with two attributes:
            X: Training data with array-like shape with n samples and m attributes
            Y: Target data with array-like shape with n samples
            attributes: A list of length 2 with integer values that referince which columns in training data to include in analysis
            h: # Step size in the mesh
        '''

        # Verify X and Y are Series, DataFrame or Numpy Array objects
        if not isinstance(X, (pd.Series, pd.DataFrame, np.ndarray)) or not isinstance(Y, (pd.Series, pd.DataFrame, np.ndarray)):
            raise TypeError(
                f'X and Y must be Series, DataFrame or Numpy Array object but X is {type(X)} and Y is {type(Y)}')
        # Verify X and Y have identical number of rows
        elif X.shape[0] != Y.shape[0]:
            raise ValueError(
                f'X and Y must have identical number of rows but X has shape {X.shape} and Y has shape {Y.shape}')
        # Verify attribute is a list with correct length and valur types
        elif type(attributes) != list or len(attributes) != 2 or not all(isinstance(n, int) for n in attributes):
            raise TypeError(
                f'Attributes must be a list of length 2 and should only contain integer values')
        # Verify h is of float type
        elif type(h) == 'float64':
            raise TypeError(f'h should be float data type but it is {type(h)}')

        print("This make time some time.  Decrease size of X and Y or h arguments to increase processing time.")

        # Minimum and maximum values of x and y coordinates in mesh grid
        x_min, x_max = X.iloc[:, attributes[0]].min(
        ) - 1, X.iloc[:, attributes[0]].max() + 1
        y_min, y_max = X.iloc[:, attributes[1]].min(
        ) - 1, X.iloc[:, attributes[1]].max() + 1

        # Range of values for x and y coordinates in mesh grid
        x_range = np.arange(x_min, x_max, h)
        y_range = np.arange(y_min, y_max, h)

        # Set of x and y coordinates in mesh grid
        x_coord, y_coord = np.meshgrid(x_range, y_range)
        coords = np.c_[x_coord.ravel(), y_coord.ravel()]

        y_ = np.arange(y_min, y_max, h)

        self.fit(X[attributes], Y)

        # Generate probability predictions for all coordinates in mesh grid
        predictions = []
        count = 0
        for coord in coords:
            prediction = self.predict(coord)
            predictions.append(self.predict_prob[prediction])
            print(count)
            count += 1
        Z = np.array(predictions).reshape(x_coord.shape)

        # Necessary in case function is run in a jupyter notebook
        def enable_plotly_in_cell():
            import IPython
            from plotly.offline import init_notebook_mode
            display(IPython.core.display.HTML(
                '''<script src="/static/components/requirejs/require.js"></script>'''))
            init_notebook_mode(connected=False)
        enable_plotly_in_cell()

        # Generate heat map and scatterplot
        trace1 = go.Heatmap(x=x_coord[0],
                            y=y_,
                            z=Z,
                            colorscale='Jet',
                            showscale=False)
        trace2 = go.Scatter(x=X[attributes[0]],
                            y=X[attributes[1]],
                            mode='markers',
                            showlegend=False,
                            marker=dict(size=10,
                                        color=Y_train,
                                        colorscale='Jet',
                                        reversescale=True,
                                        line=dict(color='black', width=1))
                            )
        layout = go.Layout(autosize=True,
                           title='Naive Bayes Gaussian Surface Probability Map',
                           hovermode='closest',
                           showlegend=False,
                           width=800, height=800)

        data = [trace1, trace2]
        fig = go.Figure(data=data, layout=layout)
        fig.show()
