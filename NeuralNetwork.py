# Importing the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from numpy import genfromtxt
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


class NeuralNetwork:
    def __init__(self):
        self.sc = StandardScaler()
        print("A Neural Network has been created")
        self.PATH_TO_RESULTS = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classifier = KerasClassifier(build_fn=self.build_classifier, batch_size=3000, epochs=300)

    def train(self, results_loc, shuffle=False):
        # Importing the dataset
        self.PATH_TO_RESULTS = results_loc # all sets of training data in separate lists
        training = glob.glob(self.PATH_TO_RESULTS + "train/*/*.csv")
        dev = glob.glob(self.PATH_TO_RESULTS + "dev/*/*.csv")

        # --- all training data added to a nd array
        train_data = []
        for file in range(len(training)):  # loop through list of files to process          # using train & dev set
            file_data = genfromtxt(training[file], delimiter=',')
            train_data.append(file_data)
        for file in range(len(dev)):  # loop through list of files to process
            file_data = genfromtxt(dev[file], delimiter=',')
            train_data.append(file_data)

        for file in range(len(train_data)):
            if file is 0:  # create the ndarray
                data_for_training = train_data[file]
            else:  # start stacking them
                data_for_training = np.vstack([data_for_training, train_data[file]])

        # assigned to X_train
        self.X_train = data_for_training[:, :-1]  # this is the training data
        self.y_train = data_for_training[:, -1]
        if shuffle:
            # shuffle data in arrays
            self.X_train = self.X_train
        self.X_train = self.sc.fit_transform(self.X_train)
        accuracies = cross_val_score(estimator=self.classifier,
                                     X=self.X_train,
                                     y=self.y_train,
                                     cv=10,  # cv is how many folds you want to use
                                     n_jobs=-1)     # what CPU to use for calculations, -1 means all CPUs runs calc's in
                                                    # parallel
        mean = accuracies.mean()
        var = accuracies.std()
        print("Training results:\n"
              "Mean is: %.2f \n"
              "Variance is: %.2f" % (mean, var))

    def test(self, results_loc=None):
        if results_loc is None:
            test = glob.glob(self.PATH_TO_RESULTS + "test/*/*.csv")         # returns list of csv files
        else:
            self.PATH_TO_RESULTS = results_loc
            test = glob.glob(self.PATH_TO_RESULTS + "*.csv")
        # --- all test data added to a nd array
        test_data = []
        test = test[:-1]
        for file in range(len(test)):  # loop through list of files to process
                file_data = genfromtxt(test[file], delimiter=',')
                test_data.append(file_data)     # list of nd_arrays from each file
        for file in range(len(test_data)):
            if file is 0:  # create the nd array
                data_for_test = test_data[file]
            else:  # start stacking them
                data_for_test = np.vstack([data_for_test, test_data[file]])
        self.X_test = data_for_test[:, :-1]
        self.y_test = data_for_test[:, -1]
        self.X_test = self.sc.transform(self.X_test)
        # Make a prediction
        self.classifier.fit(self.X_train, self.y_train)
        y_pred = self.classifier.predict(self.X_test)
        y_pred = (y_pred > 0.5)

        cm = confusion_matrix(self.y_test, y_pred)

        print(cm)
        print(self.X_test.shape[0])
        print("Accuracy of training predictions: ", (cm[0, 0] + cm[1, 1]) / self.X_test.shape[0])

    def predict_file(self, path_to_result):
        prediction_data = genfromtxt(path_to_result, delimiter=',')
        X_predict = prediction_data[:, :-1]
        y_pred = self.classifier.predict(X_predict)
        return y_pred


    def build_classifier(self):
        classifier = Sequential()
        classifier.add(Dense(units=(int(self.X_train.shape[1] / 2)),
                             kernel_initializer='uniform',
                             activation='relu',
                             input_dim=(self.X_train.shape[1])))  # adding input layer and first hidden layer
        classifier.add(Dropout(rate=0.4))
        classifier.add(Dense(units=1,
                             kernel_initializer='uniform',
                             activation='sigmoid'))
        classifier.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        return classifier

