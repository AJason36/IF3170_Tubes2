import joblib
import pandas as pd
import numpy as np

class KNN:
    data = None

    def __init__(self):
        self.data = None
        self.var_cols = None
        self.bool_cols = None
        self.num_cols = None

    def fit(self, df):
        cols = df.columns.values.tolist()
        self.var_cols = cols[:-1]
        self.bool_cols = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
        self.num_cols = self.var_cols.copy()
        for col in self.bool_cols:
            self.num_cols.remove(col)

        # df_outliers = self.getOutlierRows(df)
        # df = df.drop(df_outliers)

        self.data = df

    def predict(self, test, k, method):
        if (self.data is None) :
            print('Please fit data first')
            return 
        result = test.copy()
        test_without_label = test.copy()
        test_without_label.drop("price_range",axis=1)
        predictions = []
        for i in range(len(test_without_label)):
            current = test_without_label.iloc[i]
            prediction = self.get_prediction(current, k, method)
            predictions.append(prediction)
        result["predictions"] = predictions
        diff = 0
        check = []
        for i in range(len(result)):
            if(result.iloc[i]["predictions"] != result.iloc[i]["price_range"]):
                check.append(i)
                diff+=1
            # print(diff)
        accuracy = (len(result)-diff)/len(result) * 100
        print(accuracy)

    def get_prediction(self, validY, k, method):
        train_without_label = self.data.copy()
        train_without_label.drop("price_range",axis=1)
        distances = self.calculate_distance(method,train_without_label,validY)
        final_result = self.data.copy()
        final_result["distance"] = distances
        # print(final_result)
        sorted_result = final_result.sort_values(by="distance", ascending = True)
        # print(sorted_result[:k])
        k_neighbors = sorted_result[:k]
        result = self.get_majority(k_neighbors)
        # print(result)
        return result

    # Menghitung jarak antara kedua data dengan menggunakan euclidean distance
    def euclidean_distance(self, train, test):
        train_array = train.to_numpy()
        test_array = test.to_numpy()
        total_distance = 0
        for i in range(len(test_array)):
            distance = (train_array[i] - test_array[i])**2
            total_distance += distance
        return np.sqrt(total_distance)

    # Menghitung jarak antara kedua data dengan menggunakan manhattan distance
    def manhattan_distance(self, train,test):
        train_array = train.to_numpy()
        test_array = test.to_numpy()
        total_distance = 0
        for i in range(len(test_array)):
            distance = np.abs(train_array[i] - test_array[i])
            total_distance += distance
        return total_distance

    # Menghitung jarak antar kedua data dengan
    def minkowski_distance(self, train, test, p=2):
        return np.sum(np.abs(train-test)**p)**(1/p)

    def calculate_distance(self, method, train, target):
        distances = []
        for i in range(len(train)):
            current_row = train.iloc[i]
            distance = self.euclidean_distance(current_row, target)
            distances.append(distance)
        return distances

    def get_majority(self, dataset):
        return dataset["price_range"].mode()[0]

    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

train = pd.read_csv("data_train.csv")
valid = pd.read_csv("data_validation.csv")
model = KNN()
model = KNN.load_model('knn.joblib')
model.predict(valid, 19, "euclidean")
# model.fit(train)
# model.save_model('knn.joblib')