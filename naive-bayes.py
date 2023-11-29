import joblib
import pandas as pd
import numpy as np
import math

class NaiveBayes:
    def __init__(self):
        self.statisticsDict = None
        self.targetDict = None
        self.countDict = None
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

        df_outliers = self.getOutlierRows(df)
        df = df.drop(df_outliers)
        print(len(df))

        self.statisticsDict = self.makeColumnsStatisticsDictionary(df)
        self.targetDict = self.seperateByTarget(df)
        self.countDict = self.makeColumnProbabilityDictionary(df, self.targetDict)
        
    def predict(self, df):
        # Implement your prediction logic here

        error_count = 0
        indexList = df.index.tolist()
        actualTarget = df['price_range'].values.tolist()
        totalData = len(df)

        for i in range(totalData):
            rowData = df.iloc[i][:-1]
            prob = [1,1,1,1] #sorted as the index, for target val 0, 1, 2, 3

            # Probability of target column
            for k in range(len(prob)):
                prob[k] *= self.targetDict[k] / totalData

            # Probability of other dependent column
            for j in range(len(rowData)):
                column = self.var_cols[j]
                if (column in self.num_cols):
                    for k in range(len(prob)):
                        x = rowData[j]
                        mean = self.statisticsDict[column][k]['mean']
                        std = self.statisticsDict[column][k]['std']
                        numProbability = self.calculateNumerical(x, mean, std)
                        prob[k] = prob[k] * numProbability

                elif (column in self.bool_cols):
                    for k in range(len(prob)):
                        countColGivenTarget = self.countDict[column][rowData[j]][k]
                        countTarget = self.targetDict[k]
                        boolProbability = countColGivenTarget/countTarget
                        prob[k] = prob[k] * boolProbability


            # print(indexList[i] , "  ", prob)
            max = 0
            for p in range(len(prob)):
                if (prob[p] > prob[max]):
                    max = p

            
            if (max != actualTarget[i]):
                print(indexList[i], " ", max, " ", actualTarget[i])
                error_count +=1

        print("Error count: ", error_count)

    def getOutlierRows(self, df):
        threshold = 1.5
        outliers_res = []

        for col in self.num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            IQR = q3 - q1

            outliers = df[(df[col] < q1 - threshold * IQR) | (df[col] > q3 + threshold * IQR)]
            for i in (outliers.index.tolist()):
                if i not in outliers_res:
                    outliers_res.append(i)

        return outliers_res


    def makeColumnsStatisticsDictionary(self, df):
        dictionary = dict()
        for col in self.num_cols:
            dictionary[col] = dict()
            smallerDf = df[[col, 'price_range']]
            mean = smallerDf.groupby(['price_range'])[col].mean()
            std = smallerDf.groupby(['price_range'])[col].std()

            indexList = mean.index.tolist()
            meanList = mean.values.tolist()
            stdList = std.values.tolist()

            for i in range(len(indexList)):
                dictionary[col][indexList[i]] = dict()
                dictionary[col][indexList[i]]['mean'] = meanList[i]
                dictionary[col][indexList[i]]['std'] = stdList[i]
        return dictionary

    def makeColumnProbabilityDictionary(self, df, targetDict):
        dictionary = dict()
        for col in self.bool_cols:
            dictionary[col] = dict()
            smallerDf = df[[col, 'price_range']]
            temp = smallerDf.groupby([ col, 'price_range'])['price_range'].count()

            indexList = temp.index.tolist()
            valuesList = temp.values.tolist()
            length = len(indexList)

            currVal = None
            prevVal = None
            for i in range(length):
                colValue = indexList[i][0]
                targetValue = indexList[i][1]
                currVal = colValue
                if (currVal != prevVal):
                    dictionary[col][colValue] = dict()
                dictionary[col][colValue][targetValue] = valuesList[i]
                prevVal = currVal

        return dictionary

    def calculateNumerical(self, x, mean, std):
        leftpart = 1 / (std * math.sqrt(2 * math.pi))
        exppart = (-1/2) * (((x - mean)/std)**2)
        rightpart = math.exp(exppart)
        return leftpart * rightpart

    def seperateByTarget(self, df):
        container = dict()

        count = df['price_range'].value_counts()

        indexList = count.index.tolist()
        countList = count.tolist()

        for i in range(len(indexList)):
            container[indexList[i]] = countList[i]

        return container

    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

train = pd.read_csv("data_train.csv")
valid = pd.read_csv("data_validation.csv")
model = NaiveBayes.load_model('naive.joblib')
model.predict(valid)
# model = NaiveBayes()
# model.fit(train)
# model.save_model('naive.joblib')