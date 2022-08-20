# knn_optimizer_framework_rev.py

from sklearn.neighbors import KNeighborsClassifier
from statistics import mean
import pandas as pd
from itertools import combinations
import pickle
from sklearn.model_selection import cross_val_score


class knn_framework():
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.X = X
        self.y = y
        self.optimal_features_names = None
        self.optimal_feature_score = None
        self.optimal_features_combination = None
        self.optimal_n = None
        self.optimal_n = None
        self.optimal_n_score = None

    def __repr__(self) -> str:
        if self.optimal_features is not None and self.optimal_n is not None:
            s = str(self.X.head) + '\n\n' + str(self.y.head()) + '\n\n' + \
                str(self.optimal_features) + '\n\n' + str(self.optimal_n)
            return s
        elif self.optimal_features is not None:
            s = s = str(self.X.head) + '\n\n' + str(self.y.head()
                                                    ) + '\n\n' + str(self.optimal_features)
            return s
        elif self.optimal_n is not None:
            s = str(self.X.head) + '\n\n' + str(self.y.head()) + \
                '\n\n' + str(self.optimal_n)
            return s
        s = str(self.X.head()) + '\n\n' + str(self.y.head())
        return s

    def feature_optimizer(self) -> list:
        results = list()
        X = self.X
        y = self.y
        feature_size = len(X.columns)

        for step in range(1, feature_size - 1):
            print(f'combining features with feature size of: {step}')
            for combination in list(combinations(range(0, feature_size), r=step)):
                combination = list(combination)
                if combination:
                    X_step = X.iloc[:, combination]
                    model = KNeighborsClassifier().fit(X_step, y)
                    score = mean(cross_val_score(model, X_step, y))
                    feature_names = [X.columns[i] for i in combination]
                    result = (combination, score, feature_names)
                    results.append(result)

        optimal_result = max(results, key=lambda x: x[1])

        print(
            f"\noptimal feature combination: {optimal_result[2]}\n score: {round(optimal_result[1]*100,2)}\n")

        self.optimal_features_combination = optimal_result[0]
        self.optimal_feature_score = optimal_result[1]
        self.optimal_features_names = optimal_result[2]

        return results

    def n_optimizer(self) -> list:
        results = list()
        X = self.X
        y = self.y

        if self.optimal_features_combination is None:
            print('Optimal features not found, using all features\n')
            n_max = int(len(X) / 2)
            for n in range(1, n_max):
                model = KNeighborsClassifier(
                    n_neighbors=n).fit(X, y)
                score = mean(cross_val_score(model, X, y))
                result = (n, score)
                results.append(result)
        else:
            X = X.iloc[:, self.optimal_features_combination]
            print(
                f'Optimizing n for optimal features: {self.optimal_features_names}\n')
            n_max = int(len(X) / 2)
            for n in range(1, n_max):
                model = KNeighborsClassifier(
                    n_neighbors=n).fit(X, y)
                score = mean(cross_val_score(model, X, y))
                result = (n, score)
                results.append(result)

        optimal_result = max(results, key=lambda x: x[1])
        optimal_score = optimal_result[1]
        optimal_n = optimal_result[0]
        print(
            f'\noptimal n: {optimal_n}\nscore: {round(optimal_score*100,2)}\n')
        self.optimal_n = optimal_n
        self.optimal_n_score = optimal_score
        return results

    def final_score(self) -> list:
        if self.optimal_features_combination is None:
            print('Optimal features have not yet been found for this dataset.\n')
            return []
        elif self.optimal_n is None:
            print('Optimal n-neighbors n has not yet been found for this dataset.\n')
            return []
        else:
            X = self.X.iloc[:, self.optimal_features_combination]
            y = self.y
            model = KNeighborsClassifier(
                n_neighbors=self.optimal_n).fit(X, y)
            result = mean(cross_val_score(model, X, y))
            print(f'final score: {round(result * 100, 2)}\n')
            return result

    def save_model(self, file_name: str = 'knn_model') -> KNeighborsClassifier:
        X = self.X
        y = self.y
        file_name += '.pkl'
        if self.optimal_features_combination is not None:
            X = X.iloc[:, self.optimal_features_combination]
        if self.optimal_n is not None:
            model = KNeighborsClassifier(
                n_neighbors=self.optimal_n).fit(X, y)
        else:
            model = KNeighborsClassifier().fit(X, y)

        with open(file_name, 'wb') as f:
            pickle.dump(model, f)
        print(f'model saved to {file_name}')

        return model


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    data = load_digits(as_frame=True)
    data_X = data.data
    data_y = data.target
    framework = knn_framework(data_X, data_y)
    framework.feature_optimizer()
    framework.n_optimizer()
    framework.final_score()
    framework.save_model('wine_model.pkl')
