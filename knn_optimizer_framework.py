from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import combinations
import pickle


class knn_framework:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.X = X
        self.y = y
        self.optimal_features = None
        self.optimal_n = None
        return None

    def __repr__(self) -> str:
        if self.optimal_features is not None and self.optimal_n is not None:
            s = (
                str(self.X.head)
                + "\n\n"
                + str(self.y.head())
                + "\n\n"
                + str(self.optimal_features)
                + "\n\n"
                + str(self.optimal_n)
            )
            return s
        elif self.optimal_features is not None:
            s = s = (
                str(self.X.head)
                + "\n\n"
                + str(self.y.head())
                + "\n\n"
                + str(self.optimal_features)
            )
            return s
        elif self.optimal_n is not None:
            s = (
                str(self.X.head)
                + "\n\n"
                + str(self.y.head())
                + "\n\n"
                + str(self.optimal_n)
            )
            return s
        s = str(self.X.head()) + "\n\n" + str(self.y.head())
        return s

    def data_split(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.5, random_state=0
        )
        return (X_train, X_test, y_train, y_test)

    def feature_optimizer(self) -> list:
        results = list()
        X_train, X_test, y_train, y_test = self.data_split()
        feature_size = len(X_train.columns)
        for step in range(1, feature_size - 1):
            print(f"combining features with feature size of: {step}")
            for combination in list(combinations(range(0, feature_size), r=step)):
                combination = list(combination)
                if combination:
                    X_train_temp = X_train.iloc[:, combination]
                    X_test_temp = X_test.iloc[:, combination]
                    model = KNeighborsClassifier().fit(X_train_temp, y_train)
                    result = (
                        step,
                        combination,
                        model.score(X_test_temp, y_test),
                        [X_train.columns[i] for i in combination],
                    )
                    results.append(result)
        best_result = max(results, key=lambda x: x[2])
        print(
            f"\noptimal feature combination: {best_result[3]}\n score: {round(best_result[2]*100,2)}\n"
        )
        self.optimal_features = best_result

        # print(f'\n{sorted(results, key=lambda x : (x[2], x[0]), reverse=False)}')
        return results

    def n_optimizer(self):
        results = list()
        X_train, X_test, y_train, y_test = self.data_split()
        if self.optimal_features is None:
            print("Optimal features not found, using all features\n")
            n_max = int(len(X_train) / 2)
            for n in range(1, n_max):
                model = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
                result = (n, model.score(X_test, y_test))
                results.append(result)
        else:
            X_train = X_train.iloc[:, self.optimal_features[1]]
            X_test = X_test.iloc[:, self.optimal_features[1]]
            print(f"Optimizing n for optimal features: {self.optimal_features[3]}\n")
            n_max = int(len(X_train) / 2)
            for n in range(1, n_max):
                model = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
                result = (n, model.score(X_test, y_test))
                results.append(result)
        best_result = max(results, key=lambda x: x[1])
        print(f"\noptimal n: {best_result[0]}\nscore: {round(best_result[1]*100,2)}\n")
        self.optimal_n = best_result
        return results

    def final_score(self):
        if self.optimal_features is None:
            print("Optimal features have not yet been found for this dataset.\n")

        elif self.optimal_n is None:
            print("Optimal n-neighbors n has not yet been found for this dataset.\n")
        else:
            X = self.X.iloc[:, self.optimal_features[1]]
            y = self.y
            model = KNeighborsClassifier(n_neighbors=self.optimal_n[0]).fit(X, y)
            result = model.score(X, y)
            print(f"final score: {round(result * 100, 2)}\n")
            return result

    def save_model(self):
        X = self.X
        y = self.y
        if self.optimal_features is not None:
            X = X.iloc[:, self.optimal_features[1]]
        if self.optimal_n is not None:
            model = KNeighborsClassifier(n_neighbors=self.optimal_n[0]).fit(X, y)
        else:
            model = KNeighborsClassifier().fit(X, y)
        with open("knn_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("model saved to knn_model.pkl")
        return model


if __name__ == "__main__":
    from sklearn.datasets import load_wine

    data = load_wine(as_frame=True)
    data_X = data.data
    data_y = data.target
    framework = knn_framework(data_X, data_y)
    framework.feature_optimizer()
    framework.n_optimizer()
    framework.final_score()
    framework.save_model()
