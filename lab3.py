import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def read_dataset(filename):
    data = pd.read_csv(filename, encoding = "latin-1")
    feature_names = {
        "Pregnancies": "pregnancies",
        "Glucose": "glucose",
        "BloodPressure": "pressure",
        "SkinThickness": "thickness",
        "Insulin": "insulin",
        "BMI": "bmi",
        "DiabetesPedigreeFunction": "dpf",
        "Age": "age",
        "Outcome": "outcome",
    }
    data = data.rename(columns = feature_names)
    return data


def print_heatmap(data):
    corr = data.corr()
    plt.figure(figsize = (7, 7))
    sns.heatmap(corr, annot = True, fmt = ".2f")
    plt.yticks(rotation = 0)
    plt.subplots_adjust(left = 0.17, right = 1, top = 0.95, bottom = 0.2)
    plt.show()


def print_importance(clf, X_train):
    coeff = list(clf.coef_[0])
    labels = list(X_train.columns)
    features = pd.DataFrame()
    features["Features"] = labels
    features["importance"] = coeff
    features.sort_values(by = ["importance"], ascending = True, inplace = True)
    features["positive"] = features["importance"] > 0
    features.set_index("Features", inplace = True)

    plt.figure(figsize = (7, 7))
    features.importance.plot(kind = "barh", figsize = (11, 6), color = features.positive.map({True: "blue", False: "red"}))
    plt.xlabel("Importance")
    plt.show()


def main():
    pd.set_option("display.max_colwidth", None)
    filename = "input/diabetes.csv"
    data = read_dataset(filename)

    data.info()
    print_heatmap(data)

    X = data.drop(["outcome"], axis = 1)
    y = data["outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include = [np.number]).columns.tolist()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    clf = LogisticRegression(max_iter = 1000, random_state = 10)
    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    print("\n\nScores:")
    print(classification_report(y_test, predicted))
    print(confusion_matrix(y_test, predicted))

    print_importance(clf, X_train)


if __name__ == "__main__":
    main()
