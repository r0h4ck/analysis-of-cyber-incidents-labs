# 5. С4.5 та CART

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


def build_decision_tree(type, X_train, X_test, y_train, y_test):
    if type == "C4.5":
        criterion = "entropy"
    elif type == "CART":
        criterion = "gini"
    else:
        raise Exception("[-] Invalid type of decision tree is passed")

    max_depth = 3

    dt = DecisionTreeClassifier(max_depth = max_depth, criterion = criterion)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    print(f"\n\n{type} accuracy:    {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Visualization
    plt.figure(figsize = (10, 10))
    plot_tree(dt,
              feature_names = X_train.columns,
              class_names = ["0", "1"],
              filled = True)
    plt.show()


def main():
    random.seed(10)
    np.random.seed(10)

    pd.set_option("display.max_colwidth", None)
    filename = "input/diabetes.csv"
    data = read_dataset(filename)

    data.info()
    print_heatmap(data)

    X = data.drop(["outcome"], axis = 1)
    y = data["outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include = [np.number]).columns.tolist()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    build_decision_tree("C4.5", X_train, X_test, y_train, y_test)
    build_decision_tree("CART", X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
