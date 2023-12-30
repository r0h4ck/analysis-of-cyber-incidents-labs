import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats
from sklearn import metrics

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.tree import DecisionTreeClassifier


def read_dataset(filename):
    data = pd.read_csv(filename, encoding = "latin-1")
    if filename == "input/Mall_Customers.csv":
        feature_names = {
            "CustomerID": "id",
            "Genre": "sex",
            "Age": "age",
            "Annual Income (k$)": "income",
            "Spending Score (1-100)": "score",
        }

    else:
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


def main():
    random.seed(10)
    np.random.seed(10)

    pd.set_option("display.max_colwidth", None)
    filename = "input/Mall_Customers.csv"
    data = read_dataset(filename)

    data.info()

    X = data.loc[:, ["income", "score"]].values
    y = data["score"]
    print(f"X shape: {X.shape}\n")

    nn = NearestNeighbors(n_neighbors = 2)
    neighbors = nn.fit(X)
    distances, indices = neighbors.kneighbors(X)

    distances = np.sort(distances, axis = 0)
    distances = distances[:, 1]
    plt.figure(figsize = (7, 5))
    plt.plot(distances)
    plt.title("Sorted distances")
    plt.xlabel("Index")
    plt.ylabel("Distance")
    plt.show()

    eps = 8
    min_samples = 2 * X.shape[1]
    print(f"Epsilon: {eps}")
    print(f"Min samples: {min_samples}\n")

    dbscan = DBSCAN(eps = eps, min_samples = min_samples)
    dbscan.fit(X)
    labels = dbscan.labels_
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}\n")

    plt.figure(figsize = (10, 10))
    for label in np.unique(labels):
        plt.scatter(X[labels == label, 0], X[labels == label, 1], label = label)
    plt.xlabel("Income")
    plt.ylabel("Spending Score")
    plt.legend(title = "Cluster Labels", bbox_to_anchor = (1, 1), loc = "upper left")
    plt.show()

    nmi = metrics.normalized_mutual_info_score(y, labels)
    print(f"Mutual info score:  {nmi}\n\n")

    print("-----------------------------------------\n")

    filename = "input/diabetes.csv"
    data = read_dataset(filename)
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(["outcome"], axis = 1))
    y = data["outcome"]

    score_list_A = []
    score_list_B = []
    SAMPLE_SIZE = 50
    max_depth = 5
    for i in range(SAMPLE_SIZE):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = i)

        model_A = DecisionTreeClassifier(max_depth = max_depth, criterion = "gini", random_state = i)
        model_A.fit(X_train, y_train)
        prediction = model_A.predict(X_test)
        score = accuracy_score(y_test, prediction)
        score_list_A.append(score)

        model_B = DecisionTreeClassifier(max_depth = max_depth, criterion = "entropy", random_state = i)
        model_B.fit(X_train, y_train)
        prediction = model_B.predict(X_test)
        score = accuracy_score(y_test, prediction)
        score_list_B.append(score)

    t_stat, p_value = stats.ttest_ind(score_list_A, score_list_B, equal_var = False)
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")


if __name__ == "__main__":
    main()
