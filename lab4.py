# 4. SVM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA


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


def print_hyperplane(X, y, C, gamma, kernel):
    pca = PCA(n_components = 2, random_state = 10)
    pca_result = pca.fit_transform(X)
    pca_data = pd.DataFrame({"Component 0": pca_result[:, 0], "Component 1": pca_result[:, 1]})
    X_train, X_test, y_train, y_test = train_test_split(pca_data, y, test_size = 0.2, random_state = 10)

    model = SVC(C = C, gamma = gamma, kernel = kernel)
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)

    hyperplane_coefs = model.coef_[0]
    intercept = model.intercept_
    print(f"\n\nHyperplane coefficients: {hyperplane_coefs}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    min_x = pca_data["Component 0"].min()
    max_x = pca_data["Component 0"].max()
    lim_x = (1.1 * min_x, 1.1 * max_x)

    min_y = pca_data["Component 1"].min()
    max_y = pca_data["Component 1"].max()
    lim_y = (1.1 * min_y, 1.1 * max_y)

    x_values = np.linspace(model.intercept_, 1.1 * max_x, 100)
    y_values = (-hyperplane_coefs[0] / hyperplane_coefs[1]) * x_values - (intercept / hyperplane_coefs[1])
    y_test[y_test.index[y_test != y_pred].tolist()] = -1

    plt.figure(figsize = (10, 6))
    pca_data_pred = pd.DataFrame({"Component 0": X_test["Component 0"], "Component 1": X_test["Component 1"], "outcome": y_test})
    scatter = sns.scatterplot(x = "Component 0", y = "Component 1", hue = "outcome", data = pca_data_pred, palette = "Set1")
    scatter.set_xlim(lim_x)
    scatter.set_ylim(lim_y)
    sns.move_legend(scatter, "upper left", bbox_to_anchor = (1, 1))
    plt.plot(x_values, y_values, label = "Hyperplane", color = "purple")
    plt.show()


def main():
    pd.set_option("display.max_colwidth", None)
    filename = "input/diabetes.csv"
    data = read_dataset(filename)

    data.info()

    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(["outcome"], axis = 1))
    y = data["outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

    svc = SVC(C = 1.0, gamma = 1.0, kernel = "rbf")
    svc.fit(X_train, y_train.ravel())
    y_pred = svc.predict(X_test)

    print(f"\n\nDefault parameters:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    param_grid = {"C": [0.01, 0.1, 1, 10, 100], "gamma": ["auto", "scale"], "kernel": ["linear", "rbf", "poly"]}
    grid = GridSearchCV(SVC(), param_grid, refit = True)
    grid.fit(X_train, y_train.ravel())
    grid_predictions = grid.predict(X_test)

    print(f"\n\nBest parameters: {grid.best_params_}")
    print(classification_report(y_test, grid_predictions))
    print(confusion_matrix(y_test, grid_predictions))

    C = grid.best_params_["C"]
    gamma = grid.best_params_["gamma"]
    kernel = grid.best_params_["kernel"]
    print_hyperplane(X, y, C, gamma, kernel)


if __name__ == "__main__":
    main()
