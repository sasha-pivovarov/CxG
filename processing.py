import pandas as pd
import numpy as np
from sklearn import linear_model
import graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("tagged2.csv")
print(data.shape)
data = data[data["VERB_DEP"].notnull()].astype(str)
counts_data = data.groupby("TARGET").nunique()
print(data.shape)
print(counts_data.head())

# sns.stripplot(x="TARGET", y="VERB_DEP", data=counts_data, jitter=True)

enc = LabelEncoder()
X = data.loc[:, data.columns != "TARGET"]
# X = X.loc[:, data.columns != "Unnamed: 0"]
for colname in X.columns:
    X[colname] = enc.fit_transform(X[colname])

X.drop(X.columns[[0, 1]], axis=1, inplace=True)
y = data["TARGET"]

chi2data = (chi2(X, y))
anovadata = f_classif(X, y)
print("Done")
statframe = pd.DataFrame({"chi2":chi2data[0], "chi2_pval":chi2data[1], "anova":anovadata[0], "anova_pval":anovadata[1]}, index=X.columns)
#statframe = statframe.drop(statframe.index[[0, 1]])

statframe.to_csv("stats.csv")
forest_classifier = RandomForestClassifier(n_estimators=50)
forest_classifier.fit(X, y)
print(X.columns)
plt.xlabel(list(X.columns))
plt.plot(forest_classifier.feature_importances_)
plt.show()

tree_classifier = DecisionTreeClassifier(criterion="entropy")
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
tree_classifier.fit(X_train, y_train)
pred = tree_classifier.predict(X_test)
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))

dot_data = export_graphviz(tree_classifier, out_file=None,
                         feature_names=list(X_train.columns),
                         class_names=["over", "about"],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree.png")