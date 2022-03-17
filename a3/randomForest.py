from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Do not change anything in this code
X, y = make_classification(
    n_samples=1000,
    n_features=12,
    n_informative=4,
    n_redundant=0,
    n_classes=2,
    random_state=5,
    shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=43)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print(rfc.feature_importances_)
plt.bar([0,1,2,3,4,5,6,7,8,9,10,11], rfc.feature_importances_)
plt.show()
print(rfc.score(X_test, y_test))

