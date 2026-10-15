from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from get_dataset import get_ds

train, x_train, y_train, valid, x_valid, y_valid, test, x_test, y_test = get_ds()

lg = LogisticRegression()
lg = lg.fit(x_train, y_train)

y_pred = lg.predict(x_test)
print(classification_report(y_test, y_pred))
