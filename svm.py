from sklearn.metrics import classification_report
from sklearn.svm import SVC

from get_dataset import get_ds

train, x_train, y_train, valid, x_valid, y_valid, test, x_test, y_test = get_ds()

svm = SVC()
svm = svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

print(classification_report(y_test, y_pred))
