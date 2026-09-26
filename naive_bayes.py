# Bayes Rule
# P(A | B) = P(B | A) * P (A) / P(B)
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

from get_dataset import get_ds

train, x_train, y_train, valid, x_valid, y_valid, test, x_test, y_test = get_ds()

model = GaussianNB()
model = model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))
