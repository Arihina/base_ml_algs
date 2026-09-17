from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from get_dataset import get_ds

train, x_train, y_train, valid, x_valid, y_valid, test, x_test, y_test = get_ds()

for i in range(1, 10, 2):
    print(f"n neighbors = {i}")
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(x_train, y_train)

    y_pred = knn_model.predict(x_test)
    print(classification_report(y_test, y_pred))
