import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


# 1 Load the data.
digits = load_digits()

# figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))
# for item in zip(axes.ravel(), digits.images, digits.target):
#     axes, image, target = item
#     axes.imshow(image, cmap=plt.cm.gray_r)
#
#     axes.set_xticks([])
#     axes.set_yticks([])
#
#     axes.set_title(target)
#
# plt.tight_layout()
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    random_state=11, train_size=0.75, test_size=0.25)

print(X_train.shape)
print(X_test.shape)

knn = KNeighborsClassifier(weights="distance", metric="minkowski", n_jobs=-1)

pipe = Pipeline([("knn", knn)])

search_space = [{"knn__n_neighbors": range(1, 12)}]

knn_classifier = GridSearchCV(pipe, search_space, cv=5, scoring="accuracy").fit(X_train, y_train)

best_k = knn_classifier.best_estimator_.get_params()['knn__n_neighbors']

print(f"The neighborhood size k is {best_k}")

predicted_knn = knn_classifier.predict(X=X_test)
expected_knn = y_test

print(predicted_knn[:20])
print(expected_knn[:20])

svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

predicted_svm = svm_classifier.predict(X=X_test)
expected_svm = y_test

print(predicted_svm[:20])
print(expected_svm[:20])

nb_classifier = MultinomialNB()

nb_classifier.fit(X_train, y_train)

predicted_nb = nb_classifier.predict(X=X_test)
expected_nb = y_test

print(predicted_nb[:20])
print(expected_nb[:20])

print(f"{knn_classifier.score(X_test, y_test): .2%}")
print(f"{svm_classifier.score(X_test, y_test): .2%}")
print(f"{nb_classifier.score(X_test, y_test): .2%}")

confusion = confusion_matrix(y_true=expected_knn, y_pred=predicted_knn)

print(confusion)

names = [str(digit) for digit in digits.target_names]
print(classification_report(predicted_knn, expected_knn, target_names=names))


def processing_img(img_path: str) -> np.array:

    img = Image.open(img_path)

    img = img.convert("L")

    img = img.resize((8, 8), Image.Resampling.LANCZOS)

    img_array = np.array(img, dtype=np.float32)

    img_array *= (16/256)

    np.round(img_array, out=img_array)

    img_array = 16 - img_array

    img_array = img_array.reshape(1, -1)

    return img_array


number = processing_img("Number_images/2.jpg")

predicted_knn_digit = knn_classifier.predict(number)[0]
predicted_svm_digit = svm_classifier.predict(number)[0]
predicted_nb_digit = nb_classifier.predict(number)[0]

print(predicted_knn_digit, predicted_svm_digit, predicted_nb_digit)

# img_1 = Image.open("Number_images/2.jpg")
#
# img_1 = img_1.convert("L")
#
# img_1 = img_1.resize((8, 8), Image.Resampling.LANCZOS)
#
# img_1.show()
