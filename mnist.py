import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from sklearn import metrics
import os
import cv2
from zipfile import ZipFile

classifier_file_path = "clf.p"
mnist784_zip_file_path = "mnist_784.zip"

def load_mnist():
    all_data = pd.read_csv(mnist784_zip_file_path, compression='zip')
    X = all_data.drop(columns='class')
    y = all_data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, train_size = 0.2 ,random_state = 10)

    return X_train, X_test, y_train, y_test 


def load_classiffier():
    if os.path.isfile(classifier_file_path):
        with open(classifier_file_path, "rb") as f:
            return pickle.load(f)
    else:
        return None

def fit_classifier():
    X_train, X_test, y_train, y_test = load_mnist()
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

    with open(classifier_file_path, 'wb+') as f:
        pickle.dump(classifier, f)

    return classifier


def invert(im):
    return (255 - im)


def detect_edges(grayImg):  # Edge Detection, returns image array
    minInt, maxInt, _, _ = cv2.minMaxLoc(grayImg)  # Grayscale: MinIntensity, Max, and locations
    beam = cv2.mean(grayImg)  # Find the mean intensity in the img pls.
    mean = float(beam[0])
    CannyOfTuna = cv2.Canny(grayImg, (mean + minInt) / 2,
                            (mean + maxInt) / 2)  # Finds edges using thresholding and the Canny Edge process.
    return CannyOfTuna


def recognize_digits_from_image(image_path):
    recognized_digits = []

    # classifier = load_classiffier()
    if True:#not classifier:
        classifier = fit_classifier()

    # Read the input image
    im = invert(cv2.imread(image_path))
    
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    _, image_threshold = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours in the image
    ctrs, _ = cv2.findContours(detect_edges(image_threshold.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(contour) for contour in ctrs]

    # For each rectangular region, calculate HOG features and predict the digit using Linear SVM.
    for rect in rects:
        # Draw the rectangles
        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        center_x = x + int(w / 2)
        center_y = y + int(h / 2)
        length = max(w, h) * 1.4
        start_x = center_x - int(length / 2)
        finish_x = center_x + int(length / 2)
        start_y = center_y - int(length / 2)
        finish_y = center_y + int(length / 2)
        roi = image_threshold[start_y:finish_y, start_x:finish_x]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        roi = roi.reshape(1, -1)
        nbr = classifier.predict(roi)
        recognized_digits.append(int(nbr[0]))
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    
    cv2.imwrite('output.png', im)
    return recognized_digits


if __name__ =="__main__":
    print(recognize_digits_from_image("1 2 3.png"))
