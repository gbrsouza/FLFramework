from numpy import average
from src.models.abstract_model import Model

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def evaluate_model(x_test, y_test, model:Model, classes=None):
    """Evaluate a model using a test dataset

    Args:
        x_test (list): A list of data to test the model
        y_test (list): A list of labels from data
        model (Model): The model to evaluate

    Returns:
        floats: return the accuracy, precision, recall, and confusion matrix
    """

    y_pred = model.predict(x_test)
    matrix = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    if classes != None:
        clr = classification_report(y_test, y_pred, target_names=classes, digits= 4)
        print("Classification Report:\n----------------------\n", clr)

    if model.get_type() == 'multiclass':
        average = 'macro'
    else:
        average = 'binary'
    
    pre = precision_score(y_test, y_pred, average=average)
    rec = recall_score(y_test, y_pred, average=average)
    return acc, pre, rec, matrix
