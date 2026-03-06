from preprocess import preprocess_data
from model import train_models, predict
from evaluate import evaluate
from sklearn.model_selection import train_test_split

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    X, y, selected_features = preprocess_data('student_data.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    train_models(X_train, y_train)

    print("Naive Bayes Evaluation:")
    y_pred_nb = predict('naive_bayes.pkl', X_test)
    evaluate(y_test, y_pred_nb)

    print("\nLogistic Regression Evaluation:")
    y_pred_lr = predict('logistic_regression.pkl', X_test)
    evaluate(y_test, y_pred_lr)
