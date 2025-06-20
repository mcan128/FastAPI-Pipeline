import pytest
import numpy as np
import pandas as pd
from ml.model import train_model, inference, save_model, load_model, compute_model_metrics
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "age": [25, 35],
        "workclass": ["Private", "Self-emp-not-inc"],
        "education": ["Bachelors", "Masters"],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Tech-support", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "native-country": ["United-States", "India"],
        "salary": [">50K", "<=50K"]
    })


@pytest.fixture
def cat_features():
    return [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]


def test_one(sample_data, cat_features):
    X, y, encoder, lb = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray), "Not a numpy array"
    

def test_two(sample_data, cat_features):
    X, y, _, _ = process_data(sample_data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Not a RandomForestClassifier model"


def test_three():
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    for metric in (precision, recall, fbeta):
        assert isinstance(metric, float), "Not a float"
        assert 0.0 <= metric <= 1.0, "Out of range"
    assert round(precision, 4) == 1.0
    assert round(recall, 4) == 0.6667
    assert round(fbeta, 4) == 0.8