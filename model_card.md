# Model Card

## Model Details
- This is a random forest classifier trained model that predicts whether an individuals salary is greater than 50k based on demographic features from the census dataset. The featues include education, workclass, occupation, marital status, and sex. One hot encoding is used for categorical features and label binarization.

## Intended Use
- This model is intended to show supervised maching learning workflows for binary classification.
- It is for educational purposes and not for deployment in production.
  
## Training Data
- The training data comes from the cesnsus.csv file, which is cleaned and prepared for modeling.
- A random 80% of the dataset was used for training.
  
## Evaluation Data
- The other 20% of the dataset was used as a test set.
- Evaluation was conducted on both the full dataset and on slices of the data, grouped by unique values within the features.
  
## Metrics
- The model was evaluated using precision, recall, and f1 score. The overall performance on the test set was 0.7419 for precision, 0.6384 for recall and 0.6863 f1 score.
- The slice performance is detailed in the slice_output text file. This shows where the model performs better or worse for different feature values.
- The values for a masters education are 0.8271 precision, 0.8551 recall and 0.8409 f1. For 10th grade education they are 0.4 precision, 0.1667 recall and 0.2353 f1.

## Ethical Considerations
- This model is trained on real world census data, however it may reflect or amplify biases related to income inequality, race, gender, or nationality.

## Caveats and Recommendations
- A lot of slices have really small populations causing unreliable metrics.
- The pipeline is best used for learning rather than real world decisions.
  