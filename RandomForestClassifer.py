import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# load dataset
FILE_PATH = 'cc_institution_details.csv'
data = pd.read_csv(FILE_PATH)

# remove rows with missing values in 'awards_per_value'
features = ['chronname', 'state', 'control', 'level', 'fte_value', 'exp_award_value', 'hbcu', 'flagship', 'awards_per_value']
df = data[features].dropna(subset=['awards_per_value'])

# drop the 'chronname' column (unneeded for RandomForest)
df = df.drop(columns=['chronname'])

# convert boolean columns to binary
df['hbcu'] = (df['hbcu'] == 'X').astype(int)
df['flagship'] = (df['flagship'] == 'X').astype(int)

# create a binary target: top 10% in awards_per_value
threshold = df['awards_per_value'].quantile(0.90)
df['is_top10'] = (df['awards_per_value'] >= threshold).astype(int)

# define features and target variable
x = df.drop(columns=['awards_per_value', 'is_top10'])
y = df['is_top10']

# split the data into train and test sets (80/20)
x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# split training into training and validation 25 percent of training will be used for validation
x_train, X_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)


# preprocessing - 
# OneHotEncoder: categorical features
# StandardScaler: numerical features
category_features = ['state', 'control', 'level', 'hbcu', 'flagship']
numerical_features = ['fte_value', 'exp_award_value']

# preprocessing step
preprocessor = ColumnTransformer(
    transformers=[
        ('category', OneHotEncoder(drop='first', handle_unknown='ignore'), category_features),
        ('numerical', StandardScaler(), numerical_features)
    ])

# preprocess then classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# define the hyperparameter grid for RandomizedSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200, 300, 500],
    'classifier__max_depth': [None, 10, 20, 30, 40],
    'classifier__min_samples_split': [2, 5, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 5, 10],
    'classifier__max_features': ['sqrt', 'log2', 5, 10],
    'classifier__oob_score': [True, False],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__class_weight': ['balanced', None],
    'classifier__max_samples': [None, 0.8],
}

# RandomizedSearchCV with 5-fold cross-validation
grid_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=40,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# fit RandomizedSearchCV on the training data
grid_search.fit(x_train, y_train)

# print the best parameters and best score from RandomizedSearchCV
print("Best parameters:")

for param in grid_search.best_params_:
    print(param, grid_search.best_params_[param])

print("Best cross-validation score:", grid_search.best_score_)

# evaluate on the validation and test sets
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(x_test)


# print the classification report for detailed performance metrics
print("Validation Set Results:\n")
print(classification_report(y_val, y_val_pred))
print("Test Set Results:\n")
print(classification_report(y_test, y_test_pred))


# generate and plot the confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Top 10", "Top 10"])
cmd.plot(cmap="Blues", values_format='d')

plt.title("Confusion Matrix")
plt.show()