import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
# Load and preprocess the data
def load_and_preprocess_data(path='Training.csv'):
    # Load the data
    data = pd.read_csv(path)

    # Separate features and target
    X = data.drop('disease', axis=1)
    y = data['disease']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Normalize the features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y

# Train and evaluate model
def train_evaluate_model(X, y, model, model_name):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle imbalanced data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Train the model
    model.fit(X_train_balanced, y_train_balanced)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro') # Changed line
    recall = recall_score(y_test, y_pred, average='micro') # Changed line
    f1 = f1_score(y_test, y_pred, average='micro') # Changed line

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return model, accuracy, precision, recall, f1

# Feature importance analysis
def analyze_feature_importance(model, X):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = abs(model.coef_[0])
    else:
        print("Model doesn't provide feature importances.")
        return

    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and preprocess the data
    X, y = load_and_preprocess_data('Training.csv')  # Replace with your dataset

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier()
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        trained_model, accuracy, precision, recall, f1 = train_evaluate_model(X, y, model, name)
        results[name] = {'model': trained_model, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    # Find the best model
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']

    print(f"\nBest Model: {best_model_name}")
    print(f"F1-score: {results[best_model_name]['f1']:.4f}")

    # Analyze feature importance for the best model
    analyze_feature_importance(best_model, X)