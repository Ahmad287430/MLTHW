import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import warnings
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # لإخفاء التحذيرات إن أردت

def preprocess_data(path='loan_prediction.csv'):
    data = pd.read_csv(path)
    
    # Handle missing values
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'] = data['Dependents'].replace('3+', '4')
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)

    # Drop Loan_ID and separate target
    data.drop('Loan_ID', axis=1, inplace=True)
    data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Balance the classes
    majority = data[data['Loan_Status'] == 1]
    minority = data[data['Loan_Status'] == 0]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    data_balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    y = data_balanced['Loan_Status']
    X = data_balanced.drop('Loan_Status', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns


def evaluate_model(model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(model, X, y, cv=kf, scoring=scoring)

    mean_scores = {metric: results[f'test_{metric}'].mean() for metric in scoring}

    print(f"\n📊 Model: {model.__class__.__name__}")
    for metric, score in mean_scores.items():
        print(f"{metric.capitalize()}: {score:.4f}")

    # 🔒 حفظ النتائج إلى ملف metrics.pkl
    joblib.dump(mean_scores, 'metrics.pkl')
    print("\n✅ Metrics saved to 'metrics.pkl'")

    return mean_scores['f1']  # إعادة F1 فقط للمقارنة


def tune_model(model, param_dist, X, y):
    # Adjust n_iter based on number of combinations
    total_combinations = 1
    for v in param_dist.values():
        total_combinations *= len(v)
    
    n_iter = min(10, total_combinations)

    rs = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=5, scoring='f1', random_state=42)
    rs.fit(X, y)
    return rs.best_estimator_


# --- Main Execution ---
def main():
    X, y, scaler, feature_names = preprocess_data()

    # استخراج أسماء الأصناف بشكل آمن
    class_names = np.unique(y)

    models = {
        "LogisticRegression": (
            LogisticRegression(),
            {"C": [0.01, 0.1, 1, 10]}
        ),
        "DecisionTree": (
            DecisionTreeClassifier(),
            {"max_depth": [3, 5, 10, None]}
        ),
        "RandomForest": (
            RandomForestClassifier(),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5, 10]
            }
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(),
            {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7]}
        ),
        "XGBoost": (
            XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5, 7]}
        )
    }

    best_models = {}
    model_scores = {}

    for name, (model, params) in models.items():
        print(f"\n--- Tuning and Evaluating: {name} ---")
        best_model = tune_model(model, params, X, y)
        f1 = evaluate_model(best_model, X, y)

        # التنبؤ بكامل البيانات باستخدام cross-validation
        y_pred = cross_val_predict(best_model, X, y, cv=5)

        # إنشاء مصفوفة الالتباس
        cm = confusion_matrix(y, y_pred)

        print("Confusion Matrix:")
        print(cm)

        # تقرير الأداء الكامل
        print("\nClassification Report:")
        print(classification_report(y, y_pred))

        # رسم مصفوفة الالتباس
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name}")
        plt.show()

        best_models[name] = best_model
        model_scores[name] = f1

    # أفضل نموذج حسب F1 score
    best_model_name = max(model_scores, key=model_scores.get)
    final_model = best_models[best_model_name]

    print(f"\n✅ النموذج الأفضل هو: {best_model_name} بـ F1 Score = {model_scores[best_model_name]:.4f}")

    # حفظ النموذج والمحولات
    joblib.dump(final_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    print("\n✅ Model saved as 'best_model.pkl'")

    return final_model, scaler, feature_names

# Run training
if __name__ == "__main__":
    model, scaler, columns = main()

