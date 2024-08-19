import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

train_data = pd.read_csv("/content/drive/MyDrive/data science/Aviakompaniya/train_dataset.csv")
test_data = pd.read_csv("/content/drive/MyDrive/data science/Aviakompaniya/test_dataset.csv")

print(train_data.info())
print(train_data.describe())

sns.heatmap(train_data.isnull(), cbar=False, cmap="viridis")
plt.show()

train_data_encoded = pd.get_dummies(train_data, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], drop_first=True)

for col in train_data.columns:
    if train_data[col].dtype == 'object':
        sns.countplot(x=col, data=train_data)
        plt.show()
    else:
        sns.histplot(train_data[col], kde=True)
        plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(train_data_encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

train_data = train_data_encoded
test_data = pd.get_dummies(test_data, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], drop_first=True)

features = [col for col in train_data.columns if col != 'satisfaction']
X_train = train_data[features]
y_train = train_data['satisfaction']

X_test = test_data[features]

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(model):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
    print(f"ROC AUC: {np.mean(auc_scores):.4f}")

print("RandomForestClassifier")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(rf_model)

print("GradientBoostingClassifier")
gb_model = GradientBoostingClassifier(random_state=42)
evaluate_model(gb_model)

print("LogisticRegression")
lr_model = LogisticRegression(random_state=42)
evaluate_model(lr_model)

print("Support Vector Machine")
svm_model = SVC(probability=True, random_state=42)
evaluate_model(svm_model)

final_model = rf_model
final_model.fit(X_train_scaled, y_train)

y_pred = final_model.predict(X_test_scaled)

submission = pd.DataFrame({
    'id': test_data['id'],
    'satisfaction': y_pred
})
submission.to_csv('/content/drive/MyDrive/data science/Aviakompaniya/aviakomp.csv', index=False, header=True)

print("Classification Report:")
print(classification_report(y_train, final_model.predict(X_train_scaled)))

y_pred_proba = final_model.predict_proba(X_train_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC AUC: {roc_auc_score(y_train, y_pred_proba):.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
