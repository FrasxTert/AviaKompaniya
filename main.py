from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

data = pd.read_csv('data.csv')
data = pd.get_dummies(data, columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'], drop_first=True)

features = [col for col in data.columns if col != 'satisfaction']

X = data[features]
y = data['satisfaction']

indices = data.index

X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC: {roc_auc:.4f}')

test_indices = test_indices

submission = pd.DataFrame({
    'id': test_indices,
    'satisfaction': y_pred
})
submission.to_csv('submission.csv', index=False, header=True)
