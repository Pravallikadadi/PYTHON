import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1️⃣ Load dataset
data = pd.read_csv(r"C:\Users\Pravallika\OneDrive\Desktop\PYTHON PROJECTS\loan.csv")

# 2️⃣ Handle missing values
data.ffill(inplace=True)  # updated from fillna(method='ffill')

# 3️⃣ Fix '3+' in Dependents column
data['Dependents'] = data['Dependents'].replace('3+', 3)
data['Dependents'] = data['Dependents'].astype(int)

# 4️⃣ Encode categorical variables
label_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
le = LabelEncoder()
for col in label_cols:
    data[col] = le.fit_transform(data[col])

# 5️⃣ Split features and target
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# 6️⃣ Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7️⃣ Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8️⃣ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9️⃣ Predictions
y_pred = model.predict(X_test)

# 🔟 Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 1️⃣1️⃣ Predict new applicant
new_applicant = pd.DataFrame([[1, 0, 0, 1, 0, 5000, 0, 200, 360, 1, 2]], columns=X.columns)
new_applicant_scaled = scaler.transform(new_applicant)
prediction = model.predict(new_applicant_scaled)
print("Loan Status Prediction (0=No, 1=Yes):", prediction[0])

