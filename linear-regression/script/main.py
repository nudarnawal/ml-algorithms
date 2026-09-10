import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv(r"C:\Users\user\Desktop\TKCL\Projects\ML\ml-algorithms\linear-regression\dataset\Student_Performance.csv")

df["Extracurricular Activities"] = df["Extracurricular Activities"].map({
    "Yes": 1,
    "No": 0
})

X = df[[
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced"
]]

# the target
y = df["Performance Index"]

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compare actual vs predicted
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("Actual vs Predicted:")
print(comparison)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_,
    "Absolute Coefficient": abs(model.coef_)
})

importance = importance.sort_values(
    by="Absolute Coefficient",
    ascending=False
)

print("\nFeature Importance:")
print(importance)
print("\nModel Evaluation:")
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print("Mean Absolute Percentage Error:", mape, "%")
print("Mean Absolute Error:", mae)
print("R² Score:", r2)
