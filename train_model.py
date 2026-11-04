import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("dataset/stock_data.csv")

# Features
X = data[["Open", "High", "Low", "Volume", "Previous_Close"]]

# Target
y = data["Close"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("Model trained successfully!")
print("Mean Absolute Error:", round(mae, 2))

# Save model
joblib.dump(model, "stock_model.pkl")

print("Model saved as stock_model.pkl")