import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Load dataset
data = pd.read_csv("C:/Users/diva1/OneDrive/Documents/task 2/car data.csv")  # Replace with actual dataset file

print("\nDataset Information:\n")
print(data.info())

print("\nSummary Statistics:\n")
print(data.describe())

print("\nMissing Values:\n")
print(data.isnull().sum())

print("\nExample Data")
print(data.head)

# Feature Engineering
data["Car_Age"] = 2024 - data["Year"]
data.drop(columns=["Year", "Car_Name"], inplace=True)  # Dropping less useful columns

# Define categorical and numerical columns
categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
numerical_features = ["Present_Price", "Driven_kms", "Car_Age", "Owner"]

# Data Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Splitting dataset
X = data.drop(columns=["Selling_Price"])
y = data["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Define Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, verbose=1)

# Model Evaluation
y_pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R²): {r2:.2f}")

# Visualization: Selling Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data["Selling_Price"], bins=10, kde=True, color="blue")
plt.xlabel("Selling Price (Lakhs)")
plt.ylabel("Count")
plt.title("Distribution of Car Selling Prices")
plt.show()

# Actual vs Predicted Prices
plt.figure(figsize=(8, 5))

# Compute absolute error
error = abs(y_test - y_pred)

# Create scatter plot with colormap
scatter = plt.scatter(y_test, y_pred, c=error, cmap="coolwarm", alpha=0.7)

# Add colorbar
plt.colorbar(scatter, label="Absolute Error")
plt.xlabel("Actual Selling Price (Lakhs)")
plt.ylabel("Predicted Selling Price (Lakhs)")
plt.title("Actual vs Predicted Selling Prices (Colored by Error)")
plt.show()


# Feature Importance Analysis


# Create SHAP explainer
explainer = shap.Explainer(model, X_train)

# Get SHAP values for test data
shap_values = explainer(X_test)

# Convert to mean importance
shap_importance = abs(shap_values.values).mean(axis=0)

# Create DataFrame for visualization
feature_names = numerical_features + list(preprocessor.transformers_[1][1].get_feature_names_out())
feature_importance = pd.DataFrame({"Feature": feature_names, "Importance": shap_importance})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
plt.title("SHAP Feature Importance in Car Price Prediction")
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=10, kde=True, color="red")
plt.xlabel("Residuals (Errors)")
plt.ylabel("Count")
plt.title("Residual Distribution")
plt.show()

# Correlation Heatmap
numeric_data = data.select_dtypes(include=["number"])
plt.figure(figsize=(8, 5))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss Curve")
plt.legend()
plt.show()

# Line Plot: Actual vs Predicted Selling Prices
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test, label="Actual Price", linestyle="-", marker="o", color="blue", alpha=0.7)
plt.plot(range(len(y_test)), y_pred, label="Predicted Price", linestyle="-", marker="x", color="red", alpha=0.7)

plt.xlabel("Car Index")
plt.ylabel("Selling Price (Lakhs)")
plt.title("Actual vs Predicted Selling Prices")
plt.legend()
plt.show()
 


# Function to Predict Selling Price
def predict_car_price(present_price, driven_kms, fuel_type, selling_type, transmission, owner, car_age):
    input_data = pd.DataFrame({
        "Present_Price": [present_price],
        "Driven_kms": [driven_kms],
        "Fuel_Type": [fuel_type],
        "Selling_type": [selling_type],
        "Transmission": [transmission],
        "Owner": [owner],
        "Car_Age": [car_age]
    })
    input_data = preprocessor.transform(input_data)
    predicted_price = model.predict(input_data)[0][0]
    return predicted_price

# Example Prediction
predicted_price = predict_car_price(5.0, 30000, "Petrol", "Dealer", "Manual", 0, 5)
print(f"Predicted Selling Price: {predicted_price:.2f} Lakhs")
