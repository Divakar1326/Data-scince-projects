import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Creating the dataset (Replace this with reading from CSV if available)
data = ("C:/Users/diva1/OneDrive/Documents/task 3/Advertising.csv")
df = pd.read_csv(data)

print("\nDataset Information:\n")
print(df.info())

print("\nSummary Statistics:\n")
print(df.describe())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nExample Data")
print(df.head())

  # Checking first few rows
sns.pairplot(df)
plt.show()

plt.figure(figsize=(15, 5))
sns.boxplot(data=df)
plt.title("Outliers in the df")
plt.show()

# Plot histograms for independent variables
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['TV'], kde=True, color='blue')
plt.title('Distribution of TV Advertising')

plt.subplot(1, 3, 2)
sns.histplot(df['Radio'], kde=True, color='green')
plt.title('Distribution of Radio Advertising')

plt.subplot(1, 3, 3)
sns.histplot(df['Newspaper'], kde=True, color='orange')
plt.title('Distribution of Newspaper Advertising')

plt.show()

# Correlation heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=1)
plt.title("Feature Correlation")
plt.show()

X = df[['TV', 'Radio', 'Newspaper']]  # Features
y = df['Sales']  # Target variable

# Splitting into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Define models and hyperparameter grids
models = {
    'Linear Regression': (LinearRegression(), None),
    'Random Forest': (RandomForestRegressor(random_state=42), {
        'Random Forest__n_estimators': [50, 100, 200],
        'Random Forest__max_depth': [None, 10, 20, 30],
        'Random Forest__min_samples_split': [2, 5, 10]
    }),
    'XGBoost': (XGBRegressor(random_state=42, objective='reg:squarederror'), {
        'XGBoost__n_estimators': [50, 100, 200],
        'XGBoost__learning_rate': [0.01, 0.1, 0.2],
        'XGBoost__max_depth': [3, 6, 10]
    })
}

# Train and evaluate models
results = {}

for model_name, (model, param_grid) in models.items():
    print(f"Training {model_name}...")
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        (model_name, model)
    ])

    if param_grid:
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2")
        grid_search.fit(X_train, y_train)

        # Evaluate the model
        y_pred = grid_search.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Store results
        results[model_name] = {
            "Best Parameters": grid_search.best_params_,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        }
    else:
        # No hyperparameter tuning for models without param_grid
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Store results
        results[model_name] = {
            "Best Parameters": "N/A",
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        }

# Sort and display results
sorted_results = sorted(results.items(), key=lambda x: x[1]["RMSE"])

print("\nModel Evaluation Results (Sorted by RMSE in Ascending Order):")
for model_name, result in sorted_results:
    print(f"{model_name}:")
    print(f"  Best Parameters: {result['Best Parameters']}")
    print(f"  MAE: {result['MAE']}")
    print(f"  MSE: {result['MSE']}")
    print(f"  RMSE: {result['RMSE']}")
    print(f"  R²: {result['R²']}")
    print()

# Display the best model with all metrics
best_model_name = sorted_results[0][0]
best_model_result = sorted_results[0][1]
print("\nBest Model:")
print(f"{best_model_name}:")
print(f"  Best Parameters: {best_model_result['Best Parameters']}")
print(f"  MAE: {best_model_result['MAE']}")
print(f"  MSE: {best_model_result['MSE']}")
print(f"  RMSE: {best_model_result['RMSE']}")
print(f"  R²: {best_model_result['R²']}")

#Evaluate the best model on the training set
if best_model_result["Best Parameters"] != "N/A":
    best_pipeline = GridSearchCV(Pipeline(steps=[
        ("scaler", StandardScaler()),
        (best_model_name, models[best_model_name][0])
    ]), param_grid=models[best_model_name][1], cv=5, scoring="r2")
    best_pipeline.fit(X_train, y_train)
else:
    best_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        (best_model_name, models[best_model_name][0])
    ])
    best_pipeline.fit(X_train, y_train)

train_predictions = best_pipeline.predict(X_train)
train_mae = mean_absolute_error(y_train, train_predictions)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, train_predictions)

print("Best Model Evaluation on Training Set:")
print(f"  MAE: {train_mae}")
print(f"  MSE: {train_mse}")
print(f"  RMSE: {train_rmse}")
print(f"  R²: {train_r2}")

# Evaluate the best model on the test set
test_predictions = best_pipeline.predict(X_test)
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, test_predictions)

print("Best Model Evaluation on Test Set:")
print(f"  MAE: {test_mae}")
print(f"  MSE: {test_mse}")
print(f"  RMSE: {test_rmse}")
print(f"  R²: {test_r2}")

new_data = pd.DataFrame({'TV': [200], 'Radio': [40], 'Newspaper': [50]})
predicted_sales = best_pipeline.predict(new_data)
print("Predicted Sales:", predicted_sales[0])

plt.figure(figsize=(8, 5))
# Compute absolute error
error = abs(y_test - y_pred)
# Create scatter plot with colormap
scatter = plt.scatter(y_test, y_pred, c=error, cmap="coolwarm", alpha=0.7,label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='dashed', linewidth=2, label="Ideal Fit")
plt.colorbar(scatter, label="Absolute Error")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.legend()
plt.show()
