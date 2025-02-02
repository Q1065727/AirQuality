import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_and_preprocess_data(file_path): # Step 1 -> Read the csv and pre process the dataset

    df = pd.read_csv(file_path, delimiter=';')

    df_original = df[['country', 'location_name']].copy() # Save original dataset for merging later

    def convert_to_seconds(time_str): # Convert AM and PM times to seconds to calculate daylight duration
        if pd.isna(time_str) or time_str.strip() == "":
            return np.nan  # Handle missing values
        dt = pd.to_datetime(time_str, format="%I:%M %p")
        return dt.hour * 3600 + dt.minute * 60 + dt.second


    df["sunrise"] = df["sunrise"].astype(str).apply(convert_to_seconds) # Convert sunrise & sunset to total seconds
    df["sunset"] = df["sunset"].astype(str).apply(convert_to_seconds)

    df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce') # Convert last_updated column to datetime
    df["time_of_day"] = df["last_updated"].dt.hour * 3600 + df["last_updated"].dt.minute * 60 + df["last_updated"].dt.second
    df["daylight_duration"] = np.round((df["sunset"] - df["sunrise"]) / 3600, 1) # Compute daylight duration in hours


    df["is_daylight"] = (df["time_of_day"] >= df["sunrise"]) & (df["time_of_day"] <= df["sunset"]) # If daylight:1 if not 0
    df["is_daylight"] = df["is_daylight"].astype(int)

    # Extracting time based values
    df['year'] = df['last_updated'].dt.year
    df['month'] = df['last_updated'].dt.month
    df['day'] = df['last_updated'].dt.day
    df['hour'] = df['last_updated'].dt.hour

    df["condition_text"] = df["condition_text"].astype('category').cat.codes # Encode categorical weather conditions

    df = df.drop(["country", "location_name", "timezone", "last_updated"], axis=1, errors='ignore') # Drop all non numeric columns for model trainng

    print("\nData Types After Preprocessing:") # Print data types to verify all is numeric
    print(df.dtypes)

    return df, df_original

def split_data(df, target_column): # Step 2 -> Train test split (%20 of the data will be test)
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def select_important_features(rf_model, X_train): # Step 3 -> Important features selection
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("--- Feature Importance ---")
    print(feature_importances)

    # Adjust threshold for feature selection
    important_features = feature_importances[feature_importances['Importance'] > 0.005]['Feature']
    return important_features

def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test): # Step 4 -> Train and Evaluate Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train) # Train the model

    # Feature selection implementation
    important_features = select_important_features(rf_model, X_train)
    X_train_important = X_train[important_features]
    X_test_important = X_test[important_features]

    # Retrain the model on selected features
    rf_model.fit(X_train_important, y_train)
    y_pred = rf_model.predict(X_test_important)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Random Forest Model Performance ---")
    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ MAE: {mae:.4f}")
    print(f"✅ R²: {r2:.4f}")

    return rf_model, y_pred, rmse, mae, r2

def plot_actual_vs_predicted(y_test, y_pred, model_name): # Step 5 -> Visualize Actual vs Predicted Values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
    plt.xlabel("Actual PM2.5 Levels")
    plt.ylabel("Predicted PM2.5 Levels")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.legend()
    plt.show()

def check_outlier_severity(file_path): # Control Step -> Outlier severity check. If below 10% then do nothing.
    # Load dataset
    df = pd.read_csv(file_path, delimiter=';')

    # Air quaility columns
    pollution_cols = [
        "air_quality_Carbon_Monoxide",
        "air_quality_Ozone",
        "air_quality_Nitrogen_dioxide",
        "air_quality_Sulphur_dioxide",
        "air_quality_PM2.5",
        "air_quality_PM10"
    ]

    outlier_counts = {}

    # Count outliers with IQR method
    for col in pollution_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)

    # Calculate outlier percentage
    total_rows = len(df)
    total_outliers = sum(outlier_counts.values())
    outlier_percentage = (total_outliers / (total_rows * len(pollution_cols))) * 100

    # Print results
    print("\n*Outlier Summary")
    for col, count in outlier_counts.items():
        print(f"{col}: {count} outliers")

    print(f"\nTotal Outliers: {total_outliers}")
    print(f"Outlier Percentage: {outlier_percentage:.2f}% of the dataset")

    # Determine severity level
    if outlier_percentage > 10:
        print("\nCritical Outlier Level! More than 10% of data points are outliers.")
    else:
        print("\nOutliers are within an acceptable range. No critical issue detected")

def save_output_predictions(y_test, y_pred, rmse, mae, r2, output_file): # Step 6 -> Save output predictions csv file


    output_df = pd.DataFrame({  # Create DataFrame for actual vs predicted values
        "Actual_PM2.5": y_test.values,
        "Predicted_PM2.5": y_pred
    })


    metrics_df = pd.DataFrame({ # Create DataFrame for model metrics
        "Actual_PM2.5": ["Evaluation Metrics"],
        "Predicted_PM2.5": [""],
        "RMSE": [rmse],
        "MAE": [mae],
        "R²": [r2]
    })

    output_df = pd.concat([output_df, metrics_df], ignore_index=True) # Append metrics to the dataset

    output_df.to_csv(output_file, index=False) # Save to CSV

    print(f"\nOutput predictions and metrics saved to: {output_file}")

def save_edited_dataset(df, original_columns, output_file_edited_dataset): # Step 7 -> Save processed dataset

    edited_df = pd.concat([original_columns, df], axis=1) # Restore country & location_name to processed dataset

    edited_df.to_csv(output_file_edited_dataset, index=False)     # Save to CSV

    print(f"\nEdited dataset saved to: {output_file_edited_dataset}")

def select_important_features_gbr(gbr_model, X_train): # Gradient Boosting important feature selection
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': gbr_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\n--- Gradient Boosting Feature Importance ---")
    print(feature_importances)

    important_features = feature_importances[feature_importances['Importance'] > 0.005]['Feature']

    return important_features

def train_and_evaluate_gradient_boosting(X_train, X_test, y_train, y_test): # Step 8 -> Train and Evaluate Gradient Boosting
    gbr_model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    #Train the model
    gbr_model.fit(X_train, y_train)

    # Extract important features
    important_features = select_important_features_gbr(gbr_model, X_train)
    X_train_important = X_train[important_features]
    X_test_important = X_test[important_features]

    # Train the model again using only selected features
    gbr_model.fit(X_train_important, y_train)

    # Make predictions
    y_pred = gbr_model.predict(X_test_important)

    # Evaluate model performance
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Optimized Gradient Boosting Performance ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    return gbr_model, y_pred, rmse, mae, r2

def save_gradient_boosting_output(y_test, y_pred, rmse, mae, r2, output_file): # Step 9 -> Save GB predictions csv [SAME AS RANDOM FOREST]

    output_df = pd.DataFrame({
        "Actual_PM2.5": y_test.values,
        "Predicted_PM2.5": y_pred
    })

    metrics_df = pd.DataFrame({
        "Actual_PM2.5": ["Evaluation Metrics"],
        "Predicted_PM2.5": [""],
        "RMSE": [round(rmse, 3)],
        "MAE": [round(mae, 3)],
        "R²": [round(r2, 3)]
    })

    output_df = pd.concat([output_df, metrics_df], ignore_index=True)

    output_df.to_csv(output_file, index=False)

    print(f"\nGradient Boosting predictions and metrics saved to: {output_file}")

def RandomForestRun(): # Method to call and start random forest process
    file_path = '/Users/safaorhan/Downloads/AirDataset.csv'  # Referencing AirDatasetRawFile.csv here.
    df, original_columns = load_and_preprocess_data(file_path)
    target_column = 'air_quality_PM2.5'
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    rf_model, y_pred_rf, rmse, mae, r2 = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)
    plot_actual_vs_predicted(y_test, y_pred_rf, "Random Forest")
    output_file = "/Users/safaorhan/Documents/AirQuality/AirQuality_Predictions.csv"
    output_file_edited_dataset = "/Users/safaorhan/Documents/AirQuality/AirQuality_Edited.csv"
    save_edited_dataset(df, original_columns, output_file_edited_dataset)
    save_output_predictions(y_test, y_pred_rf, rmse, mae, r2, output_file)

def GBRun(): # Method to call and start gradient boosting process
    file_path = '/Users/safaorhan/Downloads/AirDataset.csv'  # Referencing AirDatasetRawFile.csv here.
    df, original_columns = load_and_preprocess_data(file_path)
    target_column = 'air_quality_PM2.5'
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    gbr_model, y_pred_gbr, rmse_gbr, mae_gbr, r2_gbr = train_and_evaluate_gradient_boosting(X_train, X_test, y_train, y_test)
    save_gradient_boosting_output(y_test, y_pred_gbr, rmse_gbr, mae_gbr, r2_gbr, "/Users/safaorhan/Documents/AirQuality/AirQuality_GB_Predictions.csv") 


RandomForestRun()
GBRun()
