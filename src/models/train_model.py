"""
Model training module for Metro Bike Share analysis.

This module contains functions to train demand prediction models.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def prepare_features(df, top_n_stations=10):
    """
    Prepare features for demand prediction models.

    Args:
        df (pd.DataFrame): DataFrame with trip data
        top_n_stations (int): Number of top stations to include in model

    Returns:
        tuple: (X_with_station, y, feature_names, station_list)
    """
    logging.info(f"Preparing features for demand prediction model...")

    # Aggregate data for hourly demand modeling
    hourly_demand = (
        df.groupby(["start_station", pd.Grouper(key="start_time", freq="H")])
        .size()
        .reset_index(name="demand")
    )

    # Add temporal features to hourly demand
    hourly_demand["hour"] = hourly_demand["start_time"].dt.hour
    hourly_demand["day_of_week"] = hourly_demand[
        "start_time"
    ].dt.dayofweek  # 0=Monday, 6=Sunday
    hourly_demand["month"] = hourly_demand["start_time"].dt.month
    hourly_demand["is_weekend"] = hourly_demand["day_of_week"].isin([5, 6]).astype(int)
    hourly_demand["is_rush_hour"] = (
        hourly_demand["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    )
    hourly_demand["is_business_hours"] = (
        hourly_demand["hour"].between(9, 17).astype(int)
    )

    # Add cyclical time features (to handle the circular nature of time)
    hourly_demand["hour_sin"] = np.sin(2 * np.pi * hourly_demand["hour"] / 24)
    hourly_demand["hour_cos"] = np.cos(2 * np.pi * hourly_demand["hour"] / 24)
    hourly_demand["day_sin"] = np.sin(2 * np.pi * hourly_demand["day_of_week"] / 7)
    hourly_demand["day_cos"] = np.cos(2 * np.pi * hourly_demand["day_of_week"] / 7)

    # Select top stations for modeling to reduce complexity
    top_stations = (
        df["start_station"].value_counts().nlargest(top_n_stations).index.tolist()
    )
    station_demand = hourly_demand[hourly_demand["start_station"].isin(top_stations)]

    logging.info(f"Selected {len(top_stations)} stations with highest demand")
    logging.info(
        f"Total number of hourly demand records for modeling: {len(station_demand)}"
    )

    # Select features and target
    feature_names = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_rush_hour",
        "is_business_hours",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]

    X = station_demand[feature_names]
    y = station_demand["demand"]

    # Encode the start_station categorical variable
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    station_encoded = encoder.fit_transform(station_demand[["start_station"]])
    station_encoded_df = pd.DataFrame(
        station_encoded,
        columns=[f"station_{i}" for i in range(station_encoded.shape[1])],
        index=station_demand.index,
    )

    # Combine numerical and one-hot encoded features
    X_with_station = pd.concat(
        [X.reset_index(drop=True), station_encoded_df.reset_index(drop=True)], axis=1
    )

    return X_with_station, y, feature_names, top_stations


def train_demand_model(X, y, model_type="gradient_boosting"):
    """
    Train a demand prediction model.

    Args:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable
        model_type (str): Type of model to train ('linear', 'random_forest', or 'gradient_boosting')

    Returns:
        tuple: (model_pipeline, results_dict, training_time)
    """
    logging.info(f"Training {model_type} model...")
    start_time = time.time()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Testing data shape: {X_test.shape}")

    # Get feature names
    numeric_features = [col for col in X.columns if not col.startswith("station_")]
    station_features = [col for col in X.columns if col.startswith("station_")]

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "pass",
                "passthrough",
                station_features,
            ),  # Station features are already one-hot encoded
        ]
    )

    # Select model based on type
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Train model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation score
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    cv_mean = cv_scores.mean()

    # Training time
    duration = time.time() - start_time

    # Create results dictionary
    results = {
        "model_type": model_type,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "cv_mean": cv_mean,
        "training_time": duration,
        "feature_names": numeric_features + station_features,
        "test_predictions": y_pred,
        "test_actual": y_test,
    }

    logging.info(
        f"Model training complete - R² = {r2:.4f}, CV R² = {cv_mean:.4f}, RMSE = {rmse:.2f}"
    )

    return pipeline, results


def extract_feature_importance(model_pipeline, feature_names):
    """
    Extract feature importance from a trained model.

    Args:
        model_pipeline (Pipeline): Trained model pipeline
        feature_names (list): List of feature names

    Returns:
        pd.DataFrame: DataFrame with feature importances
    """
    model = model_pipeline.named_steps["model"]

    # Extract importances based on model type
    if hasattr(model, "feature_importances_"):
        # Tree-based models have feature_importances_
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear models have coefficients
        importances = np.abs(model.coef_)
    else:
        return pd.DataFrame(
            {"feature": feature_names, "importance": np.ones(len(feature_names))}
        )

    # Create DataFrame
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    return importance_df


def save_model(model_pipeline, results, model_dir, model_name):
    """
    Save trained model and results to disk.

    Args:
        model_pipeline (Pipeline): Trained model pipeline
        results (dict): Dictionary with model results
        model_dir (Path): Directory to save model files
        model_name (str): Base name for model files

    Returns:
        dict: Dictionary with file paths
    """
    # Create model directory if it doesn't exist
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / f"{model_name}.joblib"
    joblib.dump(model_pipeline, model_path)

    # Save results
    results_path = model_dir / f"{model_name}_results.joblib"
    joblib.dump(results, results_path)

    # Save feature importance if available
    importance_path = model_dir / f"{model_name}_importance.csv"
    feature_names = results["feature_names"]
    importance_df = extract_feature_importance(model_pipeline, feature_names)
    importance_df.to_csv(importance_path, index=False)

    # Save paths
    paths = {
        "model": model_path,
        "results": results_path,
        "importance": importance_path,
    }

    logging.info(f"Model saved to {model_path}")
    return paths


def evaluate_models(df, model_dir, top_n_stations=10):
    """
    Train and evaluate multiple models for comparison.

    Args:
        df (pd.DataFrame): Input data
        model_dir (Path): Directory to save model files
        top_n_stations (int): Number of top stations to model

    Returns:
        pd.DataFrame: Model comparison results
    """
    logging.info("Starting model evaluation...")

    # Prepare features
    X, y, feature_names, top_stations = prepare_features(df, top_n_stations)

    # Define models to test
    model_types = ["linear", "random_forest", "gradient_boosting"]

    # Train and evaluate each model
    results_list = []
    for model_type in model_types:
        logging.info(f"Evaluating {model_type} model...")
        model_pipeline, results = train_demand_model(X, y, model_type)

        # Save model
        save_model(
            model_pipeline,
            results,
            model_dir,
            f"demand_prediction_{model_type.replace('_', '')}",
        )

        # Add to results list for comparison
        results_list.append(
            {
                "Model": model_type.replace("_", " ").title(),
                "R-Squared": results["r2"],
                "CV R-Squared": results["cv_mean"],
                "RMSE": results["rmse"],
                "MAE": results["mae"],
                "Training Time (s)": results["training_time"],
            }
        )

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_list)

    # Save comparison
    comparison_path = Path(model_dir) / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    logging.info(f"Model evaluation complete. Results saved to {comparison_path}")
    return comparison_df
