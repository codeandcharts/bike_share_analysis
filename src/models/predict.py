"""
Prediction module for Metro Bike Share analysis.

This module contains functions to generate predictions from trained models.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime, timedelta


def load_model(model_path):
    """
    Load a trained model from disk.

    Args:
        model_path (Path): Path to the saved model file

    Returns:
        object: Loaded model
    """
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


def prepare_prediction_features(start_time, station_id, top_stations):
    """
    Prepare features for a single prediction time point and station.

    Args:
        start_time (datetime): Prediction timestamp
        station_id (int): Station ID to predict demand for
        top_stations (list): List of top station IDs used in the model

    Returns:
        pd.DataFrame: DataFrame with features for prediction
    """
    # Create temporal features
    features = {
        "hour": start_time.hour,
        "day_of_week": start_time.weekday(),  # 0=Monday, 6=Sunday
        "month": start_time.month,
        "is_weekend": 1 if start_time.weekday() >= 5 else 0,
        "is_rush_hour": 1 if start_time.hour in [7, 8, 9, 16, 17, 18] else 0,
        "is_business_hours": 1 if 9 <= start_time.hour <= 17 else 0,
    }

    # Add cyclical time features
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
    features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

    # Create dataframe
    df = pd.DataFrame([features])

    # Add station one-hot encoding
    if station_id in top_stations:
        # Find the index of the station in the list (minus 1 because we drop the first in OneHotEncoder)
        station_idx = top_stations.index(station_id)
        if station_idx > 0:  # Adjust for the dropped first category
            station_idx -= 1

        # Add station columns
        for i in range(len(top_stations) - 1):  # -1 because we drop the first category
            col_name = f"station_{i}"
            df[col_name] = 1 if i == station_idx else 0
    else:
        # Station not in training data, set all to 0
        for i in range(len(top_stations) - 1):
            col_name = f"station_{i}"
            df[col_name] = 0
        logging.warning(
            f"Station {station_id} not in top stations list, using zeros for encoding"
        )

    return df


def predict_demand(model, start_time, station_id, top_stations):
    """
    Predict demand for a specific time and station.

    Args:
        model: Trained model pipeline
        start_time (datetime): Time to predict for
        station_id (int): Station ID to predict for
        top_stations (list): List of top station IDs used in the model

    Returns:
        float: Predicted demand (number of rides)
    """
    # Prepare features
    features_df = prepare_prediction_features(start_time, station_id, top_stations)

    # Make prediction
    try:
        prediction = model.predict(features_df)[0]
        # Ensure prediction is non-negative
        prediction = max(0, prediction)
        return prediction
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return 0


def predict_next_day(model, reference_date, station_id, top_stations):
    """
    Generate hourly predictions for the next 24 hours.

    Args:
        model: Trained model pipeline
        reference_date (datetime): Reference date (predictions start from next hour)
        station_id (int): Station ID to predict for
        top_stations (list): List of top station IDs used in the model

    Returns:
        pd.DataFrame: DataFrame with hourly predictions
    """
    # Start from the next hour
    start_hour = reference_date.replace(minute=0, second=0, microsecond=0) + timedelta(
        hours=1
    )

    # Generate 24 hourly predictions
    results = []
    for i in range(24):
        prediction_time = start_hour + timedelta(hours=i)
        predicted_demand = predict_demand(
            model, prediction_time, station_id, top_stations
        )

        results.append(
            {
                "prediction_time": prediction_time,
                "hour": prediction_time.hour,
                "day_of_week": prediction_time.strftime("%A"),
                "station_id": station_id,
                "predicted_demand": predicted_demand,
            }
        )

    return pd.DataFrame(results)


def predict_rebalancing_needs(
    model, reference_date, stations_df, top_stations, threshold=5
):
    """
    Predict stations that will need rebalancing in the next day.

    Args:
        model: Trained model pipeline
        reference_date (datetime): Reference date
        stations_df (pd.DataFrame): DataFrame with station information
        top_stations (list): List of top station IDs used in the model
        threshold (float): Demand threshold to flag for rebalancing

    Returns:
        pd.DataFrame: DataFrame with rebalancing recommendations
    """
    results = []

    # Only predict for stations in the top stations list
    station_ids = [s for s in stations_df["station_id"].unique() if s in top_stations]

    for station_id in station_ids:
        # Get station information
        station_info = stations_df[stations_df["station_id"] == station_id].iloc[0]

        # Predict next day
        predictions = predict_next_day(model, reference_date, station_id, top_stations)

        # Identify morning peak (7-9 AM)
        morning_predictions = predictions[predictions["hour"].isin([7, 8, 9])]
        morning_peak = morning_predictions["predicted_demand"].max()

        # Identify evening peak (4-7 PM)
        evening_predictions = predictions[predictions["hour"].isin([16, 17, 18, 19])]
        evening_peak = evening_predictions["predicted_demand"].max()

        # Calculate dock capacity
        total_docks = station_info.get("total_docks", 0)

        # Determine if rebalancing is needed
        morning_rebalance = morning_peak > threshold
        evening_rebalance = evening_peak > threshold

        results.append(
            {
                "station_id": station_id,
                "station_name": station_info.get(
                    "station_name", f"Station {station_id}"
                ),
                "total_docks": total_docks,
                "morning_peak_demand": morning_peak,
                "evening_peak_demand": evening_peak,
                "morning_rebalance_needed": morning_rebalance,
                "evening_rebalance_needed": evening_rebalance,
                "priority": "High"
                if (morning_peak > threshold * 2 or evening_peak > threshold * 2)
                else "Medium"
                if (morning_rebalance or evening_rebalance)
                else "Low",
            }
        )

    # Create DataFrame and sort by priority
    rebalance_df = pd.DataFrame(results)
    rebalance_df["priority_rank"] = rebalance_df["priority"].map(
        {"High": 1, "Medium": 2, "Low": 3}
    )
    rebalance_df = rebalance_df.sort_values(
        ["priority_rank", "morning_peak_demand", "evening_peak_demand"],
        ascending=[True, False, False],
    )
    rebalance_df = rebalance_df.drop(columns=["priority_rank"])

    return rebalance_df


def batch_predict(
    model_path, stations_df, prediction_date=None, top_n_stations=10, output_path=None
):
    """
    Generate batch predictions for all stations.

    Args:
        model_path (Path): Path to the saved model
        stations_df (pd.DataFrame): DataFrame with station information
        prediction_date (datetime, optional): Date to predict for (defaults to current time)
        top_n_stations (int): Number of top stations used in the model
        output_path (Path, optional): Path to save predictions

    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    logging.info("Starting batch prediction...")

    # Load model
    model = load_model(model_path)
    if model is None:
        return None

    # Set default prediction date to current time if not provided
    if prediction_date is None:
        prediction_date = datetime.now()

    # Get top stations by trip count
    top_stations = (
        stations_df.sort_values("trips", ascending=False)["station_id"]
        .head(top_n_stations)
        .tolist()
    )

    # Predict rebalancing needs
    rebalancing_df = predict_rebalancing_needs(
        model, prediction_date, stations_df, top_stations
    )

    # Save predictions if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rebalancing_df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")

    return rebalancing_df
