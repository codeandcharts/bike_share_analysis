#!/usr/bin/env python3
"""
Metro Bike Share Analysis Runner
================================

This script runs a complete analysis on Metro Bike Share data:
1. Loads prepared data from DuckDB
2. Performs exploratory data analysis
3. Creates key visualizations
4. Trains predictive models
5. Generates rebalancing recommendations

Run this script after `etl_pipeline.py` has processed the raw data.
"""

import logging
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Import modules from src package
from src.data import load_from_duckdb, clean_bike_share_data, create_merged_dataset

from src.visualization.visualize import (
    set_visualization_defaults,
    save_all_visualizations,
    plot_hourly_usage,
    plot_day_of_week,
    plot_seasonal_trends,
    plot_hourly_heatmap,
    plot_monthly_trends,
    plot_top_stations,
    plot_rider_segments,
)
from src.models.train_model import evaluate_models, prepare_features
from src.models.predict import batch_predict, load_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Metro Bike Share Analysis Runner")

    parser.add_argument(
        "--db-path",
        type=str,
        default="./data/processed/metro_bike_share.duckdb",
        help="Path to DuckDB database file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Path to output directory for visualizations and models",
    )

    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip creating visualizations",
    )

    parser.add_argument(
        "--skip-modeling", action="store_true", help="Skip training models"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    return parser.parse_args()


def setup_logging(log_level):
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    """Run the analysis."""
    # Parse arguments
    args = parse_args()

    # Set up logging
    setup_logging(args.log_level)

    # Convert string paths to Path objects
    DB_PATH = Path(args.db_path)
    OUTPUT_DIR = Path(args.output_dir)

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
    VISUALIZATIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR = OUTPUT_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)

    # Check if database exists
    if not DB_PATH.exists():
        logging.error(f"Database file not found: {DB_PATH}")
        logging.error("Please run etl_pipeline.py first")
        return 1

    logging.info("=== METRO BIKE SHARE ANALYSIS START ===")

    # Load data from DuckDB
    logging.info("Loading data from DuckDB")
    trips_df = load_from_duckdb(DB_PATH, "SELECT * FROM trips")
    stations_df = load_from_duckdb(DB_PATH, "SELECT * FROM stations")

    # Check if data was loaded successfully
    if trips_df.empty:
        logging.error("Failed to load trip data from DuckDB")
        return 1

    # Ensure datetime columns are in the correct format
    trips_df["start_time"] = pd.to_datetime(trips_df["start_time"])
    if "end_time" in trips_df.columns:
        trips_df["end_time"] = pd.to_datetime(trips_df["end_time"])

    # Create merged dataset
    logging.info("Creating merged analysis dataset")
    if stations_df is not None and not stations_df.empty:
        df_merged = create_merged_dataset(trips_df, stations_df)
    else:
        logging.warning("No station data available - using trip data only")
        df_merged = trips_df

    # Clean merged dataset
    logging.info("Cleaning merged dataset")
    df_merged = clean_bike_share_data(df_merged)

    # Save the cleaned data for future use
    clean_data_path = OUTPUT_DIR / "clean_data.parquet"
    df_merged.to_parquet(clean_data_path)
    logging.info(f"Saved cleaned data to {clean_data_path}")

    # Create visualizations
    if not args.skip_visualizations:
        logging.info("Creating visualizations")

        # Initialize visualization settings
        set_visualization_defaults()

        # Save all standard visualizations
        saved_paths = save_all_visualizations(df_merged, VISUALIZATIONS_DIR)
        logging.info(f"Created {len(saved_paths)} visualizations")

        # Display path to visualizations
        print(f"\n📊 Visualizations saved to: {VISUALIZATIONS_DIR}")

    # Train models
    if not args.skip_modeling:
        logging.info("Training predictive models")

        # Evaluate multiple models
        model_comparison = evaluate_models(df_merged, MODELS_DIR)
        print("\n📈 Model Comparison:")
        print(model_comparison)

        # Generate sample predictions using the best model
        logging.info("Generating sample predictions")

        # Find the best model path (gradient boosting typically performs best)
        model_path = MODELS_DIR / "demand_prediction_gradientboosting.joblib"
        if not model_path.exists():
            # Fall back to any model file
            model_files = list(MODELS_DIR.glob("*.joblib"))
            if model_files:
                model_path = model_files[0]
            else:
                logging.warning("No model files found - skipping predictions")
                model_path = None

        if model_path:
            # Load model
            model = load_model(model_path)

            # Get top 10 stations for predictions
            X, y, feature_names, top_stations = prepare_features(
                df_merged, top_n_stations=10
            )

            # Prepare station information for predictions
            station_info = (
                df_merged.groupby("start_station")
                .agg(
                    {
                        "station_name": "first",
                        "trip_id": "count",
                        "total_docks": "first",
                    }
                )
                .rename(columns={"trip_id": "trips"})
                .reset_index()
            )

            # Generate predictions
            predictions_path = OUTPUT_DIR / "rebalancing_predictions.csv"
            predictions = batch_predict(
                model_path=model_path,
                stations_df=station_info,
                output_path=predictions_path,
            )

            if predictions is not None:
                print(f"\n🔮 Rebalancing predictions saved to: {predictions_path}")
                print("\nSample rebalancing recommendations:")
                print(predictions.head().to_string(index=False))

        # Display path to models
        print(f"\n🧠 Models saved to: {MODELS_DIR}")

    # Print summary
    logging.info("=== ANALYSIS COMPLETE ===")
    print("\n✅ Analysis completed successfully!")
    print(f"📊 Total trips analyzed: {len(trips_df):,}")
    if stations_df is not None:
        print(f"🚉 Total stations analyzed: {len(stations_df):,}")
    print(f"\n💾 All outputs saved to: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    exit(main())
