"""
Visualization module for Metro Bike Share analysis.

This module contains functions to create standardized visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path


def set_visualization_defaults():
    """
    Set consistent styling for all visualizations.

    Returns:
        dict: Dictionary of custom color palettes
    """
    # Base settings
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("notebook", font_scale=1.2)

    # Figure size and fonts
    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]

    # Text sizing
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 18

    # Define color palettes for different analysis types
    palettes = {
        "temporal": sns.color_palette("viridis", 12),
        "user": sns.color_palette("Set2", 8),
        "comparison": sns.color_palette("RdYlBu_r", 10),
        "business": sns.color_palette("YlGnBu", 8),
        "categorical": sns.color_palette("tab10"),
    }

    return palettes


def plot_hourly_usage(df, save_path=None):
    """
    Plot hourly usage patterns.

    Args:
        df (pd.DataFrame): DataFrame with trip data
        save_path (Path, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Count trips by hour
    hourly_rides = df.groupby("trip_hour").size().reset_index(name="rides")

    # Create figure
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=hourly_rides,
        x="trip_hour",
        y="rides",
        marker="o",
        color="purple",
        linewidth=2.5,
    )

    # Identify peak hour
    max_hour = hourly_rides.loc[hourly_rides["rides"].idxmax(), "trip_hour"]
    max_rides = hourly_rides["rides"].max()

    # Add annotations for key periods
    plt.annotate(
        f"Peak: {max_hour}:00 ({max_rides:,} rides)",
        xy=(max_hour, max_rides),
        xytext=(max_hour + 1, max_rides * 1.05),
        arrowprops=dict(arrowstyle="->"),
        fontsize=12,
    )

    # Highlight morning and evening rush hours
    plt.axvspan(7, 9, alpha=0.2, color="orange", label="Morning Rush")
    plt.axvspan(16, 19, alpha=0.2, color="green", label="Evening Rush")

    # Customize the chart
    plt.title("Hourly Ridership Distribution", fontsize=16)
    plt.xlabel("Hour of Day (24h)", fontsize=14)
    plt.ylabel("Number of Rides", fontsize=14)
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.gcf()


def plot_day_of_week(df, save_path=None):
    """
    Plot day of week ridership patterns.

    Args:
        df (pd.DataFrame): DataFrame with trip data
        save_path (Path, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Define weekday order
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Count rides by day of week
    daily_rides = df["weekday"].value_counts().reindex(weekday_order).reset_index()
    daily_rides.columns = ["weekday", "rides"]

    # Calculate weekend vs. weekday averages
    weekday_avg = daily_rides[
        daily_rides["weekday"].isin(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        )
    ]["rides"].mean()
    weekend_avg = daily_rides[daily_rides["weekday"].isin(["Saturday", "Sunday"])][
        "rides"
    ].mean()

    # Create day of week ridership visualization
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=daily_rides, x="weekday", y="rides", palette="magma")

    # Add value labels on bars
    for i, p in enumerate(ax.patches):
        ax.annotate(
            f"{p.get_height():,.0f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Color code weekdays vs. weekends
    for i, day in enumerate(weekday_order):
        if day in ["Saturday", "Sunday"]:
            ax.patches[i].set_facecolor("crimson")

    # Add reference lines for averages
    plt.axhline(
        y=weekday_avg,
        color="blue",
        linestyle="--",
        label=f"Weekday Avg: {weekday_avg:.0f}",
    )
    plt.axhline(
        y=weekend_avg,
        color="red",
        linestyle="--",
        label=f"Weekend Avg: {weekend_avg:.0f}",
    )

    # Customize the chart
    plt.title("Ridership by Day of Week", fontsize=16)
    plt.xlabel("Day of Week", fontsize=14)
    plt.ylabel("Number of Rides", fontsize=14)
    plt.legend()
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.gcf()


def plot_seasonal_trends(df, save_path=None):
    """
    Plot seasonal ridership trends.

    Args:
        df (pd.DataFrame): DataFrame with trip data
        save_path (Path, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Define season order
    season_order = ["Winter", "Spring", "Summer", "Autumn"]

    # Count rides by season
    seasonal_rides = df["season"].value_counts().reindex(season_order).reset_index()
    seasonal_rides.columns = ["season", "rides"]

    # Calculate seasonal percentages
    total_rides = seasonal_rides["rides"].sum()
    seasonal_rides["percentage"] = (seasonal_rides["rides"] / total_rides * 100).round(
        1
    )

    # Create seasonal ridership visualization
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=seasonal_rides, x="season", y="rides", palette="coolwarm")

    # Add value labels on bars
    for i, p in enumerate(ax.patches):
        ax.annotate(
            f"{p.get_height():,.0f} ({seasonal_rides.iloc[i]['percentage']}%)",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add reference line for even distribution
    plt.axhline(
        y=total_rides / 4,
        color="gray",
        linestyle="--",
        label=f"Even Distribution: {total_rides / 4:.0f}",
    )

    # Customize the chart
    plt.title("Seasonal Ridership Distribution", fontsize=16)
    plt.xlabel("Season", fontsize=14)
    plt.ylabel("Number of Rides", fontsize=14)
    plt.legend()
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.gcf()


def plot_hourly_heatmap(df, save_path=None):
    """
    Create a heatmap of hourly patterns by day of week.

    Args:
        df (pd.DataFrame): DataFrame with trip data
        save_path (Path, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Define weekday order
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Create a crosstab of hour vs. day
    hourly_daily = pd.crosstab(
        index=df["trip_hour"],
        columns=df["weekday"],
        values=df["trip_id"],
        aggfunc="count",
    )

    # Reorder columns to standard weekday order
    hourly_daily = hourly_daily.reindex(columns=weekday_order)

    # Create the heatmap visualization
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(
        hourly_daily,
        cmap="viridis",
        linewidths=0.5,
        annot=True,
        fmt="g",
        cbar_kws={"label": "Number of Rides"},
    )

    # Customize the chart
    plt.title("Heatmap of Rides by Hour and Day of Week", fontsize=16)
    plt.xlabel("Day of Week", fontsize=14)
    plt.ylabel("Hour of Day (24h)", fontsize=14)
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.gcf()


def plot_monthly_trends(df, save_path=None):
    """
    Plot monthly ridership trends.

    Args:
        df (pd.DataFrame): DataFrame with trip data
        save_path (Path, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Count rides by month and year
    monthly_rides = df.groupby(["year", "month"]).size().reset_index(name="rides")
    monthly_rides["date"] = pd.to_datetime(
        monthly_rides[["year", "month"]].assign(day=1)
    )

    # Create monthly trend visualization
    plt.figure(figsize=(14, 6))
    ax = sns.lineplot(data=monthly_rides, x="date", y="rides", marker="o", linewidth=2)

    # Format the x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    # Customize the chart
    plt.title("Monthly Ridership Trend", fontsize=16)
    plt.xlabel("Month", fontsize=14)
    plt.ylabel("Number of Rides", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.gcf()


def plot_top_stations(df, n=10, save_path=None):
    """
    Plot top N stations by usage.

    Args:
        df (pd.DataFrame): DataFrame with trip data
        n (int): Number of top stations to show
        save_path (Path, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Count trips starting at each station
    station_trips = df.groupby("station_name").size().reset_index(name="trips")
    station_trips = station_trips.sort_values("trips", ascending=False)

    # Select top N stations
    top_stations = station_trips.head(n)

    # Create visualization
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=top_stations, x="trips", y="station_name", palette="viridis")

    # Add value labels
    for i, p in enumerate(ax.patches):
        ax.annotate(
            f"{p.get_width():,}",
            (p.get_width(), p.get_y() + p.get_height() / 2),
            ha="left",
            va="center",
            fontsize=10,
            xytext=(5, 0),
            textcoords="offset points",
        )

    # Customize the chart
    plt.title(f"Top {n} Most Popular Stations", fontsize=16)
    plt.xlabel("Number of Trips", fontsize=14)
    plt.ylabel("Station Name", fontsize=14)
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.gcf()


def plot_station_utilization(df, save_path=None):
    """
    Create a two-panel visualization of station utilization.

    Args:
        df (pd.DataFrame): DataFrame with aggregated station metrics
        save_path (Path, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create a two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Panel 1: Utilization rate histogram
    sns.histplot(data=df, x="utilization_rate", bins=20, kde=True, ax=ax1)
    ax1.axvline(
        x=df["utilization_rate"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['utilization_rate'].mean():.1f}",
    )
    ax1.axvline(
        x=df["utilization_rate"].median(),
        color="green",
        linestyle="--",
        label=f"Median: {df['utilization_rate'].median():.1f}",
    )
    ax1.set_title("Distribution of Station Utilization Rates", fontsize=14)
    ax1.set_xlabel("Utilization Rate (Trips per Dock)", fontsize=12)
    ax1.set_ylabel("Number of Stations", fontsize=12)
    ax1.legend()

    # Panel 2: Capacity vs. usage scatter plot
    scatter = ax2.scatter(
        df["total_docks"],
        df["trips"],
        c=df["utilization_rate"],
        cmap="viridis",
        s=100,
        alpha=0.7,
    )

    # Add labels for top utilized stations
    top_util = df.sort_values("utilization_rate", ascending=False).head(5)
    for _, station in top_util.iterrows():
        ax2.annotate(
            station["station_name"],
            (station["total_docks"], station["trips"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    # Add diagonal reference lines for utilization
    max_val = max(df["total_docks"].max(), df["trips"].max() / 100)
    ax2.plot([0, max_val], [0, max_val * 100], "r--", label="100 trips per dock")
    ax2.plot([0, max_val], [0, max_val * 50], "g--", label="50 trips per dock")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Trips per Dock (Utilization Rate)")

    # Customize panel 2
    ax2.set_title("Station Capacity vs. Usage", fontsize=14)
    ax2.set_xlabel("Station Capacity (Total Docks)", fontsize=12)
    ax2.set_ylabel("Number of Trips", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_rider_segments(df, save_path=None):
    """
    Create a pie chart of rider segments.

    Args:
        df (pd.DataFrame): DataFrame with trip data
        save_path (Path, optional): Path to save the figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Check rider segment distributions
    rider_segments = df["rider_segment"].value_counts()
    rider_pcts = df["rider_segment"].value_counts(normalize=True) * 100

    # Create the pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(
        rider_pcts.values,
        labels=rider_pcts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Set2", len(rider_pcts)),
        explode=[0.05] * len(rider_pcts),
    )
    plt.title("Distribution of Rider Segments", pad=20, fontsize=16)
    plt.axis("equal")
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.gcf()


def save_all_visualizations(df, output_dir):
    """
    Create and save all standard visualizations to the output directory.

    Args:
        df (pd.DataFrame): DataFrame with trip data
        output_dir (Path): Directory to save visualizations

    Returns:
        list: List of saved visualization paths
    """
    # Make sure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set visualization defaults
    set_visualization_defaults()

    # Create and save standard visualizations
    saved_paths = []

    # 1. Hourly usage
    hourly_path = output_dir / "hourly_usage.png"
    plot_hourly_usage(df, save_path=hourly_path)
    saved_paths.append(hourly_path)

    # 2. Day of week
    dow_path = output_dir / "day_of_week.png"
    plot_day_of_week(df, save_path=dow_path)
    saved_paths.append(dow_path)

    # 3. Seasonal trends
    season_path = output_dir / "seasonal_trends.png"
    plot_seasonal_trends(df, save_path=season_path)
    saved_paths.append(season_path)

    # 4. Hourly heatmap
    heatmap_path = output_dir / "hourly_heatmap.png"
    plot_hourly_heatmap(df, save_path=heatmap_path)
    saved_paths.append(heatmap_path)

    # 5. Monthly trends
    monthly_path = output_dir / "monthly_trends.png"
    plot_monthly_trends(df, save_path=monthly_path)
    saved_paths.append(monthly_path)

    # 6. Top stations
    stations_path = output_dir / "top_stations.png"
    plot_top_stations(df, save_path=stations_path)
    saved_paths.append(stations_path)

    # 7. Rider segments
    segments_path = output_dir / "rider_segments.png"
    plot_rider_segments(df, save_path=segments_path)
    saved_paths.append(segments_path)

    # 8. Station utilization
    if "total_docks" in df.columns and "station_name" in df.columns:
        # Create station metrics
        station_metrics = (
            df.groupby("station_name")
            .agg(
                trips=("trip_id", "count"),
                total_docks=("total_docks", "first"),
                lat=("start_lat", "first"),
                lon=("start_lon", "first"),
            )
            .reset_index()
        )

        station_metrics["utilization_rate"] = (
            station_metrics["trips"] / station_metrics["total_docks"]
        )

        util_path = output_dir / "station_utilization.png"
        plot_station_utilization(station_metrics, save_path=util_path)
        saved_paths.append(util_path)

    logging.info(f"Saved {len(saved_paths)} visualizations to {output_dir}")
    return saved_paths
