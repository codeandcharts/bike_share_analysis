# Metro Bike Share Analysis Project

![Project Banner](https://via.placeholder.com/1200x300/e8f4f8/2980b9?text=Metro+Bike+Share+Analysis)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Data Pipeline](#data-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Key Findings](#key-findings)
- [Recommendations](#recommendations)
- [Technical Implementation](#technical-implementation)
- [Skills Demonstrated](#skills-demonstrated)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

## 🚲 Project Overview

This end-to-end data analytics project explores Metro Bike Share system data to uncover operational insights, optimize revenue streams, and enhance user experience. By analyzing millions of trip records across multiple quarters, this project showcases my ability to transform raw data into actionable business recommendations through a comprehensive ETL pipeline, exploratory analysis, statistical modeling, and data visualization.

### Project Objectives

1. Build a scalable ETL pipeline for processing bike share data
2. Identify temporal usage patterns to optimize operations
3. Analyze station popularity and network efficiency
4. Understand user segments and their behaviors
5. Develop predictive models for demand forecasting
6. Generate data-driven recommendations for business improvement

## 🔄 Data Pipeline

The project begins with a robust ETL (Extract, Transform, Load) pipeline implemented in Python:

### ETL Architecture
1. **EXTRACT**: Unzip raw monthly trip data files and read station metadata
2. **LOAD**: Read and concatenate CSVs and station JSON data
3. **TRANSFORM**: Clean data, impute missing values, remove outliers, engineer features
4. **LOAD**: Persist cleaned tables into DuckDB for analysis
5. **VALIDATE**: Perform integrity checks to ensure data quality

### Key Features of the Pipeline
- Handles schema variations across different monthly data files
- Implements robust error handling for various file formats
- Performs outlier detection and removal using IQR method
- Engineers temporal, spatial, and behavioral features
- Integrates trip data with station metadata
- Creates optimized database views for analysis

## 📊 Exploratory Data Analysis

The analysis is organized into four main thematic areas:

### 1. Temporal Utilization Analysis
- Hourly usage patterns 
- Day of week analysis
- Seasonal trends
- Monthly ridership patterns

### 2. Station and Network Analysis
- Station popularity and demand
- Capacity utilization metrics
- Rebalancing needs assessment
- Route analysis and flow patterns
- Geospatial clustering of stations

### 3. User Experience and Segmentation
- Rider segmentation (subscribers vs. casual riders)
- Trip duration analysis by segment
- Weekly and hourly patterns by user segment
- Bike type preferences
- Trip distance patterns

### 4. Revenue and Pricing Analysis
- Revenue by plan type
- Overage charge patterns
- Price elasticity assessment
- Pricing efficiency metrics
- Duration-based revenue optimization

## 🔍 Key Findings

### Temporal Patterns
- **Peak Usage**: Clear bimodal distribution with morning (8:00) and evening (17:00-18:00) peaks
- **Day of Week**: Weekdays show commuter-focused usage; weekends display leisure patterns
- **Seasonality**: Summer rides account for ~32% of annual total; winter only ~18%
- **Monthly Variation**: 1.8x difference between highest and lowest months

### Station Insights
- Top 10 stations account for ~30% of all trip origins
- High-demand stations show 5-10x utilization rates compared to system average
- Several stations experience frequent shortage events
- Five distinct station clusters identified through k-means analysis

### User Behavior
- Subscribers (65% of riders) take shorter trips (avg. 18 min) compared to casual riders (avg. 35 min)
- Weekend rides are 70% casual riders; weekday rush hours are 85% subscribers
- Strong price elasticity observed at the 30-minute included time threshold
- Casual riders travel ~40% farther distances than subscribers

### Predictive Modeling Results
- Gradient Boosting algorithm achieved highest accuracy (R² = 0.76)
- Hour of day is the strongest predictor of demand
- Prediction accuracy varies by time of day, with rush hours most predictable
- Model can forecast demand at station level with reasonable accuracy

## 📈 Recommendations

### Operational Improvements
1. **Dynamic Rebalancing**:
   - Implement predictive rebalancing based on forecast model
   - Prioritize top 10 stations for morning availability
   - Establish a tiered rebalancing schedule based on historical patterns

2. **Fleet Optimization**:
   - Adjust fleet size seasonally (reduce winter fleet by ~15%)
   - Relocate docks from low to high-utilization stations
   - Implement preventative maintenance during predicted low-demand periods

### Revenue Optimization
1. **Pricing Refinements**:
   - Test a 45-minute included time threshold
   - Implement dynamic pricing during peak hours (+10-15%)
   - Create weekend-specific passes for casual riders

2. **Subscription Enhancements**:
   - Develop targeted conversion campaigns for frequent casual riders
   - Create a "weekend warrior" subscription option
   - Implement corporate bulk discount programs

### User Experience Improvements
1. **Mobile App Enhancements**:
   - Add time-remaining notifications to help avoid overage charges
   - Create personalized route recommendations
   - Implement turn-by-turn navigation for tourists

2. **Communication Strategy**:
   - Send personalized usage summaries with savings calculations
   - Create a gamification system to encourage off-peak usage
   - Develop targeted communication by rider segment

### Strategic Growth
1. **Network Expansion**:
   - Use the predictive model to identify optimal locations for new stations
   - Create satellite mini-stations around highest-demand areas
   - Establish designated bike lanes along highest-volume routes

## 💻 Technical Implementation

### Technologies Used
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computation
- **DuckDB**: High-performance analytical database
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning and predictive modeling
- **GeoPy**: Geospatial calculations
- **Statsmodels**: Time series analysis

### Advanced Techniques
- Outlier detection using Interquartile Range (IQR)
- K-means clustering for station grouping
- Random Forest and Gradient Boosting algorithms
- Time series decomposition to identify seasonal patterns
- Feature importance analysis
- Cross-validation for robust model evaluation

## 🛠️ Skills Demonstrated

This project showcases my proficiency in:
- End-to-end data pipeline development
- Data cleaning and transformation at scale
- Feature engineering for business insights
- Advanced statistical analysis and visualization
- Machine learning model development and evaluation
- Translating technical findings into business recommendations
- Documentation and technical communication

## 📁 Project Structure

```
metro_bike_share_analysis/
│
├── data/
│   ├── raw/                  # Original data files
│   ├── interim/              # Extracted and intermediate data
│   └── processed/            # Cleaned, transformed data in DuckDB
│
├── notebooks/
│   ├── 1.0-data-exploration.ipynb
│   ├── 2.0-temporal-analysis.ipynb
│   ├── 3.0-station-analysis.ipynb
│   ├── 4.0-user-experience.ipynb
│   ├── 5.0-revenue-analysis.ipynb
│   └── 6.0-predictive-modeling.ipynb
│
├── src/
│   ├── data/
│   │   ├── extract.py        # Data extraction utilities
│   │   ├── transform.py      # Data transformation functions
│   │   └── load.py           # Database loading scripts
│   │
│   ├── visualization/
│   │   └── visualize.py      # Visualization utilities
│   │
│   └── models/
│       ├── train_model.py    # Model training scripts
│       └── predict.py        # Prediction utilities
│
├── etl_pipeline.py           # Main ETL script
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/metro-bike-share-analysis.git
cd metro-bike-share-analysis
```

### 2. Set up your environment

#### Option A: Using conda (recommended for data science)
```bash
# Create a new conda environment
conda create -n bike-share python=3.10

# Activate the environment
conda activate bike-share

# Install dependencies
pip install -r requirements.txt

# Register the environment with Jupyter
python -m ipykernel install --user --name=bike-share --display-name="Python (Bike Share)"
```

#### Option B: Using Python venv
```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure data paths
Edit the configuration section in `etl_pipeline.py` to specify your local data paths:

```python
# Edit these paths to match your local directory structure
RAW_DATA_PATH = Path("./data/raw")
EXTRACT_DIR = Path("./data/interim/extracted_csvs")
DUCKDB_PATH = Path("./data/processed/metro_bike_share.duckdb")
```

### 4. Run the ETL pipeline
```bash
# Process the raw data and create the database
python etl_pipeline.py
```

### 5. Explore the analysis notebooks
```bash
# Start Jupyter Lab for interactive exploration
jupyter lab
```

### 6. Troubleshooting

If you encounter any package compatibility issues:
```bash
# Update pip
pip install --upgrade pip

# Install packages individually if there are conflicts
pip install pandas==2.0.0
pip install duckdb==0.8.1
# etc.
```

For large datasets, you may need to increase memory limits:
```bash
# Start Jupyter with increased memory (8GB example)
jupyter lab --NotebookApp.max_buffer_size=8589934592
```