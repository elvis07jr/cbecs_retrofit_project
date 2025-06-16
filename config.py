# Configuration settings for the CBECS Heating Retrofit Pipeline
import numpy as np

# --- Data Files ---
CBECS_DATA_FILE = "data/cbecs_2018_data.csv"  # Path to the CBECS dataset

# --- Cost and Economic Parameters ---
ENERGY_COSTS = {
    "NATURAL_GAS": 10.0,  # $/MMBtu (example value)
    "FUEL_OIL": 25.0,    # $/MMBtu (example value)
    "ELECTRICITY": 30.0, # $/MMBtu (example value, converted from $/kWh)
    "OTHER_FUEL": 15.0   # $/MMBtu (example value)
}

RETROFIT_COSTS = {
    "HIGH_EFFICIENCY_BOILER": 50,  # $/sqft (example value)
    "HEAT_PUMP_CONVERSION": 75,    # $/sqft (example value)
    "ENVELOPE_INSULATION": 30,     # $/sqft (example value)
    "WINDOW_REPLACEMENT": 60,      # $/sqft (example value)
    "AIR_SEALING": 10,             # $/sqft (example value)
    "SMART_THERMOSTAT": 5,         # $/sqft (example value)
    "BMS_IMPLEMENTATION": 40,      # $/sqft (example value)
    "LED_LIGHTING": 15,            # $/sqft (example value)
    "ZONE_CONTROL": 20             # $/sqft (example value)
}

ECONOMIC_PARAMS = {
    "DISCOUNT_RATE": 0.05,       # 5% (example value)
    "LIFESPAN_YEARS": 15,        # Assumed lifespan of retrofits (example value)
    "ANNUAL_MAINTENANCE_SAVINGS_PERCENT": 0.02 # 2% of retrofit cost (example)
}

# --- Regional and Climate Data ---
REGION_CLIMATE_ZONES = {
    "NORTHEAST": ["COLD", "VERY COLD"],
    "MIDWEST": ["COLD", "VERY COLD"],
    "SOUTH": ["HOT-DRY", "HOT-HUMID", "MIXED-HUMID", "MARINE"],
    "WEST": ["COLD", "VERY COLD", "HOT-DRY", "HOT-HUMID", "MIXED-HUMID", "MARINE"]
}

# --- Model Output ---
MODEL_OUTPUT_PATH = "models/heating_retrofit_model.pkl"

# --- Default Targeting Parameters ---
DEFAULT_TARGET_REGION = "NORTHEAST"  # Example: Focus on a specific census region
DEFAULT_TARGET_BUILDING_TYPES = ["OFFICE", "EDUCATION", "HEALTHCARE"] # Example: Focus on specific building types (PBA codes)

# --- Modeling Features ---
MODELING_FEATURE_COLUMNS = [] # List of features to be used in the model, can be populated by feature engineering steps

# --- Model Training Parameters ---
HYPERPARAMETER_GRIDS = {} # Dictionary of hyperparameter grids for different models

# --- Feature Engineering Constants (example, adjust as per actual data insights) ---
HEATING_FUEL_MAP = {
    1: "NATURAL_GAS", # Assuming 1 is Natural Gas from CBECS codebook
    2: "FUEL_OIL",    # Assuming 2 is Fuel Oil
    3: "ELECTRICITY", # Assuming 3 is Electricity
    # Add other mappings as per CBECS documentation
}

BUILDING_AGE_BINS = [0, 10, 20, 30, 40, 50, np.inf] # Example bins for age
BUILDING_AGE_LABELS = ['0-10', '11-20', '21-30', '31-40', '41-50', '50+']

# It's good practice to ensure numpy is imported if used directly in config like above.
