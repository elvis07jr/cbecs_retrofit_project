# data_processing.py

import pandas as pd
import warnings
import re # Import regex module for robust cleaning
from config import REGION_CLIMATE_ZONES

warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Handles loading and initial filtering of the CBECS dataset.
    """
    def __init__(self, target_region=None, target_building_types=None):
        self.data = None
        self.target_region = target_region
        self.target_building_types = target_building_types or []
        self.region_climate_zones = REGION_CLIMATE_ZONES

    def load_data(self, file_path):
        """Load CBECS dataset"""
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            else:
                print("Unsupported file format. Please use CSV or Excel.")
                return False

            # --- Robust column cleaning: Standardize all column names ---
            # 1. Strip leading/trailing whitespace
            self.data.columns = self.data.columns.str.strip()
            # 2. Replace multiple spaces with single underscore
            self.data.columns = self.data.columns.str.replace(r'\s+', '_', regex=True)
            # 3. Remove any characters not alphanumeric or underscore
            self.data.columns = self.data.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
            # 4. Convert all to uppercase for consistent comparison
            self.data.columns = self.data.columns.str.upper()
            # -----------------------------------------------------------

            print(f"Data loaded successfully: {self.data.shape}")
            print("\n--- Columns in loaded dataset (after robust cleaning): ---")
            print(self.data.columns.tolist())
            print("--------------------------------------------------------\n")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def explore_data(self):
        """Comprehensive data exploration for heating retrofit analysis"""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return

        print("=== CBECS HEATING RETROFIT DATA EXPLORATION ===")
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Missing Values Summary:")
        if not self.data.empty:
            print(self.data.isnull().sum().sort_values(ascending=False).head(20))
        else:
            print("DataFrame is empty, cannot compute missing values summary.")

        # All checks use consistent cleaned (uppercase) names
        print("\n=== HEATING SYSTEM OVERVIEW ===")
        if 'HEATHOME' in self.data.columns:
            print("Buildings with Heating Equipment:")
            print(self.data['HEATHOME'].value_counts())
        else:
            print("HEATHOME column not found for detailed heating system overview after cleaning.")

        if 'FUELHEAT' in self.data.columns:
            print("\nMain Heating Fuel Distribution:")
            print(self.data['FUELHEAT'].value_counts())
        else:
            print("FUELHEAT column not found for main heating fuel distribution after cleaning.")

        if 'EQUIPM' in self.data.columns:
            print("\nMain Heating Equipment Types:")
            print(self.data['EQUIPM'].value_counts())
        else:
            print("EQUIPM column not found for main heating equipment types after cleaning.")

        if 'MFHTBTU' in self.data.columns:
            print(f"\n=== HEATING CONSUMPTION ANALYSIS ===")
            temp_mfhtbtu = pd.to_numeric(self.data['MFHTBTU'], errors='coerce')
            heating_data = temp_mfhtbtu[temp_mfhtbtu > 0]
            print(f"Buildings with heating consumption: {len(heating_data)}")
            if not heating_data.empty:
                print(f"Mean heating consumption: {heating_data.mean():.2f} BTU")
                print(f"Median heating consumption: {heating_data.median():.2f} BTU")
                print(f"95th percentile: {heating_data.quantile(0.95):.2f} BTU")
            else:
                print("No positive heating consumption data found.")
        else:
            print("MFHTBTU column not found for heating consumption analysis after cleaning.")

        print(f"\n=== BUILDING CHARACTERISTICS ===")
        if 'PBA' in self.data.columns:
            print("Building Types:")
            # If PBA is numerical, show value counts (CBECS PBA can be numeric codes)
            if pd.api.types.is_numeric_dtype(self.data['PBA']):
                print("PBA column is numerical (likely encoded). Displaying top 10 value counts:")
                print(self.data['PBA'].value_counts().head(10))
            else: # If it's string/object type, show value counts directly
                print(self.data['PBA'].value_counts().head(10))
        else:
            print("PBA column not found for building types after cleaning.")

        if 'SQFT' in self.data.columns:
            sqft_data = pd.to_numeric(self.data['SQFT'], errors='coerce')
            sqft_data = sqft_data[sqft_data > 0]
            print(f"\nBuilding Size Distribution:")
            if not sqft_data.empty:
                print(f"Small (<5,000 sq ft): {(sqft_data < 5000).sum()}")
                print(f"Medium (5,000-50,000 sq ft): {((sqft_data >= 5000) & (sqft_data < 50000)).sum()}")
                print(f"Large (50,000+ sq ft): {(sqft_data >= 50000).sum()}")
            else:
                print("No positive SQFT data found.")
        else:
            print("SQFT column not found for building size distribution after cleaning.")

    def filter_by_region_and_type(self):
        """Filter data by target region and building types"""
        if self.data is None:
            print("No data loaded. Call load_data() first.")
            return None

        print("=== APPLYING REGIONAL AND BUILDING TYPE FILTERS ===")

        filtered_data = self.data.copy()

        # Check for 'PUBCLIM' existence before filtering by region
        if 'PUBCLIM' not in filtered_data.columns:
            print("Warning: 'PUBCLIM' column not found for regional filtering. Skipping regional filter.")
        elif self.target_region and self.target_region.upper() in self.region_climate_zones: # Ensure target_region is uppercased
            target_climates = self.region_climate_zones[self.target_region.upper()]
            initial_count = len(filtered_data)
            # Ensure PUBCLIM is treated as a string for comparison with target_climates
            filtered_data = filtered_data[filtered_data['PUBCLIM'].astype(str).isin(target_climates)]
            print(f"Filtered to {self.target_region} region: {len(filtered_data)} buildings (from {initial_count})")
        else:
            print("No specific region target or 'PUBCLIM' column not present for regional filtering. Including all regions.")

        # Check for 'PBA' existence before filtering by building types
        if 'PBA' not in filtered_data.columns:
            print("Warning: 'PBA' column not found for building type filtering. Skipping building type filter.")
        elif self.target_building_types:
            initial_count = len(filtered_data)
            # Ensure PBA is treated as a string for comparison with target_building_types (which should be uppercase/cleaned)
            filtered_data = filtered_data[filtered_data['PBA'].astype(str).isin(
                [bt.upper().replace(' ', '_').replace(r'[^a-zA-Z0-9_]', '') for bt in self.target_building_types]
            )]
            print(f"Filtered to building types {self.target_building_types}: {len(filtered_data)} buildings (from {initial_count})")
        else:
            print("No specific building types target or 'PBA' column not present for building type filtering. Including all building types.")

        self.data = filtered_data
        print(f"Final filtered dataset: {len(self.data)} buildings")

        return self.data
