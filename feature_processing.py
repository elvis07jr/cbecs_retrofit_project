# feature_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import ENERGY_COSTS, RETROFIT_COSTS, ECONOMIC_PARAMS, MODELING_FEATURE_COLUMNS
import warnings

warnings.filterwarnings('ignore')

class FeatureProcessor:
    """
    Handles feature engineering, economic metric calculation, and preprocessing
    for machine learning models.
    """
    def __init__(self, economic_params=None):
        self.processed_data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.economic_params = economic_params or ECONOMIC_PARAMS

    def create_heating_efficiency_features(self, data):
        """
        Create features specifically for heating efficiency analysis.
        Includes robust handling for 'HEATHOME' column and numeric coercion.
        """
        if data is None or data.empty:
            print("No data provided or data is empty for feature creation.")
            self.processed_data = pd.DataFrame()
            return self.processed_data

        print("=== CREATING HEATING EFFICIENCY FEATURES ===")

        # No need for explicit column cleaning here as it's done in DataProcessor.load_data.
        # But we'll re-check for consistency.
        data.columns = data.columns.str.strip()
        data.columns = data.columns.str.replace(r'\s+', '_', regex=True)
        data.columns = data.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
        data.columns = data.columns.str.upper()

        print("\n--- Columns in data received by create_heating_efficiency_features (re-checked): ---")
        print(data.columns.tolist())
        print("--------------------------------------------------------------------\n")

        # Define critical columns required for heating efficiency features
        # These are the columns needed for the initial filtering and calculations
        critical_columns_for_features = ['HEATHOME', 'MFHTBTU', 'SQFT', 'HDD65', 'YRCONC', 'PBA', 'PUBCLIM']
        missing_cols = [col for col in critical_columns_for_features if col not in data.columns]

        if missing_cols:
            print(f"Warning: Missing critical columns for heating efficiency feature creation: {', '.join(missing_cols)}")
            print("Please ensure your dataset contains these columns with expected values.")
            self.processed_data = pd.DataFrame() # Return empty DataFrame if critical columns are missing
            return self.processed_data

        # --- Robust HEATHOME handling ---
        # Convert 'HEATHOME' to a standardized boolean/integer format
        hearthome_standardized = data['HEATHOME'].astype(str).str.strip().str.lower()
        # Map 'yes', 'y', '1', 'true' to True, others to False (including 'no', '0', empty, NaN)
        data['HEATHOME_STANDARDIZED'] = hearthome_standardized.map({'yes': True, 'y': True, '1': True, 'true': True}).fillna(False)
        print(f"Standardized HEATHOME values (True/False counts):")
        print(data['HEATHOME_STANDARDIZED'].value_counts())

        # --- Numeric Coercion and Filtering for Data Quality ---
        # Convert critical columns to numeric, coercing errors to NaN
        data['MFHTBTU'] = pd.to_numeric(data['MFHTBTU'], errors='coerce')
        data['SQFT'] = pd.to_numeric(data['SQFT'], errors='coerce')
        data['HDD65'] = pd.to_numeric(data['HDD65'], errors='coerce')
        data['YRCONC'] = pd.to_numeric(data['YRCONC'], errors='coerce')

        # Drop rows where critical numeric columns became NaN after coercion or are non-positive
        # This ensures we only work with valid numeric data for calculations
        initial_rows = len(data)
        data_filtered_quality = data.dropna(subset=['MFHTBTU', 'SQFT', 'HDD65', 'YRCONC']).copy()
        data_filtered_quality = data_filtered_quality[
            (data_filtered_quality['MFHTBTU'] > 0) &
            (data_filtered_quality['SQFT'] > 0) &
            (data_filtered_quality['HDD65'] > 0)
        ]
        print(f"Dropped {initial_rows - len(data_filtered_quality)} rows due to missing/invalid critical numeric data.")

        # Filter buildings with heating systems based on the standardized column
        heating_buildings = data_filtered_quality[data_filtered_quality['HEATHOME_STANDARDIZED'] == True].copy()

        print(f"Buildings with heating systems and valid data: {len(heating_buildings)}")

        if heating_buildings.empty:
            print("No valid heating buildings found after initial filtering or data quality checks on numeric values and HEATHOME.")
            self.processed_data = heating_buildings
            return heating_buildings

        # Calculate heating intensity metrics
        heating_buildings['HEATING_INTENSITY_SQFT'] = heating_buildings['MFHTBTU'] / heating_buildings['SQFT']
        heating_buildings['HEATING_INTENSITY_HDD'] = heating_buildings['MFHTBTU'] / heating_buildings['HDD65']
        # Avoid division by zero if SQFT or HDD65 are somehow still zero (should be handled by filters above, but as a safeguard)
        heating_buildings['HEATING_INTENSITY_CLIMATE'] = heating_buildings['MFHTBTU'] / (heating_buildings['SQFT'] * heating_buildings['HDD65'].replace(0, np.nan))
        heating_buildings['HEATING_INTENSITY_CLIMATE'].fillna(0, inplace=True) # Fill inf/NaN from division by zero with 0 or a sensible value

        # Building age
        heating_buildings['BUILDING_AGE'] = 2018 - heating_buildings['YRCONC']

        # Create peer groups for benchmarking
        heating_buildings['SIZE_CATEGORY'] = pd.cut(
            heating_buildings['SQFT'],
            bins=[0, 5000, 25000, 100000, float('inf')],
            labels=['SMALL', 'MEDIUM', 'LARGE', 'VERY_LARGE'], # Use uppercase labels for consistency
            right=False
        )

        heating_buildings['AGE_CATEGORY'] = pd.cut(
            heating_buildings['BUILDING_AGE'],
            bins=[0, 20, 40, 60, float('inf')],
            labels=['NEW', 'RECENT', 'MATURE', 'OLD'], # Use uppercase labels
            right=False
        )

        # Calculate peer group benchmarks
        peer_stats = heating_buildings.groupby(['PBA', 'SIZE_CATEGORY', 'PUBCLIM'])['HEATING_INTENSITY_SQFT'].agg(['mean', 'std']).reset_index()
        peer_stats.columns = ['PBA', 'SIZE_CATEGORY', 'PUBCLIM', 'PEER_MEAN_INTENSITY', 'PEER_STD_INTENSITY']

        # Merge back peer statistics
        heating_buildings = heating_buildings.merge(peer_stats, on=['PBA', 'SIZE_CATEGORY', 'PUBCLIM'], how='left')

        # Fill NaN in PEER_MEAN_INTENSITY and PEER_STD_INTENSITY for unique groups
        heating_buildings['PEER_MEAN_INTENSITY'].fillna(heating_buildings['HEATING_INTENSITY_SQFT'].mean(), inplace=True)
        heating_buildings['PEER_STD_INTENSITY'].fillna(heating_buildings['HEATING_INTENSITY_SQFT'].std(), inplace=True)
        # Avoid division by zero if std is 0
        heating_buildings['PEER_STD_INTENSITY'] = heating_buildings['PEER_STD_INTENSITY'].replace(0, 1e-6)


        # Calculate efficiency scores
        heating_buildings['EFFICIENCY_SCORE'] = (
            heating_buildings['HEATING_INTENSITY_SQFT'] - heating_buildings['PEER_MEAN_INTENSITY']
        ) / heating_buildings['PEER_STD_INTENSITY']

        # Identify high retrofit potential buildings
        # This is your target variable.
        # If your data is heavily skewed or this threshold yields too few positives, you might need to adjust it.
        heating_buildings['HIGH_RETROFIT_POTENTIAL'] = (heating_buildings['EFFICIENCY_SCORE'] > 1.5).astype(int)

        print(f"Buildings with high retrofit potential: {heating_buildings['HIGH_RETROFIT_POTENTIAL'].sum()}")
        print(f"Percentage of buildings with high retrofit potential: {heating_buildings['HIGH_RETROFIT_POTENTIAL'].mean()*100:.2f}%")

        self.processed_data = heating_buildings
        return heating_buildings

    def calculate_economic_metrics(self, data):
        """Calculate economic metrics for retrofit potential."""
        if data is None or data.empty:
            print("No data provided or data is empty for economic metric calculation.")
            return None

        print("=== CALCULATING ECONOMIC METRICS ===")
        processed_data = data.copy()

        if 'FUELHEAT' in processed_data.columns:
            # Ensure FUELHEAT is handled as string before fillna
            processed_data['FUELHEAT'] = processed_data['FUELHEAT'].astype(str).str.upper().fillna('UNKNOWN')
        else:
            print("Warning: 'FUELHEAT' column not found for economic calculations. Using a default fuel cost.")
            processed_data['FUELHEAT'] = 'UNKNOWN' # Assign a default for calculations to proceed

        processed_data['POTENTIAL_SAVINGS_BTU'] = (
            processed_data['HEATING_INTENSITY_SQFT'] - processed_data['PEER_MEAN_INTENSITY']
        ) * processed_data['SQFT']

        def get_energy_cost_per_btu(fuel_type):
            # Ensure fuel_type is uppercase for matching ENERGY_COSTS keys (if they are standardized)
            fuel_type = str(fuel_type).upper().replace(' ', '_')
            if fuel_type == 'NATURAL_GAS': return ENERGY_COSTS.get('Natural Gas', 0.012) / 100000
            if fuel_type == 'ELECTRICITY': return ENERGY_COSTS.get('Electricity', 0.10) / 3412
            if fuel_type == 'FUEL_OIL': return ENERGY_COSTS.get('Fuel Oil', 0.025) / 138000
            if fuel_type == 'PROPANE': return ENERGY_COSTS.get('Propane', 0.030) / 91500
            if fuel_type == 'DISTRICT_HEAT': return ENERGY_COSTS.get('District Heat', 0.015) / 100000
            return 0.01 / 100000 # Default if fuel type is not found or unknown (small value)


        processed_data['CURRENT_FUEL_COST_PER_BTU'] = processed_data['FUELHEAT'].apply(get_energy_cost_per_btu)
        processed_data['ANNUAL_SAVINGS'] = processed_data['POTENTIAL_SAVINGS_BTU'] * processed_data['CURRENT_FUEL_COST_PER_BTU']

        avg_retrofit_cost_per_sqft = RETROFIT_COSTS['Comprehensive']['medium']
        processed_data['RETROFIT_COST_ESTIMATE'] = processed_data['SQFT'] * avg_retrofit_cost_per_sqft

        processed_data['ANNUAL_SAVINGS'] = processed_data['ANNUAL_SAVINGS'].clip(lower=0)
        processed_data['RETROFIT_COST_ESTIMATE'] = processed_data['RETROFIT_COST_ESTIMATE'].clip(lower=1)

        processed_data['SIMPLE_PAYBACK'] = processed_data['RETROFIT_COST_ESTIMATE'] / processed_data['ANNUAL_SAVINGS']
        processed_data['SIMPLE_PAYBACK'].replace([np.inf, -np.inf], np.nan, inplace=True)

        discount_rate = self.economic_params['discount_rate']
        analysis_period = self.economic_params['analysis_period']
        energy_cost_escalation = self.economic_params['energy_cost_escalation']
        maintenance_cost_factor = self.economic_params['maintenance_cost_factor']

        def calculate_npv_sir(row):
            initial_cost = row['RETROFIT_COST_ESTIMATE']
            annual_savings = row['ANNUAL_SAVINGS']

            if initial_cost <= 0 or annual_savings <= 0:
                return 0, 0, False

            npv_sum = 0
            annual_maintenance_cost = initial_cost * maintenance_cost_factor

            for year in range(1, analysis_period + 1):
                escalated_savings = annual_savings * ((1 + energy_cost_escalation)**(year - 1))
                net_cash_flow = escalated_savings - annual_maintenance_cost
                npv_sum += net_cash_flow / ((1 + discount_rate)**year)

            npv = npv_sum - initial_cost
            sir = npv_sum / initial_cost if initial_cost > 0 else 0

            economically_feasible = (npv > 0) and (sir > 1)

            return npv, sir, economically_feasible

        economic_results = processed_data.apply(calculate_npv_sir, axis=1, result_type='expand')
        processed_data['NPV'] = economic_results[0]
        processed_data['SIR'] = economic_results[1]
        processed_data['ECONOMICALLY_FEASIBLE'] = economic_results[2].astype(int)

        print(f"Buildings economically feasible: {processed_data['ECONOMICALLY_FEASIBLE'].sum()}")

        self.processed_data = processed_data
        return processed_data

    def preprocess_features(self, data):
        """Preprocess features for ML models"""
        if data is None or data.empty:
            print("No data provided or data is empty for preprocessing.")
            return None, None

        print("=== PREPROCESSING FEATURES ===")

        # Ensure all columns are consistently uppercase and clean before selecting
        data.columns = data.columns.str.strip()
        data.columns = data.columns.str.replace(r'\s+', '_', regex=True)
        data.columns = data.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
        data.columns = data.columns.str.upper()

        available_features = [col for col in MODELING_FEATURE_COLUMNS if col in data.columns]
        # Important: If 'HIGH_RETROFIT_POTENTIAL' is missing, the target `y` will fail.
        # This should ideally be ensured by create_heating_efficiency_features.
        if 'HIGH_RETROFIT_POTENTIAL' not in data.columns:
            print("Error: Target column 'HIGH_RETROFIT_POTENTIAL' not found. Cannot train model.")
            return None, None

        X = data[available_features].copy()
        y = data['HIGH_RETROFIT_POTENTIAL']

        print("Handling missing values...")
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna('UNKNOWN') # Use UNKNOWN for categorical missing values
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype(str)

        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            X[col] = X[col].fillna(X[col].median())

        print("Encoding categorical variables...")
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit and transform, handling potential unseen categories by marking them as 0 or a distinct value
            # This is critical for deployment when `transform_new_data` is used.
            X[col] = le.fit_transform(X[col].astype(str)) # Ensure string type for encoder
            self.label_encoders[col] = le

        print("Scaling numerical features...")
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        print(f"Final feature matrix shape: {X_scaled.shape}")
        print(f"Target distribution: {y.value_counts()}")

        return X_scaled, y

    def get_scaler(self):
        return self.scaler

    def get_label_encoders(self):
        return self.label_encoders

    def get_processed_data(self):
        return self.processed_data

    def transform_new_data(self, new_building_data):
        """
        Transforms new building data using the fitted scaler and label encoders.
        This method expects the *engineered* data (after create_heating_efficiency_features and calculate_economic_metrics)
        """
        processed_data = new_building_data.copy()

        # Apply same column cleaning as during training
        processed_data.columns = processed_data.columns.str.strip()
        processed_data.columns = processed_data.columns.str.replace(r'\s+', '_', regex=True)
        processed_data.columns = processed_data.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
        processed_data.columns = processed_data.columns.str.upper()

        # Ensure all expected MODELING_FEATURE_COLUMNS are present
        # Fill missing with a placeholder value suitable for each type
        for col in MODELING_FEATURE_COLUMNS:
            if col not in processed_data.columns:
                if col in self.label_encoders: # Was categorical in training
                    processed_data[col] = 'UNKNOWN'
                else: # Was numerical
                    processed_data[col] = 0.0 # Use float for numeric consistency

        # Ensure order of columns for consistent input to scaler
        processed_data = processed_data[MODELING_FEATURE_COLUMNS]


        categorical_cols = processed_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            processed_data[col] = processed_data[col].fillna('UNKNOWN')
            if isinstance(processed_data[col].dtype, pd.CategoricalDtype):
                processed_data[col] = processed_data[col].astype(str)

        numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce') # Ensure numeric
            processed_data[col] = processed_data[col].fillna(processed_data[col].median()) # Fill NaNs


        for col, encoder in self.label_encoders.items():
            if col in processed_data.columns:
                # Ensure the value is string before passing to encoder
                processed_data[col] = processed_data[col].astype(str)
                # Map unseen labels to a default (e.g., the first class seen during training)
                # This prevents ValueError for new, unseen categories in deployment
                known_classes = set(encoder.classes_)
                processed_data[col] = processed_data[col].apply(lambda x: x if x in known_classes else encoder.classes_[0])
                processed_data[col] = encoder.transform(processed_data[col])


        if self.scaler.feature_names_in_ is not None and not processed_data.empty:
            # Ensure the order of columns matches the scaler's trained features
            X_transformed = self.scaler.transform(processed_data[list(self.scaler.feature_names_in_)])
            X_transformed = pd.DataFrame(X_transformed, columns=list(self.scaler.feature_names_in_), index=processed_data.index)
        else:
            print("Scaler not fitted or processed data is empty, cannot transform new data.")
            return pd.DataFrame()

        return X_transformed
