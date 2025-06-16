# pipeline.py

import pandas as pd
import numpy as np
import warnings
import pickle
import sys
import os

# Add parent directory to sys.path to ensure modules are found
# This is crucial when running from a different directory or for certain deployment environments
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import modules and config directly
from config import (
    CBECS_DATA_FILE, ENERGY_COSTS, RETROFIT_COSTS,
    ECONOMIC_PARAMS, REGION_CLIMATE_ZONES, MODEL_OUTPUT_PATH,
    DEFAULT_TARGET_REGION, DEFAULT_TARGET_BUILDING_TYPES
)
from data_processing import DataProcessor
from feature_processing import FeatureProcessor
from model import HeatingRetrofitModel

warnings.filterwarnings('ignore')

class CBECSHeatingRetrofitPipeline:
    """
    Enhanced ML Pipeline for Identifying Commercial Buildings
    with High Heating Energy Retrofit Potential using CBECS 2018 Data

    Features:
    - Regional and building type customization
    - Economic analysis with ROI calculations
    - Integrates modular components for data, feature, and model processing.
    """

    def __init__(self, target_region=None, target_building_types=None, economic_params=None):
        self.data_processor = DataProcessor(target_region, target_building_types)
        self.feature_processor = FeatureProcessor(economic_params)
        self.model_trainer = HeatingRetrofitModel()
        self.processed_data_for_recs = None
        self.deployment_pipeline_components = None

    def generate_retrofit_recommendations(self):
        """Generate specific retrofit recommendations for buildings"""
        print("=== GENERATING RETROFIT RECOMMENDATIONS ===")

        building_data = self.feature_processor.get_processed_data()

        if building_data is None or building_data.empty:
            print("No processed data available or data is empty. Cannot generate recommendations.")
            return None, None

        high_potential = building_data[building_data['HIGH_RETROFIT_POTENTIAL'] == 1].copy()

        if len(high_potential) == 0:
            print("No buildings identified with high retrofit potential for recommendations.")
            return high_potential, [] # Return empty list of recommendations

        if 'ECONOMICALLY_FEASIBLE' not in high_potential.columns:
            print("Warning: 'ECONOMICALLY_FEASIBLE' not found. Ensure economic metrics were calculated.")
            high_potential['ECONOMICALLY_FEASIBLE'] = 0

        high_potential['NPV'] = high_potential['NPV'].fillna(0)
        high_potential['COMPREHENSIVE_RETROFIT_SCORE'] = (
            high_potential['HIGH_RETROFIT_POTENTIAL'] * 0.4 +
            high_potential['ECONOMICALLY_FEASIBLE'] * 0.6
        )

        high_potential = high_potential.sort_values(['COMPREHENSIVE_RETROFIT_SCORE', 'NPV'], ascending=[False, False])

        print(f"Top 10 Buildings for Heating Retrofits (Technical + Economic):")
        print("="*70)

        recommendations_list = []
        for idx, (_, building) in enumerate(high_potential.head(10).iterrows()):
            print(f"\nBuilding {idx+1}:")
            # Ensure attributes exist before trying to access or format them
            print(f"  Building Type: {building.get('PBA', 'Unknown')}")
            print(f"  Size: {building.get('SQFT', 0):,.0f} sq ft")
            print(f"  Age: {building.get('BUILDING_AGE', 'Unknown')} years")
            print(f"  Climate Zone: {building.get('PUBCLIM', 'Unknown')}")
            print(f"  Heating System: {building.get('EQUIPM', 'Unknown')}")
            print(f"  Fuel Type: {building.get('FUELHEAT', 'Unknown')}")
            print(f"  Current Heating Intensity: {building.get('HEATING_INTENSITY_SQFT', 0):.2f} BTU/sq ft")
            print(f"  Peer Average: {building.get('PEER_MEAN_INTENSITY', 0):.2f} BTU/sq ft")
            print(f"  Efficiency Score: {building.get('EFFICIENCY_SCORE', 0):.2f}")
            print(f"  Annual Energy Savings: ${building.get('ANNUAL_SAVINGS', 0):,.0f}")
            print(f"  Estimated Retrofit Cost: ${building.get('RETROFIT_COST_ESTIMATE', 0):,.0f}")
            print(f"  Simple Payback: {building.get('SIMPLE_PAYBACK', 0):.1f} years")
            print(f"  NPV (15-year): ${building.get('NPV', 0):,.0f}")
            print(f"  SIR: {building.get('SIR', 0):.2f}")

            specific_recs = self._generate_specific_recommendations(building)
            print(f"  Recommended Actions: {', '.join(specific_recs)}")
            recommendations_list.append({
                'BuildingID': building.name,
                'BuildingType': building.get('PBA', 'Unknown'),
                'SQFT': building.get('SQFT', 0),
                'AnnualSavings': building.get('ANNUAL_SAVINGS', 0),
                'RetrofitCost': building.get('RETROFIT_COST_ESTIMATE', 0),
                'SimplePayback': building.get('SIMPLE_PAYBACK', 0),
                'NPV': building.get('NPV', 0),
                'SIR': building.get('SIR', 0),
                'RecommendedActions': specific_recs
            })

        return high_potential, recommendations_list

    def _generate_specific_recommendations(self, building):
        """Generate specific retrofit recommendations based on building characteristics"""
        recommendations = []

        # Ensure relevant columns are accessed safely and are uppercase strings
        equipment = str(building.get('EQUIPM', '')).upper()
        building_age = building.get('BUILDING_AGE', 0)
        adq_insul = str(building.get('ADQINSUL', '')).upper()
        type_glass = str(building.get('TYPEGLASS', '')).upper()
        drafty = str(building.get('DRAFTY', '')).upper()
        sqft = building.get('SQFT', 0)
        occupy_p = building.get('OCCUPYP', 0)

        # Added safety for other potential columns needed for recommendations
        smrt_thrm = str(building.get('SMRTTHRM', 'NO')).upper()
        emcs = str(building.get('EMCS', 'NO')).upper()
        num_light = building.get('NUMLIGHT', 0)
        led_p = str(building.get('LEDP', 'NO')).upper() # Assuming LEDP indicates if LED lighting is present or percentage
        noctyp = str(building.get('NOCTYP', 'NO')).upper() # For occupancy controls

        # HVAC system recommendations
        if 'BOILER' in equipment and building_age > 20:
            recommendations.append('High-efficiency boiler upgrade')
        elif 'FURNACE' in equipment:
            recommendations.append('Heat pump conversion')
        elif 'PACKAGED' in equipment:
             recommendations.append('High-efficiency packaged HVAC replacement')

        # Envelope recommendations
        if adq_insul == 'INADEQUATE' or adq_insul == 'NONE':
            recommendations.append('Building envelope insulation')

        if type_glass == 'SINGLE_PANE':
            recommendations.append('Window replacement/upgrades')

        if drafty == 'YES':
            recommendations.append('Air sealing improvements')

        # Control system recommendations
        if smrt_thrm == 'NO': # Check for Smart Thermostat (using 'NO' as default from sample data)
            recommendations.append('Smart thermostat installation')
        if emcs == 'NO' and sqft > 20000: # For larger buildings without Energy Management System
            recommendations.append('Building Management System (BMS) implementation')

        # Lighting recommendations
        # Assuming NUMLIGHT > 0 and LEDP (LED Percentage) is 'NO' means not fully LED
        if num_light > 0 and led_p == 'NO':
            recommendations.append('LED lighting upgrade (reduces internal heat gains)')

        # Zone control for larger buildings
        if sqft > 25000 and noctyp == 'NO': # If building is large and doesn't have occupancy controls
            recommendations.append('Zone control systems and occupancy sensors')

        return recommendations[:4] # Limit to top N relevant recommendations

    def create_deployment_pipeline(self):
        """Create a deployment-ready prediction pipeline"""
        print("=== CREATING DEPLOYMENT PIPELINE ===")

        best_model = self.model_trainer.get_best_model()
        scaler = self.feature_processor.get_scaler()
        label_encoders = self.feature_processor.get_label_encoders()
        feature_importance = self.model_trainer.get_feature_importance()
        trained_feature_columns = None

        if hasattr(scaler, 'feature_names_in_'):
            trained_feature_columns = list(scaler.feature_names_in_)
        elif self.model_trainer.models and any(self.model_trainer.models.values()):
             first_model_result = next(iter(self.model_trainer.models.values()))
             if 'X_test' in first_model_result and first_model_result['X_test'] is not None:
                 trained_feature_columns = list(first_model_result['X_test'].columns)

        if best_model is None:
            print("Please train models first to create a deployment pipeline.")
            return None

        self.deployment_pipeline_components = {
            'model': best_model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_columns': trained_feature_columns, # Store these for consistent input to deployed model
            'feature_importance': feature_importance
        }

        print("Deployment pipeline created successfully!")
        print("Components included:")
        print("- Trained model")
        print("- Feature scaler")
        print("- Label encoders")
        print("- Feature column names (for consistent input order)")
        print("- Feature importance rankings")

        return self.deployment_pipeline_components

    def run_complete_pipeline(self, file_path=CBECS_DATA_FILE):
        """Run the complete pipeline from data loading to model deployment"""
        print("=== RUNNING COMPLETE CBECS HEATING RETROFIT PIPELINE ===")

        if not self.data_processor.load_data(file_path):
            print("Pipeline aborted due to data loading error.")
            return False

        self.data_processor.explore_data()

        filtered_data = self.data_processor.filter_by_region_and_type()
        if filtered_data is None or filtered_data.empty:
            print("No data left after filtering. Pipeline aborted.")
            return False

        heating_buildings = self.feature_processor.create_heating_efficiency_features(filtered_data)
        if heating_buildings is None or heating_buildings.empty:
            print("No valid heating buildings after feature creation. Pipeline aborted.")
            return False

        heating_buildings_with_economics = self.feature_processor.calculate_economic_metrics(heating_buildings)
        if heating_buildings_with_economics is None or heating_buildings_with_economics.empty:
            print("No economically viable buildings found after economic calculation. Pipeline aborted.")
            return False

        self.processed_data_for_recs = heating_buildings_with_economics

        X, y = self.feature_processor.preprocess_features(heating_buildings_with_economics)
        if X is None or y is None or X.empty:
            print("No features for modeling after preprocessing. Pipeline aborted.")
            return False

        results, X_test, y_test = self.model_trainer.train_models(X, y)

        if self.model_trainer.get_best_model() is not None: # Only tune if a model was successfully trained
            self.model_trainer.hyperparameter_tuning(X, y)
        else:
            print("Skipping hyperparameter tuning as no models were successfully trained.")


        self.generate_retrofit_recommendations()

        self.create_deployment_pipeline()

        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        print("Your model is ready for deployment!")

        return True


# --- Usage Example ---
if __name__ == "__main__":
    # Initialize pipeline with defaults from config
    pipeline = CBECSHeatingRetrofitPipeline(
        target_region=DEFAULT_TARGET_REGION,
        target_building_types=DEFAULT_TARGET_BUILDING_TYPES
    )

    if pipeline.run_complete_pipeline():
        deployment_components = pipeline.deployment_pipeline_components

        if deployment_components:
            try:
                with open(MODEL_OUTPUT_PATH, 'wb') as f:
                    pickle.dump(deployment_components, f)
                print(f"Model saved as '{MODEL_OUTPUT_PATH}' for deployment!")
            except Exception as e:
                print(f"Error saving model: {e}")

        print("\n--- DEMONSTRATING DEPLOYMENT USAGE ---")
        try:
            with open(MODEL_OUTPUT_PATH, 'rb') as f:
                loaded_pipeline_components = pickle.load(f)

            loaded_model = loaded_pipeline_components['model']
            loaded_scaler = loaded_pipeline_components['scaler']
            loaded_label_encoders = loaded_pipeline_components['label_encoders']
            loaded_feature_columns = loaded_pipeline_components['feature_columns']

            print(f"Loaded model type: {type(loaded_model).__name__}")
            print(f"Loaded scaler type: {type(loaded_scaler).__name__}")
            print(f"Number of label encoders loaded: {len(loaded_label_encoders)}")

            # Sample raw data for prediction
            # Ensure consistent casing and values with cleaning/standardization logic
            sample_raw_building_data = pd.DataFrame([{
                'PBA': 'OFFICE',
                'SQFT': 35000,
                'YRCONC': 1980,
                'NFLOOR': 5,
                'HEATHOME': 'YES', # Input should match expected values
                'MFHTBTU': 150000000,
                'HDD65': 4500,
                'CDD65': 1000,
                'FUELHEAT': 'NATURAL GAS',
                'EQUIPM': 'BOILER',
                'AIRCOND': 'YES',
                'PUBCLIM': 'COLD',
                'WLCNS': 'MASONRY', # Example, use actual CBECS values
                'RFCNS': 'BUILT_UP',
                'TYPEGLASS': 'SINGLE_PANE',
                'WINFRAME': 'ALUMINUM',
                'ADQINSUL': 'INADEQUATE',
                'DRAFTY': 'YES',
                'WKHRS': 80,
                'OPEN24': 'NO',
                'OCCUPYP': 0.8,
                'RENOV': 'NO',
                'SMRTTHRM': 'NO',
                'EMCS': 'NO',
                'NUMLIGHT': 50,
                'LEDP': 'NO',
                'NOCTYP': 'NO'
            }])

            def predict_with_loaded_model(raw_data_df, loaded_components, economic_params):
                temp_proc = FeatureProcessor(economic_params)

                # Apply data loading column cleaning logic to raw_data_df first
                raw_data_df.columns = raw_data_df.columns.str.strip()
                raw_data_df.columns = raw_data_df.columns.str.replace(r'\s+', '_', regex=True)
                raw_data_df.columns = raw_data_df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
                raw_data_df.columns = raw_data_df.columns.str.upper()

                engineered_data = temp_proc.create_heating_efficiency_features(raw_data_df.copy())
                if engineered_data is None or engineered_data.empty:
                    return "Error: Could not engineer features for prediction.", None, None

                engineered_data = temp_proc.calculate_economic_metrics(engineered_data)
                # Ensure engineered_data has all expected columns, reindex and fill if necessary
                data_for_transform = engineered_data.reindex(columns=loaded_components['feature_columns'], fill_value=np.nan)

                # Handle missing values & encode categorical using loaded components
                for col in data_for_transform.columns:
                    if col in loaded_components['label_encoders']:
                        data_for_transform[col] = data_for_transform[col].fillna('UNKNOWN').astype(str)
                        known_classes = set(loaded_components['label_encoders'][col].classes_)
                        # Map unseen labels to a known label (e.g., the first one)
                        data_for_transform[col] = data_for_transform[col].apply(lambda x: x if x in known_classes else loaded_components['label_encoders'][col].classes_[0])
                        data_for_transform[col] = loaded_components['label_encoders'][col].transform(data_for_transform[col])
                    elif data_for_transform[col].dtype == 'object':
                        data_for_transform[col] = data_for_transform[col].fillna('UNKNOWN').astype(str)

                numerical_cols = data_for_transform.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    data_for_transform[col] = pd.to_numeric(data_for_transform[col], errors='coerce')
                    data_for_transform[col] = data_for_transform[col].fillna(data_for_transform[col].median())

                if loaded_components['feature_columns'] and not data_for_transform.empty:
                    # Ensure column order matches the scaler's trained features
                    X_pred_scaled = loaded_components['scaler'].transform(data_for_transform[loaded_components['feature_columns']])
                else:
                    return "Error: No features to transform.", None, None

                pred_label = loaded_components['model'].predict(X_pred_scaled)[0]
                pred_proba = loaded_components['model'].predict_proba(X_pred_scaled)[0][1] if hasattr(loaded_components['model'], 'predict_proba') else None

                return pred_label, pred_proba, engineered_data.iloc[0]

            prediction_label, prediction_proba, engineered_building_data = predict_with_loaded_model(
                sample_raw_building_data.copy(), loaded_pipeline_components, ECONOMIC_PARAMS
            )

            print(f"\nPrediction for sample building:")
            print(f"  Raw Data: \n{sample_raw_building_data.to_string()}")
            print(f"  Predicted Retrofit Potential (1=High, 0=Low): {prediction_label}")
            if prediction_proba is not None:
                print(f"  Probability of High Potential: {prediction_proba:.4f}")

            if prediction_label == 1 and engineered_building_data is not None:
                print("\nRecommended Actions for this building:")
                temp_pipeline_for_recs = CBECSHeatingRetrofitPipeline()
                specific_recs = temp_pipeline_for_recs._generate_specific_recommendations(engineered_building_data)
                print(f"  {', '.join(specific_recs)}")

        except FileNotFoundError:
            print(f"Error: Model file '{MODEL_OUTPUT_PATH}' not found. Did the pipeline run successfully?")
        except Exception as e:
            print(f"An error occurred during deployment demonstration: {e}")
