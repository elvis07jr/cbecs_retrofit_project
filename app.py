# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os
import re # Needed for robust column cleaning in Streamlit app

# Add parent directory to sys.path to import modules
# This assumes your Streamlit app is run from the project root or similar structure
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import modules and config
from config import (
    MODEL_OUTPUT_PATH, ECONOMIC_PARAMS, REGION_CLIMATE_ZONES,
    MODELING_FEATURE_COLUMNS, ENERGY_COSTS, RETROFIT_COSTS
)
# Import the classes from their respective files
from data_processing import DataProcessor
from feature_processing import FeatureProcessor
from model import HeatingRetrofitModel
from pipeline import CBECSHeatingRetrofitPipeline # To access _generate_specific_recommendations

# --- Utility Functions ---

@st.cache_resource
def load_pipeline_components(model_path):
    """Loads the pre-trained model and preprocessing components."""
    try:
        with open(model_path, 'rb') as f:
            components = pickle.load(f)
        st.success("Machine Learning pipeline components loaded successfully!")
        return components
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found. Please ensure the pipeline.py has been run and the model saved.")
        st.stop() # Stop the app if the model isn't found
    except Exception as e:
        st.error(f"Error loading pipeline components: {e}")
        st.stop()

def preprocess_and_predict(raw_data_df, loaded_components, economic_params):
    """
    Processes raw building data, engineers features, and makes predictions.
    This function mirrors the processing steps from the training pipeline.
    """
    if raw_data_df.empty:
        st.info("No data provided for prediction.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Return empty DFs

    # Apply the same robust column cleaning as in DataProcessor.load_data
    raw_data_df.columns = raw_data_df.columns.str.strip()
    raw_data_df.columns = raw_data_df.columns.str.replace(r'\s+', '_', regex=True)
    raw_data_df.columns = raw_data_df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
    raw_data_df.columns = raw_data_df.columns.str.upper()

    # Initialize a temporary FeatureProcessor to apply feature engineering logic
    temp_feature_processor = FeatureProcessor(economic_params)

    # 1. Create heating efficiency features (includes HEATHOME standardization and numeric coercion)
    # This function expects the cleaned raw data directly
    engineered_data = temp_feature_processor.create_heating_efficiency_features(raw_data_df.copy())
    if engineered_data is None or engineered_data.empty:
        st.warning("Could not engineer heating efficiency features for the provided data. This might be due to missing critical columns or invalid data.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 2. Calculate economic metrics
    engineered_data = temp_feature_processor.calculate_economic_metrics(engineered_data)
    if engineered_data is None or engineered_data.empty:
        st.warning("Could not calculate economic metrics for the provided data.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Ensure all expected MODELING_FEATURE_COLUMNS are present and in correct order
    # Use reindex and fillna if any columns are missing in the input but expected by the model
    data_for_transform = engineered_data.reindex(columns=loaded_components['feature_columns'], fill_value=np.nan)

    # Handle missing values & encode categorical using loaded components (from training)
    for col in data_for_transform.columns:
        if col in loaded_components['label_encoders']:
            data_for_transform[col] = data_for_transform[col].fillna('UNKNOWN').astype(str) # Fill and ensure string type
            known_classes = set(loaded_components['label_encoders'][col].classes_)
            # Map unseen labels to a known label (e.g., the first class seen during training)
            data_for_transform[col] = data_for_transform[col].apply(lambda x: x if x in known_classes else loaded_components['label_encoders'][col].classes_[0])
            data_for_transform[col] = loaded_components['label_encoders'][col].transform(data_for_transform[col])
        elif data_for_transform[col].dtype == 'object':
            data_for_transform[col] = data_for_transform[col].fillna('UNKNOWN').astype(str) # Just fill for generic objects

    numerical_cols = data_for_transform.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        data_for_transform[col] = pd.to_numeric(data_for_transform[col], errors='coerce') # Ensure numeric
        data_for_transform[col] = data_for_transform[col].fillna(data_for_transform[col].median()) # Fill NaNs with median

    # Scale using the loaded scaler
    X_pred_scaled = pd.DataFrame() # Initialize empty DataFrame
    if loaded_components['feature_columns'] and not data_for_transform.empty:
        try:
            # Ensure order of columns matches the scaler's trained features
            X_pred_scaled = loaded_components['scaler'].transform(data_for_transform[loaded_components['feature_columns']])
        except Exception as e:
            st.error(f"Error during scaling: {e}. Ensure input features match training features.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    else:
        st.warning("No features to transform for prediction.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


    # Make prediction
    predictions = loaded_components['model'].predict(X_pred_scaled)
    probabilities = loaded_components['model'].predict_proba(X_pred_scaled)[:, 1] if hasattr(loaded_components['model'], 'predict_proba') else None

    # Add predictions and probabilities back to the engineered data
    engineered_data['PREDICTED_RETROFIT_POTENTIAL'] = predictions
    if probabilities is not None:
        engineered_data['PROBABILITY_HIGH_POTENTIAL'] = probabilities

    high_potential_buildings = engineered_data[engineered_data['PREDICTED_RETROFIT_POTENTIAL'] == 1]

    recommendations_summary = []
    # Instantiate the pipeline object to access the _generate_specific_recommendations method
    temp_pipeline_for_recs = CBECSHeatingRetrofitPipeline() # Temporary instance to call method
    for idx, row in high_potential_buildings.iterrows():
        recs = temp_pipeline_for_recs._generate_specific_recommendations(row)
        recommendations_summary.append({
            'Building Type': row.get('PBA', 'N/A'),
            'SQFT': f"{row.get('SQFT', 0):,.0f}",
            'Annual Savings ($)': f"{row.get('ANNUAL_SAVINGS', 0):,.0f}",
            'Retrofit Cost ($)': f"{row.get('RETROFIT_COST_ESTIMATE', 0):,.0f}",
            'Simple Payback (Years)': f"{row.get('SIMPLE_PAYBACK', 0):.1f}",
            'NPV ($)': f"{row.get('NPV', 0):,.0f}",
            'SIR': f"{row.get('SIR', 0):.2f}",
            'Recommended Actions': ", ".join(recs)
        })

    return engineered_data, high_potential_buildings, pd.DataFrame(recommendations_summary)


# --- Streamlit App Layout ---

st.set_page_config(
    page_title="CBECS Heating Retrofit Potential",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💡 Commercial Building Heating Retrofit Potential Analyzer")
st.markdown("""
This application leverages machine learning to identify commercial buildings with high heating energy retrofit potential
based on CBECS 2018 data characteristics.
""")

# Load pipeline components once
loaded_pipeline_components = load_pipeline_components(MODEL_OUTPUT_PATH)

# Sidebar for controls
st.sidebar.header("Configuration")

app_mode = st.sidebar.radio(
    "Choose Application Mode",
    ["Upload CBECS Data (Batch Prediction)", "Predict for a Single Building"]
)

if app_mode == "Upload CBECS Data (Batch Prediction)":
    st.sidebar.subheader("Upload CBECS Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)
            st.success(f"File '{uploaded_file.name}' loaded successfully. Shape: {input_df.shape}")
            st.subheader("Raw Data Preview")
            st.dataframe(input_df.head())

            # Perform prediction on the uploaded data
            with st.spinner("Analyzing building data and predicting retrofit potential..."):
                processed_data, high_potential_data, recommendations_df = preprocess_and_predict(
                    input_df, loaded_pipeline_components, ECONOMIC_PARAMS
                )

            if not processed_data.empty:
                st.subheader("Analysis Results")
                st.write(f"Total buildings analyzed: {len(processed_data)}")
                st.write(f"Buildings with high retrofit potential: {len(high_potential_data)}")
                if len(processed_data) > 0:
                    st.write(f"Percentage with high potential: {len(high_potential_data) / len(processed_data) * 100:.2f}%")
                else:
                    st.write("Percentage with high potential: 0.00%")


                if not high_potential_data.empty:
                    st.subheader("Top Buildings for Retrofit (High Potential)")
                    st.dataframe(recommendations_df)

                    st.subheader("Feature Importance (from trained model)")
                    if loaded_pipeline_components.get('feature_importance') is not None:
                        st.dataframe(loaded_pipeline_components['feature_importance'].head(15))
                    else:
                        st.info("Feature importance not available for the selected model type.")
                else:
                    st.info("No buildings with high retrofit potential found in the uploaded data based on current criteria.")
            else:
                st.warning("No data processed for prediction. Please check your input file.")

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            st.info("Please ensure your file has the expected CBECS 2018 column names like 'SQFT', 'MFHTBTU', 'PBA', 'HDD65', etc.")

elif app_mode == "Predict for a Single Building":
    st.sidebar.subheader("Enter Building Characteristics")

    with st.form("single_building_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Building Basics")
            pba_options = [ # These should ideally be derived from actual data values during training
                'OFFICE', 'RETAIL_ENCLOSED_MALL', 'RETAIL_STRIP_SHOPPING_CENTER',
                'WAREHOUSE_AND_STORAGE', 'SERVICE', 'OTHER', 'HEALTHCARE_INPATIENT',
                'EDUCATION', 'PUBLIC_ASSEMBLY', 'FOOD_SERVICE', 'LODGING', 'RELIGIOUS_WORSHIP',
                'INPATIENT_HEALTHCARE', 'OUTPATIENT_HEALTHCARE', 'LIBRARY', 'LABORATORY',
                'SPORTS_ENTERTAINMENT', 'PUBLIC_ORDER_AND_SAFETY', 'MERCANTILE_NO_MALL',
                'NON_REFRIGERATED_WAREHOUSE', 'REFRIGERATED_WAREHOUSE'
            ]
            pba = st.selectbox("Building Type (PBA)", options=pba_options, index=0)
            sqft = st.number_input("Total Square Footage (SQFT)", min_value=100, max_value=1000000, value=25000)
            yrconc = st.slider("Year Constructed (YRCONC)", min_value=1900, max_value=2018, value=1980)
            nfloor = st.number_input("Number of Floors (NFLOOR)", min_value=1, max_value=100, value=3)

        with col2:
            st.markdown("### Energy Use & Climate")
            mfhtbtu = st.number_input("Main Heating Energy (MFHTBTU) [Btu]", min_value=0, value=100000000)
            hdd65 = st.number_input("Heating Degree Days (HDD65)", min_value=0, value=4000)
            cdd65 = st.number_input("Cooling Degree Days (CDD65)", min_value=0, value=1500)
            fuelheat_options = [fuel.upper().replace(' ', '_') for fuel in list(ENERGY_COSTS.keys())] + ['OTHER']
            fuelheat = st.selectbox("Main Heating Fuel (FUELHEAT)", options=fuelheat_options, index=0)
            equipm_options = [
                'BOILER', 'FURNACE', 'HEAT_PUMP', 'ELECTRIC_RESISTANCE', 'DISTRICT_HEAT',
                'OTHER_SPACE_HEATING', 'PACKAGED_HVAC_UNIT', 'CENTRAL_FURNACE'
            ]
            equipm = st.selectbox("Main Heating Equipment (EQUIPM)", options=equipm_options, index=0)
            # Use the values from config.REGION_CLIMATE_ZONES
            flat_climate_zones = sorted(list(set([item for sublist in REGION_CLIMATE_ZONES.values() for item in sublist])))
            pubclim = st.selectbox("Public Climate Zone (PUBCLIM)", options=flat_climate_zones, index=0)

        with col3:
            st.markdown("### Envelope & Operations")
            aircond = st.radio("Has Air Conditioning (AIRCOND)?", ['YES', 'NO'], index=0)
            wlcons = st.selectbox("Wall Construction (WLCNS)", options=['MASONRY', 'WOOD', 'CONCRETE', 'GLASS', 'OTHER', 'UNKNOWN'])
            rfcons = st.selectbox("Roof Construction (RFCNS)", options=['BUILT_UP', 'METAL', 'SINGLE_PLY', 'OTHER', 'UNKNOWN'])
            typeglass = st.selectbox("Window Glass Type (TYPEGLASS)", options=['SINGLE_PANE', 'DOUBLE_PANE', 'TRIPLE_PANE', 'OTHER', 'UNKNOWN'])
            winframe = st.selectbox("Window Frame Type (WINFRAME)", options=['ALUMINUM', 'WOOD', 'VINYL', 'OTHER', 'UNKNOWN'])
            adqinsul = st.radio("Adequate Insulation (ADQINSUL)?", ['ADEQUATE', 'INADEQUATE', 'NONE'], index=1)
            drafty = st.radio("Building is Drafty (DRAFTY)?", ['YES', 'NO'], index=1)
            wkhrs = st.number_input("Weekly Operating Hours (WKHRS)", min_value=0, max_value=168, value=40)
            open24 = st.radio("Operates 24/7 (OPEN24)?", ['YES', 'NO'], index=1)
            occupyp = st.slider("Occupancy Percentage (OCCUPYP)", min_value=0.0, max_value=1.0, value=0.7)
            renov = st.radio("Recent Renovation (RENOV)?", ['YES', 'NO'], index=1)

            # Columns used in recommendations, even if not modeling features directly
            smrt_thrm = st.radio("Has Smart Thermostat (SMRTTHRM)?", ['YES', 'NO'], index=1)
            emcs = st.radio("Has Energy Management Control System (EMCS)?", ['YES', 'NO'], index=1)
            num_light = st.number_input("Number of Lights (NUMLIGHT)", min_value=0, value=50)
            led_p = st.radio("Has LED Lighting Present (LEDP)?", ['YES', 'NO'], index=1) # Simplified for example
            noctyp = st.radio("Has Occupancy Control System (NOCTYP)?", ['YES', 'NO'], index=1) # Simplified for example


        submitted = st.form_submit_button("Predict Retrofit Potential")

    if submitted:
        input_data = pd.DataFrame([{
            'PBA': pba,
            'SQFT': sqft,
            'YRCONC': yrconc,
            'NFLOOR': nfloor,
            'HEATHOME': 'YES', # Always assume 'YES' for this path for now as it's for retrofit potential
            'MFHTBTU': mfhtbtu,
            'HDD65': hdd65,
            'CDD65': cdd65,
            'FUELHEAT': fuelheat,
            'EQUIPM': equipm,
            'AIRCOND': aircond,
            'PUBCLIM': pubclim,
            'WLCNS': wlcons,
            'RFCNS': rfcons,
            'TYPEGLASS': typeglass,
            'WINFRAME': winframe,
            'ADQINSUL': adqinsul,
            'DRAFTY': drafty,
            'WKHRS': wkhrs,
            'OPEN24': open24,
            'OCCUPYP': occupyp,
            'RENOV': renov,
            'SMRTTHRM': smrt_thrm,
            'EMCS': emcs,
            'NUMLIGHT': num_light,
            'LEDP': led_p,
            'NOCTYP': noctyp
        }])

        with st.spinner("Predicting retrofit potential for your building..."):
            processed_data, high_potential_data, recommendations_df = preprocess_and_predict(
                input_data, loaded_pipeline_components, ECONOMIC_PARAMS
            )

        if not processed_data.empty:
            prediction_label = processed_data['PREDICTED_RETROFIT_POTENTIAL'].iloc[0]
            prediction_proba = processed_data['PROBABILITY_HIGH_POTENTIAL'].iloc[0] if 'PROBABILITY_HIGH_POTENTIAL' in processed_data.columns else None

            st.subheader("Prediction Result")
            if prediction_label == 1:
                st.success(f"This building has **HIGH** heating energy retrofit potential!")
                if prediction_proba is not None:
                    st.write(f"Confidence: {prediction_proba:.2f}")
            else:
                st.info(f"This building has **LOW** heating energy retrofit potential.")
                if prediction_proba is not None:
                    st.write(f"Confidence (of high potential): {prediction_proba:.2f}")

            if prediction_label == 1 and not recommendations_df.empty:
                st.subheader("Recommended Retrofit Actions and Economic Metrics:")
                st.dataframe(recommendations_df) # Show all columns here for clarity

                st.markdown("---")
                st.markdown("### Detailed Economic Projections for this Building:")
                # Safely get values, handling cases where they might not exist after filtering
                st.write(f"**Annual Energy Savings:** ${processed_data.get('ANNUAL_SAVINGS', pd.Series([0])).iloc[0]:,.0f}")
                st.write(f"**Estimated Retrofit Cost:** ${processed_data.get('RETROFIT_COST_ESTIMATE', pd.Series([0])).iloc[0]:,.0f}")
                st.write(f"**Simple Payback Period:** {processed_data.get('SIMPLE_PAYBACK', pd.Series([0])).iloc[0]:.1f} years")
                st.write(f"**Net Present Value (15-year):** ${processed_data.get('NPV', pd.Series([0])).iloc[0]:,.0f}")
                st.write(f"**Savings-to-Investment Ratio (SIR):** {processed_data.get('SIR', pd.Series([0])).iloc[0]:.2f}")
                if processed_data.get('ECONOMICALLY_FEASIBLE', pd.Series([0])).iloc[0] == 1:
                    st.success("This retrofit is **economically feasible**!")
                else:
                    st.warning("This retrofit is currently **not economically feasible** based on projections.")
            else:
                st.info("No specific recommendations generated as the building does not have high retrofit potential, or data was insufficient.")
        else:
            st.warning("Could not generate prediction. Please check your inputs, especially critical values like SQFT, MFHTBTU, HDD65, YRCONC, and ensure 'HEATHOME' is set to 'YES' for retrofit analysis.")
