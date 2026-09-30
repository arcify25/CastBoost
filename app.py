import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load('concrete_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'concrete_model.pkl' not found. Run 'train_model.py' first.")
    st.stop()

st.title("🏗️ Concrete Compressive Strength Predictor")
st.markdown("Estimate concrete compressive strength ($MPa$) using machine learning with derived mixture ratios.")

tab1, tab2 = st.tabs(["🎛️ Single Mix Prediction", "📁 Batch CSV Prediction"])

feature_order = [
    'cement', 'slag', 'flyash', 'water', 'superplasticizer', 
    'coarseagg', 'fineagg', 'age', 'water_cement_ratio', 
    'slag_cement_ratio', 'flyash_cement_ratio', 'total_binder'
]

# ==========================================
# TAB 1: SINGLE MIX PREDICTION
# ==========================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧪 Cementitious Materials (kg/m³)")
        cement = st.slider("Cement", min_value=100.0, max_value=550.0, value=280.0, step=5.0)
        slag = st.slider("Blast Furnace Slag", min_value=0.0, max_value=360.0, value=0.0, step=5.0)
        flyash = st.slider("Fly Ash", min_value=0.0, max_value=200.0, value=0.0, step=5.0)
        superplasticizer = st.slider("Superplasticizer", min_value=0.0, max_value=35.0, value=2.5, step=0.5)

    with col2:
        st.subheader("🪨 Aggregates & Curing")
        water = st.slider("Water (kg/m³)", min_value=120.0, max_value=250.0, value=185.0, step=2.0)
        coarseagg = st.slider("Coarse Aggregate (kg/m³)", min_value=800.0, max_value=1150.0, value=1000.0, step=10.0)
        fineagg = st.slider("Fine Aggregate (kg/m³)", min_value=590.0, max_value=1000.0, value=750.0, step=10.0)
        age = st.slider("Age / Curing Time (Days)", min_value=1, max_value=365, value=28, step=1)

    # Compute derived physical ratios
    input_data = {
        'cement': cement,
        'slag': slag,
        'flyash': flyash,
        'water': water,
        'superplasticizer': superplasticizer,
        'coarseagg': coarseagg,
        'fineagg': fineagg,
        'age': age,
        'water_cement_ratio': water / cement,
        'slag_cement_ratio': slag / cement,
        'flyash_cement_ratio': flyash / cement,
        'total_binder': cement + slag + flyash
    }
    input_df = pd.DataFrame([input_data])[feature_order]

    st.divider()

    if st.button("🚀 Calculate Compressive Strength", use_container_width=True):
        prediction = float(model.predict(input_df)[0])
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(label="Predicted Strength", value=f"{prediction:.2f} MPa")
            
        with res_col2:
            if prediction < 20.0:
                st.warning("⚠️ **Low Strength Concrete** (< 20 MPa): Suitable for unreinforced slabs, footpaths, and sub-bases.")
            elif 20.0 <= prediction < 40.0:
                st.success("✅ **Standard Structural Concrete** (20 - 40 MPa): Suitable for residential columns, beams, and slabs.")
            else:
                st.info("🌟 **High-Performance Concrete** (> 40 MPa): Suitable for high-rise frames, highway bridges, and precast girders.")

    # Model Explainability Plot
    with st.expander("📊 Model Feature Importance", expanded=True):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(range(len(indices)), importances[indices], color="#FF4B4B", align="center")
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_order[i] for i in indices])
            ax.set_xlabel("Relative Importance Score")
            ax.set_title("Ingredient & Derived Ratio Influence")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Feature importance visualization is not supported by the currently loaded model.")

    with st.expander("🔍 View Calculated Mixture & Ratios Summary"):
        st.dataframe(input_df.style.format("{:.2f}"))

# ==========================================
# TAB 2: BATCH CSV PREDICTION
# ==========================================
with tab2:
    st.subheader("📂 Batch Process Multiple Concrete Mixes")
    st.write("Upload a CSV file with the following raw columns: `cement, slag, flyash, water, superplasticizer, coarseagg, fineagg, age`")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        base_cols = {'cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseagg', 'fineagg', 'age'}
        
        if base_cols.issubset(set(batch_df.columns)):
            # Calculate derived features for the batch
            batch_df['water_cement_ratio'] = batch_df['water'] / batch_df['cement']
            batch_df['slag_cement_ratio'] = batch_df['slag'] / batch_df['cement']
            batch_df['flyash_cement_ratio'] = batch_df['flyash'] / batch_df['cement']
            batch_df['total_binder'] = batch_df['cement'] + batch_df['slag'] + batch_df['flyash']
            
            # Predict
            predictions = model.predict(batch_df[feature_order])
            batch_df['Predicted_Strength_MPa'] = np.round(predictions, 2)
            
            def classify(val):
                if val < 20.0:
                    return 'Low Strength (< 20 MPa)'
                elif 20.0 <= val < 40.0:
                    return 'Standard Structural (20-40 MPa)'
                return 'High Performance (> 40 MPa)'

            batch_df['Category'] = batch_df['Predicted_Strength_MPa'].apply(classify)
            
            st.success(f"Successfully processed {len(batch_df)} mixture samples.")
            st.dataframe(batch_df, use_container_width=True)
            
            csv_output = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_output,
                file_name="concrete_predictions_output.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            missing = base_cols - set(batch_df.columns)
            st.error(f"Missing required columns in CSV: {', '.join(missing)}")