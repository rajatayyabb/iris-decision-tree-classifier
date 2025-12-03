# ============================================
# STREAMLIT APP FOR IRIS DECISION TREE CLASSIFIER
# File name: app.py
# Save this file as app.py in your project folder
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide"
)

# Title and description
st.title("🌸 Iris Flower Classification")
st.markdown("""
This app uses a **Decision Tree Classifier** to predict the species of Iris flowers 
based on their measurements.
""")

# Load the model and encoders
@st.cache_resource
def load_model():
    try:
        with open('decision_tree_classifier_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('label_encoder.pkl', 'rb') as file:
            le = pickle.load(file)
        with open('feature_names.pkl', 'rb') as file:
            features = pickle.load(file)
        return model, le, features
    except Exception as e:
        st.error(f"⚠️ Error loading model files: {e}")
        st.info("Please make sure these files are in the same folder as app.py:")
        st.code("""
- decision_tree_classifier_model.pkl
- label_encoder.pkl
- feature_names.pkl
        """)
        return None, None, None

model, label_encoder, feature_names = load_model()

if model is not None:
    # Sidebar for input
    st.sidebar.header("🔧 Input Features")
    st.sidebar.markdown("Adjust the sliders to input flower measurements:")

    # Input sliders
    sepal_length = st.sidebar.slider(
        "Sepal Length (cm)", 
        min_value=4.0, 
        max_value=8.0, 
        value=5.8, 
        step=0.1
    )

    sepal_width = st.sidebar.slider(
        "Sepal Width (cm)", 
        min_value=2.0, 
        max_value=4.5, 
        value=3.0, 
        step=0.1
    )

    petal_length = st.sidebar.slider(
        "Petal Length (cm)", 
        min_value=1.0, 
        max_value=7.0, 
        value=4.3, 
        step=0.1
    )

    petal_width = st.sidebar.slider(
        "Petal Width (cm)", 
        min_value=0.1, 
        max_value=2.5, 
        value=1.3, 
        step=0.1
    )

    # Create feature array
    features_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Display input features
    st.subheader("📊 Input Features")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sepal Length", f"{sepal_length} cm")
    with col2:
        st.metric("Sepal Width", f"{sepal_width} cm")
    with col3:
        st.metric("Petal Length", f"{petal_length} cm")
    with col4:
        st.metric("Petal Width", f"{petal_width} cm")

    # Make prediction
    if st.sidebar.button("🔍 Predict", type="primary"):
        # Get prediction
        prediction = model.predict(features_input)
        probability = model.predict_proba(features_input)
        
        species_names = label_encoder.classes_
        predicted_species = species_names[prediction[0]]
        
        # Display prediction
        st.subheader("🎯 Prediction Result")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="background-color: #1f77b4; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">Predicted Species</h2>
                <h1 style="color: white; margin: 10px 0; font-size: 48px;">{predicted_species}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create probability bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=species_names,
                    y=probability[0] * 100,
                    text=[f'{p:.2f}%' for p in probability[0] * 100],
                    textposition='auto',
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
                )
            ])
            
            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Species",
                yaxis_title="Probability (%)",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display probability table
        st.subheader("📈 Detailed Probabilities")
        prob_df = pd.DataFrame({
            'Species': species_names,
            'Probability': [f'{p:.4f}' for p in probability[0]],
            'Percentage': [f'{p*100:.2f}%' for p in probability[0]]
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        
        # Show species information
        st.subheader("🌺 Species Information")
        
        if predicted_species == "Iris-setosa":
            st.info("""
            **Iris Setosa**: 
            - Smallest petals among the three species
            - Native to Alaska, Canada, and northeastern USA
            - Typically found in grassy meadows and bogs
            """)
        elif predicted_species == "Iris-versicolor":
            st.success("""
            **Iris Versicolor**: 
            - Medium-sized petals
            - Also known as "Blue Flag"
            - Native to eastern North America
            """)
        else:
            st.warning("""
            **Iris Virginica**: 
            - Largest petals among the three species
            - Native to eastern United States
            - Often found in wetlands and along streams
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Built with Streamlit | Decision Tree Classifier | Lab 09 Task 2</p>
</div>
""", unsafe_allow_html=True)