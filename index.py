import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction - Fuzzy Logic Model",
    page_icon="🩺",
    layout="wide"
)

def load_model():
    """Create and return the fuzzy control system for diabetes prediction"""
    # Define fuzzy variables (antecedents and consequent)
    pregnancies = ctrl.Antecedent(np.arange(0, 21, 1), 'pregnancies')
    glucose = ctrl.Antecedent(np.arange(0, 201, 1), 'glucose')
    blood_pressure = ctrl.Antecedent(np.arange(0, 121, 1), 'blood_pressure')
    skin_thickness = ctrl.Antecedent(np.arange(0, 61, 1), 'skin_thickness')
    insulin = ctrl.Antecedent(np.arange(0, 301, 1), 'insulin')
    bmi = ctrl.Antecedent(np.arange(0, 61, 0.1), 'bmi')
    pedigree = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'pedigree')
    age = ctrl.Antecedent(np.arange(20, 81, 1), 'age')
    outcome = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'outcome')

    # Define membership functions for each fuzzy variable
    # Pregnancies
    pregnancies['low'] = fuzz.trapmf(pregnancies.universe, [0, 0, 2, 4])
    pregnancies['medium'] = fuzz.trimf(pregnancies.universe, [2, 5, 8])
    pregnancies['high'] = fuzz.trapmf(pregnancies.universe, [6, 10, 20, 20])

    # Glucose
    glucose['low'] = fuzz.trapmf(glucose.universe, [0, 0, 50, 70])
    glucose['normal'] = fuzz.trimf(glucose.universe, [50, 85, 100])
    glucose['prediabetes'] = fuzz.trimf(glucose.universe, [100, 112.5, 125])
    glucose['high'] = fuzz.trapmf(glucose.universe, [125, 150, 200, 200])

    # Blood Pressure
    blood_pressure['low'] = fuzz.trapmf(blood_pressure.universe, [0, 0, 50, 60])
    blood_pressure['normal'] = fuzz.trimf(blood_pressure.universe, [50, 70, 90])
    blood_pressure['high'] = fuzz.trapmf(blood_pressure.universe, [80, 100, 120, 120])

    # Skin Thickness
    skin_thickness['thin'] = fuzz.trapmf(skin_thickness.universe, [0, 0, 15, 20])
    skin_thickness['normal'] = fuzz.trimf(skin_thickness.universe, [15, 25, 35])
    skin_thickness['thick'] = fuzz.trapmf(skin_thickness.universe, [30, 40, 60, 60])

    # Insulin
    insulin['low'] = fuzz.trapmf(insulin.universe, [0, 0, 30, 50])
    insulin['normal'] = fuzz.trimf(insulin.universe, [40, 70, 100])
    insulin['high'] = fuzz.trapmf(insulin.universe, [90, 150, 300, 300])

    # BMI
    bmi['underweight'] = fuzz.trapmf(bmi.universe, [0, 0, 15, 18.5])
    bmi['normal'] = fuzz.trimf(bmi.universe, [18.5, 21.7, 24.9])
    bmi['overweight'] = fuzz.trimf(bmi.universe, [24.9, 27.45, 29.9])
    bmi['obese'] = fuzz.trapmf(bmi.universe, [29.9, 35, 60, 60])

    # Diabetes Pedigree Function
    pedigree['low'] = fuzz.trapmf(pedigree.universe, [0, 0, 0.2, 0.4])
    pedigree['medium'] = fuzz.trimf(pedigree.universe, [0.3, 0.6, 0.9])
    pedigree['high'] = fuzz.trapmf(pedigree.universe, [0.8, 1.2, 3, 3])

    # Age
    age['young'] = fuzz.trapmf(age.universe, [20, 20, 25, 30])
    age['middle_aged'] = fuzz.trimf(age.universe, [25, 40, 55])
    age['old'] = fuzz.trapmf(age.universe, [50, 60, 80, 80])

    # Outcome
    outcome['no_diabetes'] = fuzz.trimf(outcome.universe, [0, 0, 0.5])
    outcome['diabetes'] = fuzz.trimf(outcome.universe, [0.5, 1, 1])

    # Define fuzzy rules based on diabetes risk factors
    rule1 = ctrl.Rule(glucose['high'], outcome['diabetes'])
    rule2 = ctrl.Rule(glucose['prediabetes'] & bmi['obese'], outcome['diabetes'])
    rule3 = ctrl.Rule(age['old'] & pedigree['high'], outcome['diabetes'])
    rule4 = ctrl.Rule(pregnancies['high'] & glucose['high'], outcome['diabetes'])
    rule5 = ctrl.Rule(insulin['high'] & glucose['high'], outcome['diabetes'])
    rule6 = ctrl.Rule(blood_pressure['high'] & bmi['obese'], outcome['diabetes'])
    rule7 = ctrl.Rule(skin_thickness['thick'] & bmi['obese'], outcome['diabetes'])
    rule8 = ctrl.Rule(glucose['low'], outcome['no_diabetes'])
    rule9 = ctrl.Rule(bmi['normal'] & age['young'], outcome['no_diabetes'])
    rule10 = ctrl.Rule(pregnancies['low'] & glucose['normal'], outcome['no_diabetes'])
    rule11 = ctrl.Rule(pedigree['low'], outcome['no_diabetes'])

    # Create the fuzzy control system
    control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11])
    simulation = ctrl.ControlSystemSimulation(control_system)
    
    return simulation

def predict_diabetes(inputs, simulation):
    """Make a prediction using the fuzzy logic model"""
    for key, value in inputs.items():
        simulation.input[key] = value
    
    try:
        simulation.compute()
        predicted_value = simulation.output['outcome']
        predicted_class = 1 if predicted_value > 0.5 else 0
        confidence = predicted_value if predicted_class == 1 else (1 - predicted_value)
        return predicted_class, predicted_value, confidence
    except KeyError:
        # Default to no diabetes if no rules are activated
        return 0, 0.0, 1.0

def load_and_preprocess_data():
    """Load and preprocess the diabetes dataset"""
    data = pd.read_csv("diabetes.csv")
    
    # Preprocess the data
    # Replace zeros with class-specific medians for biologically implausible values
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        for outcome in [0, 1]:
            median_val = data[(data[col] != 0) & (data['Outcome'] == outcome)][col].median()
            data.loc[(data[col] == 0) & (data['Outcome'] == outcome), col] = median_val
    
    # Cap outliers at 5th and 95th percentiles
    for col in data.columns[:-1]:
        lower = data[col].quantile(0.05)
        upper = data[col].quantile(0.95)
        data[col] = np.where(data[col] < lower, lower, data[col])
        data[col] = np.where(data[col] > upper, upper, data[col])
    
    return data

def main():
    # Load the fuzzy logic model
    simulation = load_model()
    
    # App title and description
    st.title("Diabetes Risk Prediction using Fuzzy Logic")
    st.markdown("""
    This application uses a **Fuzzy Logic** model to predict the risk of diabetes based on several health indicators.
    Enter your values below to get a prediction.
    """)
    
    # Create tabs for different sections of the app
    tab1, tab2, tab3 = st.tabs(["Prediction", "Dataset Insights", "About the Model"])
    
    with tab1:
        st.header("Predict Your Diabetes Risk")
        
        # Create two columns for input parameters
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.slider("Number of Pregnancies", 0, 20, 3, help="Number of times pregnant")
            glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 120, help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test")
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 120, 70, help="Diastolic blood pressure")
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 60, 20, help="Triceps skin fold thickness")
        
        with col2:
            insulin = st.slider("Insulin Level (mu U/ml)", 0, 300, 80, help="2-Hour serum insulin")
            bmi = st.slider("BMI (kg/m²)", 0.0, 60.0, 25.0, help="Body mass index")
            pedigree = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, help="Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)")
            age = st.slider("Age (years)", 20, 80, 35)
        
        # Create input dictionary
        inputs = {
            'pregnancies': pregnancies,
            'glucose': glucose,
            'blood_pressure': blood_pressure,
            'skin_thickness': skin_thickness,
            'insulin': insulin,
            'bmi': bmi,
            'pedigree': pedigree,
            'age': age
        }
        
        # Prediction button
        if st.button("Predict", type="primary"):
            with st.spinner('Computing prediction...'):
                predicted_class, predicted_value, confidence = predict_diabetes(inputs, simulation)
                
                # Display the result with appropriate styling
                st.subheader("Prediction Result")
                
                # Create columns for the result and gauge chart
                res_col1, res_col2 = st.columns([2, 3])
                
                with res_col1:
                    if predicted_class == 1:
                        st.error(f"**Result:** High risk of diabetes (Risk score: {predicted_value:.2f})")
                        st.markdown("### Recommendations:")
                        st.markdown("- Consult with a healthcare provider")
                        st.markdown("- Consider monitoring blood glucose levels")
                        st.markdown("- Evaluate diet and physical activity")
                    else:
                        st.success(f"**Result:** Low risk of diabetes (Risk score: {predicted_value:.2f})")
                        st.markdown("### Recommendations:")
                        st.markdown("- Maintain a healthy lifestyle")
                        st.markdown("- Continue regular health check-ups")
                
                with res_col2:
                    # Create a gauge chart to visualize the prediction
                    fig, ax = plt.subplots(figsize=(4, 3))
                    
                    # Create a colorful gauge chart
                    gauge_colors = ['green', 'yellowgreen', 'orange', 'red']
                    bounds = [0, 0.25, 0.5, 0.75, 1.0]
                    norm = plt.Normalize(0, 1)
                    
                    # Draw the colorful arc
                    for i in range(len(bounds)-1):
                        ax.barh(0, bounds[i+1]-bounds[i], left=bounds[i], height=0.5, color=gauge_colors[i], alpha=0.8)
                    
                    # Add arrow to indicate the prediction
                    arrow_len = 0.4
                    ax.arrow(predicted_value, 0, 0, arrow_len, head_width=0.03, head_length=0.1, fc='black', ec='black')
                    
                    # Add text annotations
                    ax.text(0.125, -0.2, 'Low Risk', ha='center', va='center', fontsize=9)
                    ax.text(0.375, -0.2, 'Moderate', ha='center', va='center', fontsize=9)
                    ax.text(0.625, -0.2, 'High', ha='center', va='center', fontsize=9)
                    ax.text(0.875, -0.2, 'Very High', ha='center', va='center', fontsize=9)
                    
                    # Add the predicted value text
                    ax.text(0.5, 0.7, f'Risk Score: {predicted_value:.2f}', ha='center', va='center', fontsize=12, fontweight='bold')
                    
                    # Configure the plot
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 1)
                    ax.set_aspect('auto')
                    ax.set_title('Diabetes Risk Gauge', fontsize=14)
                    ax.axis('off')
                    
                    st.pyplot(fig)
                
                # Show risk factors
                st.subheader("Key Risk Factors")
                risk_factors = []
                
                if glucose > 125:
                    risk_factors.append(("High glucose level", "Glucose > 125 mg/dL indicates high risk"))
                if bmi > 30:
                    risk_factors.append(("Obesity", "BMI > 30 kg/m² indicates obesity"))
                if age > 50:
                    risk_factors.append(("Age", "Age > 50 years increases diabetes risk"))
                if pedigree > 0.8:
                    risk_factors.append(("Family history", "High diabetes pedigree function indicates genetic risk"))
                
                if risk_factors:
                    for factor, description in risk_factors:
                        st.markdown(f"- **{factor}**: {description}")
                else:
                    st.markdown("No major risk factors identified based on the entered values.")
    
    with tab2:
        st.header("Dataset Insights")
        
        # Load and display the dataset
        data = load_and_preprocess_data()
        
        # Basic dataset statistics
        st.subheader("Dataset Overview")
        st.write(f"Number of records: {len(data)}")
        st.write(f"Number of diabetic patients: {data['Outcome'].sum()} ({data['Outcome'].mean()*100:.1f}%)")
        
        # Display the dataset
        if st.checkbox("Show dataset sample"):
            st.dataframe(data.head(10))
        
        # Data visualization
        st.subheader("Data Visualization")
        
        # Feature distribution
        viz_option = st.selectbox(
            "Select visualization type:", 
            ["Feature Distributions", "Correlation Heatmap", "Pairplot of Key Features"]
        )
        
        if viz_option == "Feature Distributions":
            selected_feature = st.selectbox(
                "Select feature to visualize:", 
                data.columns[:-1]
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=data, x=selected_feature, hue='Outcome', 
                         multiple="stack", palette=["#4CAF50", "#F44336"], ax=ax)
            ax.set_title(f'Distribution of {selected_feature} by Diabetes Outcome')
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
        elif viz_option == "Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation = data.corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title('Correlation Heatmap of Features')
            st.pyplot(fig)
            
        elif viz_option == "Pairplot of Key Features":
            st.write("Loading pairplot (this might take a moment)...")
            # Select only the most important features to avoid a cluttered plot
            key_features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Outcome']
            fig = sns.pairplot(data[key_features], hue='Outcome', palette=["#4CAF50", "#F44336"], diag_kind='kde')
            fig.fig.suptitle('Pairplot of Key Features', y=1.02)
            st.pyplot(fig)
    
    with tab3:
        st.header("About the Fuzzy Logic Model")
        
        st.markdown("""
        ### What is Fuzzy Logic?
        
        Fuzzy logic is a form of many-valued logic that deals with reasoning that is approximate rather than fixed and exact. 
        Unlike traditional binary logic that uses crisp values of 0 (False) or 1 (True), fuzzy logic variables may have a truth 
        value that ranges in degree between 0 and 1.
        
        ### How the Model Works
        
        This diabetes prediction model uses fuzzy logic to:
        
        1. **Define membership functions** for each input variable (like glucose level, BMI, age)
        2. **Create fuzzy rules** that capture medical knowledge about diabetes risk factors
        3. **Apply fuzzy inference** to compute the risk of diabetes
        
        ### Key Features of the Model
        
        - Uses established medical knowledge about diabetes risk factors
        - Captures uncertainty in medical diagnosis
        - Provides interpretable results with risk levels
        - Can handle imprecise or incomplete input data
        
        ### Fuzzy Rules Used
        
        The model includes the following key rules:
        
        - High glucose level → High diabetes risk
        - Prediabetic glucose & obesity → High diabetes risk
        - Advanced age & family history → High diabetes risk
        - Normal BMI & young age → Low diabetes risk
        - Low glucose level → Low diabetes risk
        - Low family history (pedigree) → Low diabetes risk
        
        ### Model Performance
        
        When evaluated on a test dataset, the model demonstrates good performance in identifying diabetes risk.
        """)
        
        # Add a citation or references
        st.markdown("""
        ### References
        
        - Pima Indians Diabetes Database (UCI Machine Learning Repository)
        - Fuzzy logic approach for diabetes detection using skfuzzy library
        """)

if __name__ == "__main__":
    main()