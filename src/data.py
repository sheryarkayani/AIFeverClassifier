import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Define column mapping for standardization
# Common columns needed for the model
REQUIRED_COLUMNS = [
    'Age', 'Sex', 'temperature', 'wbc', 'platelets', 
    'headache', 'joint_pain', 'rash', 'travel_to_hot_area', 
    'mosquito_exposure', 'fever_type'
]

def simulate_symptoms(fever_type, n, lab_data=None):
    """
    Simulates symptoms for a given fever type.
    """
    # Base rates based on fever type (derived from domain knowledge/notebook logic)
    if fever_type == "dengue":
        headache = np.random.choice([0,1], n, p=[0.1,0.9])
        joint_pain = np.random.choice([0,1], n, p=[0.3,0.7])
        rash = np.random.choice([0,1], n, p=[0.2,0.8])
        travel_to_hot_area = np.random.choice([0,1], n, p=[0.8,0.2])
        mosquito_exposure = np.ones(n, dtype=int)
        # High fever typically
        temperature = np.random.uniform(38, 40, n) 
        
    elif fever_type == "malaria":
        headache = np.ones(n, dtype=int)
        joint_pain = np.random.choice([0,1], n, p=[0.5,0.5])
        rash = np.zeros(n, dtype=int)
        travel_to_hot_area = np.random.choice([0,1], n, p=[0.7,0.3])
        mosquito_exposure = np.ones(n, dtype=int)
        # High fever cycling
        temperature = np.random.uniform(38, 41, n)
        
    elif fever_type == "typhoid":
        headache = np.random.choice([0,1], n, p=[0.3,0.7])
        joint_pain = np.random.choice([0,1], n, p=[0.5,0.5])
        rash = np.zeros(n, dtype=int)
        travel_to_hot_area = np.zeros(n, dtype=int)
        mosquito_exposure = np.zeros(n, dtype=int)
        # Step-ladder fever
        temperature = np.random.uniform(38, 40, n)
        
    elif fever_type == "viral":
        headache = np.ones(n, dtype=int)
        joint_pain = np.random.choice([0,1], n) # 50/50
        rash = np.random.choice([0,1], n)
        travel_to_hot_area = np.zeros(n, dtype=int)
        mosquito_exposure = np.zeros(n, dtype=int)
        # Mild to moderate fever
        temperature = np.random.uniform(37.5, 39, n)
        
    elif fever_type == "heat-stroke":
        headache = np.ones(n, dtype=int)
        joint_pain = np.zeros(n, dtype=int)
        rash = np.zeros(n, dtype=int)
        travel_to_hot_area = np.ones(n, dtype=int)
        mosquito_exposure = np.zeros(n, dtype=int)
        # Very high temp
        temperature = np.random.uniform(40, 41, n) # Hyperpyrexia
    else:
        raise ValueError(f"Unknown fever type: {fever_type}")

    # Create dataframe
    df = pd.DataFrame({
        'fever_type': [fever_type] * n,
        'headache': headache,
        'joint_pain': joint_pain,
        'rash': rash,
        'travel_to_hot_area': travel_to_hot_area,
        'mosquito_exposure': mosquito_exposure,
        'temperature': temperature
    })

    # Integrate lab data if provided (e.g. for Dengue/Malaria real datasets)
    # We assume lab_data is a DataFrame with matching length n
    if lab_data is not None:
        # We try to keep existing columns from lab_data
        # Standardize Age/Sex/Lab values if they exist
        
        # --- DENGUE STANDARDIZATION ---
        if 'Haemoglobin' in lab_data.columns: # Specific to Dengue dataset likely
             df['haemoglobin'] = lab_data['Haemoglobin']
        if 'WBC' in lab_data.columns:
             df['wbc'] = lab_data['WBC']
        if 'platelets' in lab_data.columns:
             df['platelets'] = lab_data['platelets']
        if 'Age' in lab_data.columns:
             df['Age'] = lab_data['Age']
        if 'Sex' in lab_data.columns:
             df['Sex'] = lab_data['Sex']

        # --- MALARIA STANDARDIZATION ---
        if 'Hemoglobin(Hb%)' in lab_data.columns:
            df['haemoglobin'] = lab_data['Hemoglobin(Hb%)']
        if 'Neutrophils' in lab_data.columns: # Malaria specific cols?
             pass # We focus on common ones for now
        
    else:
        # Simulate lab data if not provided (for Typhoid, Viral, HeatStroke)
        # Reasonable defaults/randoms
        df['Age'] = np.random.randint(5, 70, n)
        df['Sex'] = np.random.choice(['Male', 'Female'], n)
        
        # Wbc/Platelets simulation based on disease
        if fever_type == 'typhoid':
             df['wbc'] = np.random.normal(6000, 2000, n) # Normal/Low
             df['platelets'] = np.random.normal(250000, 50000, n)
        elif fever_type == 'viral':
             df['wbc'] = np.random.normal(5000, 1500, n) # Can be low
             df['platelets'] = np.random.normal(200000, 50000, n)
        elif fever_type == 'heat-stroke':
             df['wbc'] = np.random.normal(9000, 3000, n) # Stress response high
             df['platelets'] = np.random.normal(250000, 50000, n)
        else:
             # Fallback
             df['wbc'] = np.random.normal(7000, 2000, n)
             df['platelets'] = np.random.normal(250000, 50000, n)

    return df

def load_and_simulate_data(base_path="."):
    """
    Loads real datasets and combines with simulated data.
    """
    
    # 1. Load Dengue Data
    dengue_path = os.path.join(base_path, "Dengue Dataset.csv")
    dengue_df = pd.read_csv(dengue_path)
    # Ensure n matches
    n_dengue = len(dengue_df)
    # Simulate symptoms for these real patients
    dengue_sim = simulate_symptoms("dengue", n_dengue, lab_data=dengue_df)
    
    # 2. Load Malaria Data
    malaria_path = os.path.join(base_path, "Malaria Diseases dataset - .csv")
    malaria_df = pd.read_csv(malaria_path)
    n_malaria = len(malaria_df)
    malaria_sim = simulate_symptoms("malaria", n_malaria, lab_data=malaria_df)
    
    # 3. Simulate Other Diseases (Typhoid, Viral, Heat-Stroke)
    # Let's generate ~500 samples each to balance a bit (or match dengue size)
    n_synth = 500
    typhoid_sim = simulate_symptoms("typhoid", n_synth)
    viral_sim = simulate_symptoms("viral", n_synth)
    heat_sim = simulate_symptoms("heat-stroke", n_synth)
    
    # 4. Combine
    combined_df = pd.concat([
        dengue_sim, malaria_sim, typhoid_sim, viral_sim, heat_sim
    ], axis=0, ignore_index=True)
    
    # 5. Final Standardization cleanup
    # Ensure Sex is 0/1 or standardized
    combined_df['Sex'] = combined_df['Sex'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0}).fillna(0) # Simple fallback
    
    # Select only the features we want to train on
    # (Excluding haemoglobin for now if not available in all synthetic ones widely)
    # Keeping standard set:
    feature_cols = [
        'Age', 'Sex', 'temperature', 'wbc', 'platelets',
        'headache', 'joint_pain', 'rash', 'travel_to_hot_area', 'mosquito_exposure'
    ]
    
    # Ensure all exist
    for col in feature_cols:
        if col not in combined_df.columns:
            # Should not happen with our sim logic, but just in case fill 0
            combined_df[col] = 0

    return combined_df[feature_cols], combined_df['fever_type']

def get_pipeline():
    """
    Returns the Scikit-Learn preprocessing + model pipeline.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    numeric_features = ['Age', 'temperature', 'wbc', 'platelets']
    categorical_features = ['Sex', 'headache', 'joint_pain', 'rash', 'travel_to_hot_area', 'mosquito_exposure']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(
                              n_estimators=300, 
                              max_depth=20,
                              class_weight='balanced',
                              random_state=42,
                              n_jobs=-1
                          ))])
    
    return clf
