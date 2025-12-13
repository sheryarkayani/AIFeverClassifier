import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Common columns needed for the model
REQUIRED_COLUMNS = [
    'Age', 'Sex', 'temperature', 'wbc', 'platelets', 
    'headache', 'joint_pain', 'rash', 'vomiting', 'fatigue', 'chills',
    'fever_pattern', 'travel_to_hot_area', 'mosquito_exposure', 
    'sun_exposure', 'hygiene_issue', 'fever_type'
]

# Fever Patterns encoding: 0=Constant, 1=Intermittent, 2=Spikes
PATTERN_CONSTANT = 0
PATTERN_INTERMITTENT = 1
PATTERN_SPIKES = 2

def simulate_symptoms(fever_type, n, lab_data=None):
    """
    Simulates symptoms for a given fever type including new detailed inputs.
    """
    # Defaults
    headache = np.zeros(n, dtype=int)
    joint_pain = np.zeros(n, dtype=int)
    rash = np.zeros(n, dtype=int)
    vomiting = np.zeros(n, dtype=int)
    fatigue = np.zeros(n, dtype=int)
    chills = np.zeros(n, dtype=int)
    
    # 0, 1, 2
    fever_pattern = np.zeros(n, dtype=int)
    
    travel_to_hot = np.zeros(n, dtype=int)
    mosquito = np.zeros(n, dtype=int)
    sun_exposure = np.zeros(n, dtype=int)
    hygiene_issue = np.zeros(n, dtype=int)
    
    temperature = np.random.uniform(37.5, 38.5, n)

    if fever_type == "dengue":
        headache = np.random.choice([0,1], n, p=[0.1,0.9])
        joint_pain = np.random.choice([0,1], n, p=[0.1,0.9]) # Breakbone fever
        rash = np.random.choice([0,1], n, p=[0.2,0.8])
        vomiting = np.random.choice([0,1], n, p=[0.3,0.7])
        fatigue = np.random.choice([0,1], n, p=[0.1,0.9])
        chills = np.random.choice([0,1], n, p=[0.4,0.6])
        
        fever_pattern = np.random.choice([PATTERN_CONSTANT, PATTERN_SPIKES], n)
        
        travel_to_hot = np.random.choice([0,1], n, p=[0.8,0.2]) # Endemic
        mosquito = np.random.choice([0,1], n, p=[0.1,0.9])
        
        temperature = np.random.uniform(39, 40.5, n) # High fever
        
    elif fever_type == "malaria":
        headache = np.random.choice([0,1], n, p=[0.1,0.9])
        joint_pain = np.random.choice([0,1], n, p=[0.4,0.6])
        vomiting = np.random.choice([0,1], n, p=[0.4,0.6])
        fatigue = np.random.choice([0,1], n, p=[0.1,0.9])
        chills = np.ones(n, dtype=int) # Classic symptom
        
        fever_pattern = np.full(n, PATTERN_INTERMITTENT) # Cyclical
        
        travel_to_hot = np.random.choice([0,1], n, p=[0.7,0.3])
        mosquito = np.random.choice([0,1], n, p=[0.1,0.9])
        
        temperature = np.random.uniform(38.5, 41, n)
        
    elif fever_type == "typhoid":
        headache = np.random.choice([0,1], n, p=[0.2,0.8])
        joint_pain = np.random.choice([0,1], n, p=[0.3,0.7])
        vomiting = np.random.choice([0,1], n, p=[0.6,0.4])
        fatigue = np.ones(n, dtype=int)
        
        fever_pattern = np.full(n, PATTERN_CONSTANT) # Step-ladder, sustained
        
        hygiene_issue = np.random.choice([0,1], n, p=[0.1,0.9]) # Key risk
        
        temperature = np.random.uniform(39, 40.5, n)
        
    elif fever_type == "viral":
        headache = np.random.choice([0,1], n, p=[0.3,0.7])
        joint_pain = np.random.choice([0,1], n, p=[0.3,0.7]) # Body aches
        fatigue = np.random.choice([0,1], n, p=[0.2,0.8])
        chills = np.random.choice([0,1], n, p=[0.5,0.5])
        
        fever_pattern = np.random.choice([PATTERN_CONSTANT, PATTERN_SPIKES], n)
        
        temperature = np.random.uniform(37.8, 39.5, n)
        
    elif fever_type == "heat-stroke":
        headache = np.random.choice([0,1], n, p=[0.1,0.9])
        vomiting = np.random.choice([0,1], n, p=[0.3,0.7])
        fatigue = np.ones(n, dtype=int) # Exhaustion
        
        fever_pattern = np.full(n, PATTERN_CONSTANT)
        
        sun_exposure = np.ones(n, dtype=int) # Key risk
        
        temperature = np.random.uniform(40, 42, n) # Hyperpyrexia
    else:
        raise ValueError(f"Unknown fever type: {fever_type}")

    # Create dataframe
    df = pd.DataFrame({
        'fever_type': [fever_type] * n,
        'headache': headache,
        'joint_pain': joint_pain,
        'rash': rash,
        'vomiting': vomiting,
        'fatigue': fatigue,
        'chills': chills,
        'fever_pattern': fever_pattern,
        'travel_to_hot_area': travel_to_hot,
        'mosquito_exposure': mosquito,
        'sun_exposure': sun_exposure,
        'hygiene_issue': hygiene_issue,
        'temperature': temperature
    })

    # Integrate or Simulate Lab Data
    if lab_data is not None:
        if 'Haemoglobin' in lab_data.columns:
             df['haemoglobin'] = lab_data['Haemoglobin']
        if 'WBC' in lab_data.columns:
             df['wbc'] = lab_data['WBC']
        if 'platelets' in lab_data.columns:
             df['platelets'] = lab_data['platelets']
        if 'Age' in lab_data.columns:
             df['Age'] = lab_data['Age']
        if 'Sex' in lab_data.columns:
             df['Sex'] = lab_data['Sex']
        
        # Malaria specific
        if 'Hemoglobin(Hb%)' in lab_data.columns:
            df['haemoglobin'] = lab_data['Hemoglobin(Hb%)']
    else:
        # Defaults
        df['Age'] = np.random.randint(18, 60, n)
        df['Sex'] = np.random.choice(['Male', 'Female'], n)
        
        # Characteristic labs
        if fever_type == 'dengue':
             df['wbc'] = np.random.normal(3000, 1000, n) # Leukopenia
             df['platelets'] = np.random.normal(80000, 30000, n) # Thrombocytopenia
        elif fever_type == 'typhoid':
             df['wbc'] = np.random.normal(5500, 1500, n) 
             df['platelets'] = np.random.normal(200000, 50000, n)
        elif fever_type == 'viral':
             df['wbc'] = np.random.normal(6000, 2000, n)
             df['platelets'] = np.random.normal(180000, 40000, n)
        elif fever_type == 'heat-stroke':
             df['wbc'] = np.random.normal(10000, 3000, n) # Stress response
             df['platelets'] = np.random.normal(250000, 50000, n)
        else:
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
    dengue_sim = simulate_symptoms("dengue", len(dengue_df), lab_data=dengue_df)
    
    # 2. Load Malaria Data
    malaria_path = os.path.join(base_path, "Malaria Diseases dataset - .csv")
    malaria_df = pd.read_csv(malaria_path)
    malaria_sim = simulate_symptoms("malaria", len(malaria_df), lab_data=malaria_df)
    
    # 3. Simulate Other Diseases
    n_synth = 500
    typhoid_sim = simulate_symptoms("typhoid", n_synth)
    viral_sim = simulate_symptoms("viral", n_synth)
    heat_sim = simulate_symptoms("heat-stroke", n_synth)
    
    # 4. Combine
    combined_df = pd.concat([
        dengue_sim, malaria_sim, typhoid_sim, viral_sim, heat_sim
    ], axis=0, ignore_index=True)
    
    # 5. Final Standardization
    combined_df['Sex'] = combined_df['Sex'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0}).fillna(0)
    
    # Feature columns for training
    feature_cols = [
        'Age', 'Sex', 'temperature', 'wbc', 'platelets',
        'headache', 'joint_pain', 'rash', 'vomiting', 'fatigue', 'chills',
        'fever_pattern', 'travel_to_hot_area', 'mosquito_exposure', 
        'sun_exposure', 'hygiene_issue'
    ]
    
    # Ensure all exist and fill NaNs
    for col in feature_cols:
        if col not in combined_df.columns:
            combined_df[col] = 0
    
    # Handle missing lab values if any (though simulation covers it)
    combined_df.fillna(0, inplace=True)

    return combined_df[feature_cols], combined_df['fever_type']

def get_pipeline():
    """
    Returns the Scikit-Learn preprocessing + model pipeline with rich features.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    numeric_features = ['Age', 'temperature', 'wbc', 'platelets']
    categorical_features = [
        'Sex', 'headache', 'joint_pain', 'rash', 'vomiting', 
        'fatigue', 'chills', 'fever_pattern', 'travel_to_hot_area', 
        'mosquito_exposure', 'sun_exposure', 'hygiene_issue'
    ]

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
