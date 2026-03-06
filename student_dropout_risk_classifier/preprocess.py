import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler 
from sklearn.feature_selection import SelectKBest, chi2

def preprocess_data(path):
    df = pd.read_csv(path)

    # Encode categorical variables (example columns)
    for col in ['sex', 'address', 'schoolsup', 'famsup']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Define features and target (assuming target column is 'dropout')
    X = df.drop(columns=['dropout'])
    y = df['dropout']

    # Scale numeric features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection
    selector = SelectKBest(chi2, k=10)
    X_selected = selector.fit_transform(X_scaled, y)

    return X_selected, y, selector.get_support(indices=True)
