# model.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    """Defines and trains the Random Forest model pipeline."""

    # Create a preprocessing pipeline for categorical features
    preprocessor = ColumnTransformer([
        ('trf', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['BattingTeam', 'BowlingTeam', 'City'])
    ], remainder='passthrough')

    # Create the full model pipeline
    pipe = Pipeline([
        ('step1', preprocessor),
        ('step2', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the model
    pipe.fit(X_train, y_train)
    
    return pipe