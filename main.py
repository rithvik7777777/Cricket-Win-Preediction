# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import from our custom modules
import config
from process import load_and_process_data
from model import train_model
from predict import predict_winner

if __name__ == "__main__":
    # 1. Load and process data
    print("Loading and processing data...")
    final_df = load_and_process_data(config.MATCHES_FILE, config.BALL_BY_BALL_FILE)

    # 2. Split data for training and testing
    X = final_df[config.FEATURES]
    y = final_df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train the model
    print("Training model...")
    pipeline = train_model(X_train, y_train)

    # 4. Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model Accuracy on Test Set: {accuracy * 100:.2f}%\n")

    # 5. Make a sample prediction
    print("Making a sample prediction...")
    predict_winner(
        pipe=pipeline,
        batting_team='Chennai Super Kings',
        bowling_team='Mumbai Indians',
        city='Mumbai',
        runs_left=50,
        balls_left=20,
        wickets_left=4,
        target=180
    )