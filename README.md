This project uses data analytics and machine learning to predict IPL match outcomes. By analyzing extensive ball-by-ball data, match statistics, and in-game metrics, the model predicts the winning probability for each team in real-time. This project is built to be a robust, scalable, and easy-to-understand example of a sports analytics pipeline.

##Key Features:
Real-Time Prediction: The model can predict the winning chances of both teams after any ball in the second innings.
Feature-Rich Model: Utilizes key in-game metric for high accuracy, including:
Current Run Rate (CRR) & Required Run Rate (RRR)
Runs left, balls left, and wickets in hand
Target score set in the first innings
Clean Architecture: The code is split into logical modules for configuration, data processing, modeling, and prediction, making it easy to maintain and extend.
High Accuracy: Employs a RandomForestClassifier which achieves high accuracy on historical data.
##Project Structure
ipl-win-predictor/
|
├── .gitignore          # Files to be ignored by Git
├── README.md           # You are here!
├── requirements.txt    # Project dependencies
|
├── config.py           # Configuration and constants
├── main.py             # Main script to run the pipeline
├── model.py            # Model definition and training logic
├── predict.py          # Prediction function
├── process.py          # Data loading and feature engineering
|
└── data/
    ├── IPL_Ball_by_Ball_2008_2022.csv
    └── IPL_Matches_2008_2022.csv

##Install Dependencies
pip install -r requirements.txt

#To Run
python main.py
