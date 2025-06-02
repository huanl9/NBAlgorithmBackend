import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import torch
import torch.nn as nn
from datetime import datetime
import sqlite3
from pathlib import Path

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://huanl9.github.io"},
                     r"/teams": {"origins": "https://huanl9.github.io"}})

# Neural Network definition
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Database setup
DB_PATH = Path('nba_data.db')
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

def load_data_to_db():
    # Load only necessary columns
    games_cols = ['gameId', 'gameDate', 'hometeamId', 'awayteamId', 'hometeamName', 'awayteamName', 'winner']
    team_stats_cols = ['gameId', 'teamId', 'opponentTeamId', 'gameDate', 'homeOrAway', 'teamScore', 'opponentScore', 'fieldGoalsAttempted', 'reboundsOffensive', 'turnovers', 'freeThrowsAttempted', 'assists', 'reboundsTotal']
    player_stats_cols = ['gameId', 'playerteamName', 'points']
    
    games = pd.read_csv('./historical-nba-data-and-player-box-scores/versions/166/Games.csv', usecols=games_cols)
    team_stats = pd.read_csv('./historical-nba-data-and-player-box-scores/versions/166/TeamStatistics.csv', usecols=team_stats_cols)
    player_stats = pd.read_csv('./historical-nba-data-and-player-box-scores/versions/166/PlayerStatistics.csv', usecols=player_stats_cols, low_memory=False)
    
    games.to_sql('games', conn, if_exists='replace', index=False)
    team_stats.to_sql('team_stats', conn, if_exists='replace', index=False)
    player_stats.to_sql('player_stats', conn, if_exists='replace', index=False)
    
    # Precompute top3_avg_points
    top3_avg_points = player_stats.groupby(['gameId', 'playerteamName']).apply(
        lambda x: x.nlargest(3, 'points')['points'].mean() if len(x) >= 3 else x['points'].mean(),
        include_groups=False
    ).reset_index(name='top3_avg_points')
    top3_avg_points.to_sql('top3_avg_points', conn, if_exists='replace', index=False)

# Feature engineering functions using database queries
def calculate_features(team_id, date, home_away='HOME'):
    query = f"""
    SELECT ts.teamScore, ts.fieldGoalsMade, ts.fieldGoalsAttempted, ts.threePointsMade, ts.threePointsAttempted,
           ts.freeThrowsMade, ts.reboundsOffensive, ts.reboundsDefensive, ts.assists, tap.top3_avg_points
    FROM team_stats ts
    LEFT JOIN top3_avg_points tap ON ts.gameId = tap.gameId AND ts.teamId = tap.playerteamName
    WHERE ts.teamId = {team_id} AND ts.homeOrAway = '{home_away}' AND ts.gameDate < '{date}'
    ORDER BY ts.gameDate DESC
    LIMIT 5
    """
    df = pd.read_sql_query(query, conn)
    if df.empty:
        return np.zeros( siano = np.zeros(13))
    features = df.mean(numeric_only=True).to_list()
    return np.array(features)

def calculate_win_rate(team_id, date, home_away):
    query = f"""
    SELECT (SUM(CASE WHEN teamScore > opponentScore THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) as win_rate
    FROM team_stats
    WHERE teamId = {team_id} AND homeOrAway = '{home_away}' AND gameDate < '{date}'
    """
    result = pd.read_sql_query(query, conn)
    return result['win_rate'].iloc[0] if not result.empty else 0.5

def calculate_current_season_win_rate(team_id, date, season):
    query = f"""
    SELECT (SUM(CASE WHEN teamScore > opponentScore THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) as win_rate
    FROM team_stats
    WHERE teamId = {team_id} AND season = {season} AND gameDate < '{date}'
    """
    result = pd.read_sql_query(query, conn)
    return result['win_rate'].iloc[0] if not result.empty else 0.5

def get_season(date):
    year = date.year
    return year if date.month >= 10 else year - 1

# Lazy load models
def load_models():
    rf_model = joblib.load('random_forest_model.joblib')
    lr_model = joblib.load('logistic_regression_model.joblib')
    svm_model = joblib.load('svm_model.joblib')
    xgb_model = joblib.load('xgboost_model.joblib')
    voting_model = joblib.load('voting_classifier_model.joblib')
    scaler = joblib.load('scaler.joblib')
    nn_model = Net(input_size=36)
    nn_model.load_state_dict(torch.load('neural_network_model.pth'))
    nn_model.eval()
    return rf_model, lr_model, svm_model, xgb_model, voting_model, scaler, nn_model

# Endpoints
@app.route('/teams', methods=['GET'])
def get_teams():
    query = "SELECT DISTINCT hometeamName FROM games UNION SELECT DISTINCT awayteamName FROM games"
    teams = pd.read_sql_query(query, conn)['hometeamName'].tolist()
    return jsonify(teams)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    home_team = data['home_team']
    away_team = data['away_team']
    date_str = data['date']
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Get team IDs
    query_home = f"SELECT hometeamId FROM games WHERE hometeamName = '{home_team}' LIMIT 1"
    query_away = f"SELECT awayteamId FROM games WHERE awayteamName = '{away_team}' LIMIT 1"
    home_id = pd.read_sql_query(query_home, conn)['hometeamId'].iloc[0]
    away_id = pd.read_sql_query(query_away, conn)['awayteamId'].iloc[0]
    
    # Calculate features
    home_features = calculate_features(home_id, date, 'HOME')
    away_features = calculate_features(away_id, date, 'AWAY')
    
    season = get_season(date)
    home_win_rate = calculate_win_rate(home_id, date, 'HOME')
    away_win_rate = calculate_win_rate(away_id, date, 'AWAY')
    home_current_win_rate = calculate_current_season_win_rate(home_id, date, season)
    away_current_win_rate = calculate_current_season_win_rate(away_id, date, season)
    
    # Placeholder features (e.g., rest days, H2H win rate)
    features = np.concatenate([home_features, away_features, [home_win_rate, away_win_rate, home_current_win_rate, away_current_win_rate, 0, 0, 0.5, 0, 0, 0]])
    
    if np.any(np.isnan(features)):
        return jsonify({'error': 'Unable to compute features'}), 400
    
    # Load models
    rf_model, lr_model, svm_model, xgb_model, voting_model, scaler, nn_model = load_models()
    
    scaled_features = scaler.transform([features])
    
    probs = []
    for model in [rf_model, lr_model, svm_model, xgb_model, voting_model]:
        prob = model.predict_proba(scaled_features)[0, 1]
        probs.append(prob)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    prob_nn = nn_model(input_tensor).item()
    probs.append(prob_nn)
    
    average_prob = np.mean(probs)
    winner = home_team if average_prob > 0.5 else away_team
    return jsonify({'winner': winner, 'probability': float(average_prob)})

if __name__ == '__main__':
    load_data_to_db()
    app.run(host='0.0.0.0', port=5000)