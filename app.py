import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import torch
import torch.nn as nn
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://your-username.github.io"},
                     r"/teams": {"origins": "https://your-username.github.io"}})

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

# Load and preprocess data
def load_and_preprocess_data():
    games = pd.read_csv('./historical-nba-data-and-player-box-scores/versions/166/Games.csv')
    team_stats = pd.read_csv('./historical-nba-data-and-player-box-scores/versions/166/TeamStatistics.csv')
    player_stats = pd.read_csv('./historical-nba-data-and-player-box-scores/versions/166/PlayerStatistics.csv', low_memory=False)
    
    games['gameDate'] = pd.to_datetime(games['gameDate'])
    team_stats['gameDate'] = pd.to_datetime(team_stats['gameDate'])
    player_stats['gameDate'] = pd.to_datetime(player_stats['gameDate'])
    
    def get_season(date):
        year = date.year
        return year if date.month >= 10 else year - 1
    games['season'] = games['gameDate'].apply(get_season)
    team_stats['season'] = team_stats['gameDate'].apply(get_season)
    
    games = games[games['gameDate'].dt.year >= 2010]
    team_stats = team_stats[team_stats['gameDate'].dt.year >= 2010]
    player_stats = player_stats[player_stats['gameId'].isin(games['gameId'])]
    
    team_stats = pd.merge(team_stats, games[['gameId', 'hometeamId', 'awayteamId']], on='gameId', how='left')
    team_stats['homeOrAway'] = np.where(team_stats['teamId'] == team_stats['hometeamId'], 'HOME', 'AWAY')
    
    player_stats = pd.merge(player_stats, games[['gameId', 'hometeamName', 'awayteamName', 'hometeamId', 'awayteamId']], on='gameId', how='left')
    player_stats['teamId'] = np.where(player_stats['playerteamName'] == player_stats['hometeamName'], player_stats['hometeamId'], player_stats['awayteamId'])
    player_stats = player_stats.dropna(subset=['teamId'])
    
    top3_avg_points_df = player_stats.groupby(['gameId', 'teamId']).apply(
        lambda x: x.nlargest(3, 'points')['points'].mean() if len(x) >= 3 else x['points'].mean(),
        include_groups=False
    ).reset_index(name='top3_avg_points')
    
    team_stats = calculate_advanced_stats(team_stats)
    return games, team_stats, player_stats, top3_avg_points_df

def calculate_advanced_stats(team_stats_df):
    def calc_poss(row):
        return row['fieldGoalsAttempted'] - row['reboundsOffensive'] + row['turnovers'] + 0.4 * row['freeThrowsAttempted']
    team_stats_df['Poss_team'] = team_stats_df.apply(calc_poss, axis=1)
    game_poss = team_stats_df.groupby('gameId')['Poss_team'].mean().reset_index()
    game_poss.columns = ['gameId', 'Poss']
    team_stats_df = pd.merge(team_stats_df, game_poss, on='gameId')
    team_stats_df['ORtg'] = (team_stats_df['teamScore'] / team_stats_df['Poss']) * 100
    team_stats_df['DRtg'] = (team_stats_df['opponentScore'] / team_stats_df['Poss']) * 100
    team_stats_df['Pace'] = team_stats_df['Poss']
    return team_stats_df

# Feature engineering functions
def calculate_features(team_id, date, team_stats_df, top3_df, N=5, home_away='HOME'):
    team_games = team_stats_df[(team_stats_df['teamId'] == team_id) & (team_stats_df['homeOrAway'] == home_away)].sort_values('gameDate', ascending=False).head(N)
    if team_games.empty:
        return np.zeros(13)
    top3_points = top3_df[(top3_df['teamId'] == team_id) & (top3_df['gameId'].isin(team_games['gameId']))]['top3_avg_points'].mean()
    features = [
        team_games['teamScore'].mean(),
        team_games['fieldGoalsMade'].mean(),
        team_games['fieldGoalsAttempted'].mean(),
        team_games['threePointsMade'].mean(),
        team_games['threePointsAttempted'].mean(),
        team_games['freeThrowsMade'].mean(),
        team_games['ORtg'].mean(),
        team_games['DRtg'].mean(),
        team_games['Pace'].mean(),
        team_games['reboundsOffensive'].mean(),
        team_games['reboundsDefensive'].mean(),
        team_games['assists'].mean(),
        top3_points if not np.isnan(top3_points) else 0
    ]
    return np.array(features)

def calculate_win_rate(team_id, date, team_stats_df, home_away):
    team_games = team_stats_df[(team_stats_df['teamId'] == team_id) & (team_stats_df['homeOrAway'] == home_away)]
    wins = team_games['teamScore'] > team_games['opponentScore']
    return wins.mean() if not team_games.empty else 0.5

def calculate_current_season_win_rate(team_id, date, season, team_stats_df):
    team_games = team_stats_df[(team_stats_df['teamId'] == team_id) & (team_stats_df['season'] == season)]
    wins = team_games['teamScore'] > team_games['opponentScore']
    return wins.mean() if not team_games.empty else 0.5

def get_season(date):
    year = date.year
    return year if date.month >= 10 else year - 1

# Load data and models
games, team_stats, player_stats, top3_avg_points_df = load_and_preprocess_data()
team_name_to_id = pd.concat([games[['hometeamId', 'hometeamName']].rename(columns={'hometeamId': 'teamId', 'hometeamName': 'teamName'}),
                             games[['awayteamId', 'awayteamName']].rename(columns={'awayteamId': 'teamId', 'awayteamName': 'teamName'})]).drop_duplicates().set_index('teamName')['teamId'].to_dict()
team_names = sorted(set(games['hometeamName'].unique()) | set(games['awayteamName'].unique()))

rf_model = joblib.load('random_forest_model.joblib')
lr_model = joblib.load('logistic_regression_model.joblib')
svm_model = joblib.load('svm_model.joblib')
xgb_model = joblib.load('xgboost_model.joblib')
voting_model = joblib.load('voting_classifier_model.joblib')
scaler = joblib.load('scaler.joblib')
nn_model = Net(input_size=36)
nn_model.load_state_dict(torch.load('neural_network_model.pth'))
nn_model.eval()

# Endpoints
@app.route('/teams', methods=['GET'])
def get_teams():
    return jsonify(team_names)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    home_team = data['home_team']
    away_team = data['away_team']
    date_str = data['date']
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    home_id = team_name_to_id[home_team]
    away_id = team_name_to_id[away_team]
    
    past_games = games[games['gameDate'] < date]
    past_team_stats = team_stats[team_stats['gameDate'] < date]
    past_player_stats = player_stats[player_stats['gameDate'] < date]
    past_top3_avg_points_df = top3_avg_points_df[top3_avg_points_df['gameId'].isin(past_games['gameId'])]
    
    home_features = calculate_features(home_id, date, past_team_stats, past_top3_avg_points_df, N=5, home_away='HOME')
    away_features = calculate_features(away_id, date, past_team_stats, past_top3_avg_points_df, N=5, home_away='AWAY')
    
    season = get_season(date)
    home_win_rate = calculate_win_rate(home_id, date, past_team_stats, 'HOME')
    away_win_rate = calculate_win_rate(away_id, date, past_team_stats, 'AWAY')
    home_current_win_rate = calculate_current_season_win_rate(home_id, date, season, past_team_stats)
    away_current_win_rate = calculate_current_season_win_rate(away_id, date, season, past_team_stats)
    last_game_home = past_team_stats[past_team_stats['teamId'] == home_id].sort_values('gameDate', ascending=False).head(1)
    rest_days_home = (date - last_game_home['gameDate'].iloc[0]).days if not last_game_home.empty else 10
    last_game_away = past_team_stats[past_team_stats['teamId'] == away_id].sort_values('gameDate', ascending=False).head(1)
    rest_days_away = (date - last_game_away['gameDate'].iloc[0]).days if not last_game_away.empty else 10
    past_h2h_games = past_games[((past_games['hometeamId'] == home_id) & (past_games['awayteamId'] == away_id)) | 
                                ((past_games['hometeamId'] == away_id) & (past_games['awayteamId'] == home_id))]
    h2h_win_rate = (past_h2h_games['winner'] == home_id).mean() if not past_h2h_games.empty else 0.5
    is_playoff = 0
    ortg_diff = home_features[6] - away_features[7]
    drtg_diff = home_features[7] - away_features[6]
    
    features = np.concatenate([home_features, away_features, [home_win_rate, away_win_rate, home_current_win_rate, away_current_win_rate, rest_days_home, rest_days_away, h2h_win_rate, is_playoff, ortg_diff, drtg_diff]])
    
    if np.any(np.isnan(features)):
        return jsonify({'error': 'Unable to compute features'}), 400
    
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
    app.run(host='0.0.0.0', port=5000)