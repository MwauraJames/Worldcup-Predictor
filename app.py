import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('world_cup_model.pkl')
features = joblib.load('world_cup_features.pkl')
team_mapping = joblib.load("team_mapping.pkl")

st.title("World Cup Prediction App")

st.write("Enter Home team,Away team and stage values below:")

teams = ['Brazil','Argentina',
'Germany',
'Italy',
'England',
'France',
'Sweden',
'United States',
'West Germany',
'Spain',
'Netherlands',
'Japan',
'Uruguay',
'Norway',
'Belgium',
'China',
'Mexico',
'South Korea',
'Portugal',
'Nigeria',
'Soviet Union',
'Australia',
'Hungary',
'Poland',
'Yugoslavia',
'Chile',
'Switzerland',
'Cameroon',
'Austria',
'Denmark',
'Canada',
'Paraguay',
'Czechoslovakia',
'Scotland',
'Colombia',
'Croatia',
'Romania',
'North Korea',
'Costa Rica',
'Ghana',
'Morocco',
'Saudi Arabia',
'South Africa',
'Peru',
'Bulgaria',
'Tunisia',
'Algeria',
'Republic of Ireland',
'Russia',
'Ecuador',
'New Zealand',
'Honduras',
'Iran',
'Ivory Coast',
'Northern Ireland',
'Greece',
'Senegal',
'Serbia',
'Wales',
'East Germany',
'Chinese Taipei',
'Jamaica',
'Slovenia',
'Cuba',
'Turkey',
'Zaire',
'Iraq',
'Czech Republic',
'Togo',
'Slovakia',
'Thailand',
'Qatar',
'Haiti',
'United Arab Emirates',
'Bolivia',
'Trinidad and Tobago',
'Serbia and Montenegro',
'Angola',
'Ukraine',
'Equatorial Guinea',
'Bosnia and Herzegovina',
'Egypt',
'Iceland',
'Panama',
] # Add your full list here
tournament_stages = ['group', 'round of 16', 'quarter-finals', 'semi-finals', 'final']

# 2. Create the dropdowns
home = st.selectbox(
    "Home Team", 
    teams, 
    index=None, 
    placeholder="Select home team..."
)

away = st.selectbox(
    "Away Team", 
    teams, 
    index=None, 
    placeholder="Select away team..."
)

stage = st.selectbox(
    "Stage", 
    tournament_stages, 
    index=None, 
    placeholder="Choose tournament stage..."
)

if st.button("Predict"):
    def predict_match(home_id, away_id, stage_name, group_stage, knockout_stage, year=2026):
    
    # Helper to safely get values
        def get_val(feature_name, team_id, default=0):
            return features[feature_name].get(team_id, default)

    # 1. Look up Confederation
        home_conf = features['confederation'].get(home_id, 0)  # Default to Europe if missing
        away_conf = features['confederation'].get(away_id, 0)

    # 2. Construct the Feature Row
        row = {
        # --- Metadata ---
            'group_stage': group_stage,
            'knockout_stage': knockout_stage,
            'year': year,
            'tournament_id': f'WC-{year}',
            'stage_name': stage_name,
            'home_confederation_id': home_conf,
            'away_confederation_id': away_conf,

        # --- The Specific Features You Added ---
            'home_win_rate': get_val('win_rate', home_id, 0.5),
            'away_win_rate': get_val('win_rate', away_id, 0.5),
        
            'home_avg_goal_diff': get_val('avg_goal_diff', home_id, 0),
            'away_avg_goal_diff': get_val('avg_goal_diff', away_id, 0),
        
            'home_elo': get_val('elo', home_id, 1500),
            'away_elo': get_val('elo', away_id, 1500),

        # --- Other Standard Features ---
            'home_past_knockouts': get_val('knockouts', home_id),
            'away_past_knockouts': get_val('knockouts', away_id),
        
        # Calculate Diffs (if your model still uses them)
            'win_rate_diff': get_val('win_rate', home_id) - get_val('win_rate', away_id),
            'elo_diff': get_val('elo', home_id) - get_val('elo', away_id),
            'experience_diff': get_val('experience', home_id) - get_val('experience', away_id),
            'goal_diff_diff': get_val('avg_goal_diff', home_id) - get_val('avg_goal_diff', away_id),
            'wc_experience_diff': get_val('wc_experience', home_id) - get_val('wc_experience', away_id),

        # --- History ---
            'home_past_titles': get_val('past_titles', home_id),
            'home_past_finals': get_val('past_finals', home_id),
            'home_past_semis': get_val('past_semis', home_id),
            'home_years_since_top4': get_val('years_since_top4', home_id, 100),
            'away_past_titles': get_val('past_titles', away_id),
            'away_past_finals': get_val('past_finals', away_id),
            'away_past_semis': get_val('past_semis', away_id),
            'away_years_since_top4': get_val('years_since_top4', away_id, 100)
        }
        X_new = pd.DataFrame([row])
        prob = model.predict_proba(X_new)[0, 1]
        return prob

    def get_team_id(country_name):
    # Try exact match
        code = team_mapping.get(country_name)
    
        if code:
            return code
        else:
        # Optional: Fuzzy search or case-insensitive search
        # This handles cases like "usa" vs "United States"
            for name, id_code in team_mapping.items():
                if country_name.lower() in name.lower():
                #print(f"Exact match not found. Did you mean '{name}'? Using ID: {id_code}")
                    return id_code
            
            return None

    

    home_team_id_actual=get_team_id(home)
    
    away_team_id_actual=get_team_id(away)
    
    group_stage=1 if stage=='group' else 0
    knockout_stage=1 if stage in ['round of 16', 'quarter-finals', 'semi-finals', 'final'] else 0
    probability=predict_match(home_team_id_actual, away_team_id_actual, stage, group_stage, knockout_stage)
    # Check against 0.5 (50%), not 50
    if probability >= 0.5:
    # Formats 0.65 as "65.00%" automatically
        outcome = f"{home} is more likely to win against {away} with {probability:.2%} probability."
        st.success(outcome)
    else:
    # Calculate inverse (1 - 0.35 = 0.65) and format
        outcome = f"{away} is more likely to win against {home} with {1 - probability:.2%} probability."
        st.success(outcome)