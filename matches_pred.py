import pandas as pd
import joblib

# Load assets
model = joblib.load('world_cup_model.pkl')
features = joblib.load('world_cup_features.pkl')

import pandas as pd

def predict_match(home_id, away_id, stage_name, group_stage, knockout_stage, year=2026):
      
    # Helper to safely get values
    def get_val(feature_name, team_id, default=0):
        return features[feature_name].get(team_id, default)

    # 1. Look up Confederation
    home_conf = features['confederation'].get(home_id) # Default to Europe if missing
    away_conf = features['confederation'].get(away_id)

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
    
    # 3. Predict
    X_new = pd.DataFrame([row])  # Debug: Check the constructed feature row
    
    # Optional: Filter columns to match training exactly if you have issues
    # X_new = X_new[model.feature_names_in_]
    
    prob = model.predict_proba(X_new)[0, 1]
    return prob

team_mapping = joblib.load("team_mapping.pkl")

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
        
        print(f"Error: Could not find team ID for '{country_name}'")
        return None

while True:
    home_team_name=input("Enter Home team Name: ")
    if home_team_name.lower()=='exit':
        break
    home_team_id_actual=get_team_id(home_team_name)

    away_team_name=input('Enter Away team Name: ')
    if away_team_name.lower()=='exit':
        break
    
    away_team_id_actual=get_team_id(away_team_name)

    stage=input("Enter game stage (e.g., 'group', 'round of 16', 'quarter-finals', 'semi-finals', 'final'): ")
    if stage.lower()=='exit':
        break
    
    group_stage=1 if stage=='group' else 0
    knockout_stage=1 if stage in ['round of 16', 'quarter-finals', 'semi-finals', 'final'] else 0
    probability=predict_match(home_team_id_actual, away_team_id_actual, stage, group_stage, knockout_stage)
    # Check against 0.5 (50%), not 50
    if probability >= 0.5:
    # Formats 0.65 as "65.00%" automatically
        print(f"{home_team_name} is more likely to win against {away_team_name} with {probability:.2%} probability.")
    else:
    # Calculate inverse (1 - 0.35 = 0.65) and format
        print(f"{away_team_name} is more likely to win against {home_team_name} with {1 - probability:.2%} probability.")
