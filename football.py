# source: https://www.kaggle.com/airback/match-outcome-prediction-in-football
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/libs/loaders.py
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/03_football.ipynb
# source: https://github.com/Azure/fast_retraining/blob/master/experiments/03_football_GPU.ipynb

from __future__ import print_function

import os
import subprocess

import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
import time

from conversion import convert_cols_categorical_to_numeric
from metrics import classification_metrics_multilabel
from utils import *


def get_fifa_stats(match, player_stats):
    """ Aggregates fifa stats for a given match. """
    #Define variables
    match_id =  match.match_api_id
    date = match["date"]
    players = ["home_player_1", "home_player_2", "home_player_3", "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    player_stats_new = pd.DataFrame()
    names = []    
    #Loop through all players
    for player in players:
        #Get player ID
        player_id = match[player]        
        #Get player stats 
        stats = player_stats[player_stats.player_api_id == player_id]            
        #Identify current stats       
        current_stats = stats[stats.date < date].sort_values(by = "date", ascending = False)[:1]        
        if np.isnan(player_id) == True:
            overall_rating = pd.Series(0)
        else:
            current_stats.reset_index(inplace = True, drop = True)
            overall_rating = pd.Series(current_stats.loc[0, "overall_rating"])
        #Rename stat
        name = "{}_overall_rating".format(player)
        names.append(name)            
        #Aggregate stats
        player_stats_new = pd.concat([player_stats_new, overall_rating], axis = 1)    
    player_stats_new.columns = names        
    player_stats_new["match_api_id"] = match_id
    player_stats_new.reset_index(inplace = True, drop = True)
    return player_stats_new.iloc[0]

def get_fifa_data(matches, player_stats):
    """ Gets fifa data for all matches. """  
    #Apply get_fifa_stats for each match
    fifa_data = matches.apply(lambda x: get_fifa_stats(x, player_stats), axis = 1)
    return fifa_data

def get_match_label(match):
    """ Derives a label for a given match. """    
    #Define variables
    home_goals = match["home_team_goal"]
    away_goals = match["away_team_goal"]     
    label = pd.DataFrame()
    label.loc[0,"match_api_id"] = match["match_api_id"] 
    #Identify match label  
    if home_goals > away_goals:
        label.loc[0,"label"] = "Win"
    if home_goals == away_goals:
        label.loc[0,"label"] = "Draw"
    if home_goals < away_goals:
        label.loc[0,"label"] = "Defeat"
    return label.loc[0]

def get_overall_fifa_rankings(fifa, get_overall = False):
    """ Get overall fifa rankings from fifa data. """
    temp_data = fifa
    #Check if only overall player stats are desired
    if get_overall == True:
        #Get overall stats
        data = temp_data.loc[:,(fifa.columns.str.contains("overall_rating"))]
        data.loc[:,"match_api_id"] = temp_data.loc[:,"match_api_id"]
    else:
        #Get all stats except for stat date
        cols = fifa.loc[:,(fifa.columns.str.contains("date_stat"))]
        temp_data = fifa.drop(cols.columns, axis = 1)
        data = temp_data
    return data

def get_last_matches(matches, date, team, x = 10):
    """ Get the last x matches of a given team. """
    #Filter team matches from matches
    team_matches = matches[(matches["home_team_api_id"] == team) | (matches["away_team_api_id"] == team)]                   
    #Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by = "date", ascending = False).iloc[0:x,:]
    return last_matches
    
def get_last_matches_against_eachother(matches, date, home_team, away_team, x = 10):
    """ Get the last x matches of two given teams. """    
    #Find matches of both teams
    home_matches = matches[(matches["home_team_api_id"] == home_team) & (matches["away_team_api_id"] == away_team)]    
    away_matches = matches[(matches["home_team_api_id"] == away_team) & (matches["away_team_api_id"] == home_team)]  
    total_matches = pd.concat([home_matches, away_matches])    
    #Get last x matches
    try:    
        last_matches = total_matches[total_matches.date < date].sort_values(by = "date", ascending = False).iloc[0:x,:]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by = "date", ascending = False).iloc[0:total_matches.shape[0],:]        
        #Check for error in data
        if(last_matches.shape[0] > x):
            print("Error in obtaining matches")
    return last_matches
    
def get_goals(matches, team):
    """ Get the goals of a specfic team from a set of matches. """    
    #Find home and away goals
    # NOTE: somehow there are cases showing up with these arrays to be empty!
    # hence a separate step to check its length before proceeding
    hg = matches.home_team_goal[matches.home_team_api_id == team]
    ag = matches.away_team_goal[matches.away_team_api_id == team]
    if len(hg) == 0:
        home_goals = 0
    else:
        home_goals = int(hg.sum())
    if len(ag) == 0:
        away_goals = 0
    else:
        away_goals = int(ag.sum())
    total_goals = home_goals + away_goals
    return total_goals

def get_goals_conceided(matches, team):
    """ Get the goals conceided of a specfic team from a set of matches. """
    #Find home and away goals
    # NOTE: somehow there are cases showing up with these arrays to be empty!
    # hence a separate step to check its length before proceeding
    hg = matches.away_team_goal[matches.home_team_api_id == team]
    ag = matches.home_team_goal[matches.away_team_api_id == team]
    if len(ag) == 0:
        home_goals = 0
    else:
        home_goals = int(ag.sum())
    if len(hg) == 0:
        away_goals = 0
    else:
        away_goals = int(hg.sum())
    total_goals = home_goals + away_goals
    return total_goals

def get_wins(matches, team):
    """ Get the number of wins of a specfic team from a set of matches. """    
    #Find home and away wins
    home_wins = int(matches.home_team_goal[(matches.home_team_api_id == team) & (matches.home_team_goal > matches.away_team_goal)].count())
    away_wins = int(matches.away_team_goal[(matches.away_team_api_id == team) & (matches.away_team_goal > matches.home_team_goal)].count())
    total_wins = home_wins + away_wins
    return total_wins      
    
def get_match_features(match, matches, x = 10):
    """ Create match specific features for a given match. """    
    #Define variables
    date = match.date
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id    
    #Get last x matches of home and away team
    matches_home_team = get_last_matches(matches, date, home_team, x = 10)
    matches_away_team = get_last_matches(matches, date, away_team, x = 10)    
    #Get last x matches of both teams against each other
    last_matches_against = get_last_matches_against_eachother(matches, date, home_team, away_team, x = 3)    
    #Create goal variables
    home_goals = get_goals(matches_home_team, home_team)
    away_goals = get_goals(matches_away_team, away_team)
    home_goals_conceided = get_goals_conceided(matches_home_team, home_team)
    away_goals_conceided = get_goals_conceided(matches_away_team, away_team)    
    #Define result data frame
    result = pd.DataFrame()    
    #Define ID features
    result.loc[0, "match_api_id"] = match.match_api_id
    result.loc[0, "league_id"] = match.league_id
    #Create match features
    result.loc[0, "home_team_goals_difference"] = home_goals - home_goals_conceided
    result.loc[0, "away_team_goals_difference"] = away_goals - away_goals_conceided
    result.loc[0, "games_won_home_team"] = get_wins(matches_home_team, home_team) 
    result.loc[0, "games_won_away_team"] = get_wins(matches_away_team, away_team)
    result.loc[0, "games_against_won"] = get_wins(last_matches_against, home_team)
    result.loc[0, "games_against_lost"] = get_wins(last_matches_against, away_team)    
    #Add season
    result.loc[0, "season"] = int(match["season"].split("/")[0])    
    #Return match features
    return result.loc[0]
    
def create_feables(matches, fifa, bookkeepers, get_overall = False, horizontal = True, x = 10, all_leagues = True, verbose = True):
    """ Create and aggregate features and labels for all matches. """
    #Get fifa stats features
    fifa_stats = get_overall_fifa_rankings(fifa, get_overall)
    if verbose == True:
        print("Generating match features...")    
    #Get match features for all matches
    match_stats = matches.apply(lambda x: get_match_features(x, matches, x = 10), axis = 1)
    #Create dummies for league ID feature
    if all_leagues:
        dummies = pd.get_dummies(match_stats["league_id"]).rename(columns = lambda x: "League_" + str(x))
        match_stats = pd.concat([match_stats, dummies], axis = 1)
        match_stats.drop(["league_id"], inplace = True, axis = 1)
    if verbose == True:    
        print("Generating match labels...")    
    #Create match labels
    labels = matches.apply(get_match_label, axis = 1)    
    if verbose == True:    
        print("Generating bookkeeper data...")    
    #Get bookkeeper quotas for all matches
    bk_data = get_bookkeeper_data(matches, bookkeepers, horizontal = True)
    bk_data.loc[:,"match_api_id"] = matches.loc[:,"match_api_id"]
    #Merges features and labels into one frame
    features = pd.merge(match_stats, fifa_stats, on = "match_api_id", how = "left")
    features = pd.merge(features, bk_data, on = "match_api_id", how = "left")
    feables = pd.merge(features, labels, on = "match_api_id", how = "left")    
    #Drop NA values
    feables.dropna(inplace = True)    
    #Return preprocessed data
    return feables
    

def convert_odds_to_prob(match_odds):
    """ Converts bookkeeper odds to probabilities. """    
    #Define variables
    match_id = match_odds.loc[:,"match_api_id"]
    bookkeeper = match_odds.loc[:,"bookkeeper"]    
    win_odd = match_odds.loc[:,"Win"]
    draw_odd = match_odds.loc[:,"Draw"]
    loss_odd = match_odds.loc[:,"Defeat"]    
    #Converts odds to prob
    win_prob = 1 / win_odd
    draw_prob = 1 / draw_odd
    loss_prob = 1 / loss_odd    
    total_prob = win_prob + draw_prob + loss_prob    
    probs = pd.DataFrame()    
    #Define output format and scale probs by sum over all probs
    probs.loc[:,"match_api_id"] = match_id
    probs.loc[:,"bookkeeper"] = bookkeeper
    probs.loc[:,"Win"] = win_prob / total_prob
    probs.loc[:,"Draw"] = draw_prob / total_prob
    probs.loc[:,"Defeat"] = loss_prob / total_prob
    return probs
    
def get_bookkeeper_data(matches, bookkeepers, horizontal = True):
    """ Aggregates bookkeeper data for all matches and bookkeepers. """    
    bk_data = pd.DataFrame()
    for bookkeeper in bookkeepers:
        #Find columns containing data of bookkeeper
        temp_data = matches.loc[:,(matches.columns.str.contains(bookkeeper))]
        temp_data.loc[:, "bookkeeper"] = str(bookkeeper)
        temp_data.loc[:, "match_api_id"] = matches.loc[:, "match_api_id"]        
        #Rename odds columns and convert to numeric
        cols = temp_data.columns.values
        cols[:3] = ["Win","Draw","Defeat"]
        temp_data.columns = cols
        temp_data.loc[:,"Win"] = pd.to_numeric(temp_data["Win"])
        temp_data.loc[:,"Draw"] = pd.to_numeric(temp_data["Draw"])
        temp_data.loc[:,"Defeat"] = pd.to_numeric(temp_data["Defeat"])        
        #Check if data should be aggregated horizontally
        if(horizontal == True):            
            #Convert data to probs
            temp_data = convert_odds_to_prob(temp_data)
            temp_data.drop("match_api_id", axis = 1, inplace = True)
            temp_data.drop("bookkeeper", axis = 1, inplace = True)            
            #Rename columns with bookkeeper names
            win_name = bookkeeper + "_" + "Win"
            draw_name = bookkeeper + "_" + "Draw"
            defeat_name = bookkeeper + "_" + "Defeat"
            temp_data.columns.values[:3] = [win_name, draw_name, defeat_name]
            #Aggregate data
            bk_data = pd.concat([bk_data, temp_data], axis = 1)
        else:
            #Aggregate vertically
            bk_data = bk_data.append(temp_data, ignore_index = True)    
    #If horizontal add match api id to data
    if(horizontal == True):
        temp_data.loc[:, "match_api_id"] = matches.loc[:, "match_api_id"]
    return bk_data


def _prepare(infile, dbFolder):
    start = time.time()
    with sqlite3.connect(infile) as con:
        print("Reading from the sqlite file...")
        countries = pd.read_sql_query("SELECT * from Country", con)
        matches = pd.read_sql_query("SELECT * from Match", con)
        leagues = pd.read_sql_query("SELECT * from League", con)
        teams = pd.read_sql_query("SELECT * from Team", con)
        players = pd.read_sql("SELECT * FROM Player_Attributes;", con)
    print("Dataset info:")
    print("  Countries: ", countries.shape)
    print("  Matches: ", matches.shape)
    print("  Leagues: ", leagues.shape)
    print("  Teams: ", teams.shape)
    print("  Players: ", players.shape)
    # Reduce match data to fulfill run time requirements
    cols = ["country_id", "league_id", "season", "stage", "date", "match_api_id",
            "home_team_api_id", "away_team_api_id", "home_team_goal",
            "away_team_goal", "home_player_1", "home_player_2", "home_player_3",
            "home_player_4", "home_player_5", "home_player_6", "home_player_7",
            "home_player_8", "home_player_9", "home_player_10", "home_player_11",
            "away_player_1", "away_player_2", "away_player_3", "away_player_4",
            "away_player_5", "away_player_6", "away_player_7", "away_player_8",
            "away_player_9", "away_player_10", "away_player_11"]
    matches = matches.dropna(subset=cols)
    print("Match data (processed): ", matches.shape)
    fifaFile = os.path.join(dbFolder, "fifa.pkl")
    if not os.path.exists(fifaFile):
        fifa = get_fifa_data(matches, players)
        fifa.to_pickle(fifaFile)
    else:
        fifa = pd.read_pickle(fifaFile)
    print("Fifa data: ", fifa.shape)
    bk_cols_selected = ["B365", "BW"]
    feables = create_feables(matches, fifa, bk_cols_selected, get_overall=True)
    feables = convert_cols_categorical_to_numeric(feables)
    print("Feables data: ", feables.shape)
    labels = feables["label"]
    features = feables[feables.columns.difference(["match_api_id", "label"])]
    print("Features data: ", features.shape)
    print("Labels data: " , labels.shape)
    features.to_pickle(os.path.join(dbFolder, "features.pkl"))
    labels.to_pickle(os.path.join(dbFolder, "labels.pkl"))
    print("Time to prepare data: ", (time.time() - start))
    

def prepareImpl(dbFolder, test_size, shuffle):
    unzip(dbFolder, "soccer.zip", "database.sqlite")
    infile = os.path.join(dbFolder, "database.sqlite")
    featurefile = os.path.join(dbFolder, "features.pkl")
    labelfile = os.path.join(dbFolder, "labels.pkl")
    if not os.path.exists(featurefile) or not os.path.exists(labelfile):
        print("Preparing the data...")
        _prepare(infile, dbFolder)
    start = time.time()
    features = pd.read_pickle(featurefile)
    labels = pd.read_pickle(labelfile)
    print("Features: ", features.shape)
    print("Labels: ", labels.shape)
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=test_size,
                                                        shuffle=shuffle,
                                                        random_state=42,
                                                        stratify=labels)
    print("Time to read and split data: ", (time.time() - start))
    return Data(X_train, X_test, y_train, y_test)

def prepare(dbFolder, nrows):
    return prepareImpl(dbFolder, 0.2, True)


labels = [0, 1, 2]
nthreads = get_number_processors()
nTrees = 150

def metrics(y_test, y_pred):
    pred_class = np.argmax(y_pred, axis=1)
    return classification_metrics_multilabel(y_test, pred_class, labels)


xgb_common_params = {
    "eval_metric":      "merror",
    "gamma":            0.1,
    "learning_rate":    0.1,
    "min_child_weight": 5,
    "max_depth":        3,
    "max_leaves":       2**3,
    "num_class":        len(labels),
    "num_round":        nTrees,
    "objective":        "multi:softprob",
    "reg_lambda":       1,
    "subsample":        1,
    "scale_pos_weight": 2,
}

lgb_common_params = {
    "learning_rate":    0.1,
    "metric":           "multi_error",
    "min_child_weight": 5,
    "min_split_gain":   0.1,
    "num_leaves":       2**3,
    "num_round":        nTrees,
    "num_class":        len(labels),
    "objective":        "multiclass",
    "reg_lambda":       1,
    "scale_pos_weight": 2,
    "subsample":        1,
    "task":             "train",
}

cat_common_params = {
    "depth":            3,
    "iterations":       nTrees,
    "l2_leaf_reg":      0.1,
    "learning_rate":    0.1,
    "loss_function":    "Logloss",
}


# NOTES: some benchmarks are disabled!
#  . cat-gpu  currently segfaults
benchmarks = {
    "xgb-cpu":      (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="exact",
                          nthread=nthreads)),
    "xgb-cpu-hist": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, nthread=nthreads,
                          grow_policy="lossguide", tree_method="hist")),
    "xgb-gpu":      (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_exact")),
    "xgb-gpu-hist": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist")),

    "lgbm-cpu":     (True, LgbBenchmark, metrics,
                     dict(lgb_common_params, nthread=nthreads)),
    "lgbm-gpu":     (True, LgbBenchmark, metrics,
                     dict(lgb_common_params, device="gpu")),

    "cat-cpu":      (True, CatBenchmark, metrics,
                     dict(cat_common_params, thread_count=nthreads)),
    "cat-gpu":      (False, CatBenchmark, metrics,
                     dict(cat_common_params, task_type="GPU")),
}
