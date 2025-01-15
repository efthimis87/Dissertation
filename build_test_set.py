import streamlit as st
import pandas as pd
import numpy as np
import itertools
import logging
import requests
import time

import warnings
from sqlalchemy import create_engine, text

# Ignore all warnings
warnings.filterwarnings("ignore")


# Function to fetch fixture data from the API for a specific date
def fetch_fixtures_by_date(date, url, headers):
    querystring = {"date": date}
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json().get("response", [])
    else:
        st.error(f"Failed to retrieve data for date {date}: {response.status_code}")
        return []


# Function to process and clean fixture data
def process_fixtures(fixtures_df):
    # Flatten the 'fixture' column
    if 'fixture' in fixtures_df.columns:
        fixture_df = pd.json_normalize(fixtures_df['fixture'])
        fixtures_df = pd.concat([fixtures_df.drop(columns=['fixture']), fixture_df], axis=1)

    # Rename 'id' column to 'fixture_id'
    fixtures_df.rename(columns={'id': 'fixture_id'}, inplace=True)

    # Drop specified columns
    columns_to_drop = [
        'referee', 'timestamp', 'periods.first', 'periods.second',
        'venue.id', 'venue.name', 'venue.city', 'status.short', 'status.elapsed'
    ]
    fixtures_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Flatten the 'league' column
    if 'league' in fixtures_df.columns:
        league_df = pd.json_normalize(fixtures_df['league'])
        fixtures_df = pd.concat([fixtures_df.drop(columns=['league']), league_df], axis=1)

        # Rename columns
        fixtures_df.rename(columns={'id': 'league_id', 'name': 'league_name'}, inplace=True)

        # Drop specified columns
        league_columns_to_drop = ['logo', 'flag']
        fixtures_df.drop(columns=league_columns_to_drop, inplace=True, errors='ignore')

    # Flatten the 'teams' column
    if 'teams' in fixtures_df.columns:
        teams_df = pd.json_normalize(fixtures_df['teams'])
        fixtures_df = pd.concat([fixtures_df.drop(columns=['teams']), teams_df], axis=1)

        # Rename columns
        fixtures_df.rename(columns={
            'home.id': 'home_team_id', 'home.name': 'home_team_name',
            'away.id': 'away_team_id', 'away.name': 'away_team_name'
        }, inplace=True)

        # Drop specified columns
        team_columns_to_drop = [
            'home.logo', 'home.winner', 'away.logo', 'away.winner'
        ]
        fixtures_df.drop(columns=team_columns_to_drop, inplace=True, errors='ignore')

    # Flatten the 'goals' column
    if 'goals' in fixtures_df.columns:
        goals_df = pd.json_normalize(fixtures_df['goals'])
        fixtures_df = pd.concat([fixtures_df.drop(columns=['goals']), goals_df], axis=1)

    # Drop the 'score' column
    if 'score' in fixtures_df.columns:
        fixtures_df.drop(columns=['score'], inplace=True, errors='ignore')

    return fixtures_df

def remove_duplicates(df):

    # Remove duplicates based on 'fixture_id'
    return df.drop_duplicates(subset=['fixture_id'], keep='first').reset_index(drop=True)


def remove_friendlies(df):
    # Remove rows where 'league_name' contains the word 'Friendlies'
    df = df[~df['league_name'].str.contains('Friendlies', case=False, na=False)]
    return df


def remove_unwanted_status(df):
    unwanted_statuses = ['Match Cancelled', 'Time to be defined', 'Match Postponed']

    # Filter out rows with unwanted statuses
    df = df[~df['status.long'].isin(unwanted_statuses)]

    return df


# Function to get coach information by team ID
def get_coaches_by_team(team_id):
    url = "https://api-football-v1.p.rapidapi.com/v3/coachs"
    querystring = {"team": team_id}

    headers = {
        "x-rapidapi-key": "XXXXXXXXX", #Put API Key
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    # Check response status
    if response.status_code == 200:
        # Parse JSON response
        data = response.json().get("response", [])

        # Check if data is not empty
        if data:
            # Convert to DataFrame
            df = pd.json_normalize(data)

            # Specify columns to drop
            columns_to_drop = [
                'name', 'age', 'nationality', 'height', 'weight',
                'photo', 'birth.date', 'birth.place', 'birth.country', 'team.logo'
            ]

            # Drop specified columns if they exist
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            # Check if 'career' column exists
            if 'career' in df.columns:
                # Flatten the 'career' column
                career_df = df['career'].explode().apply(pd.Series)

                # Add an index to maintain the relationship
                career_df = career_df.reset_index().rename(columns={'index': 'coach_index'})
                df = df.reset_index().rename(columns={'index': 'coach_index'})

                # Merge the flattened 'career' data back to the main DataFrame
                merged_df = pd.merge(df, career_df, on='coach_index', how='left')

                # Drop the original 'career' column
                merged_df = merged_df.drop(columns=['career'])

                # Sort the DataFrame by the 'start' column in descending order
                if 'start' in merged_df.columns:
                    merged_df = merged_df.sort_values(by='start', ascending=False)

                # Keep only the first row of the sorted DataFrame
                merged_df = merged_df.head(1)

                # Keep only the 'id' and 'team.id' columns
                merged_df = merged_df[['id', 'team.id']]

                return merged_df
            else:
                print("'career' column not found in the DataFrame.")
                return None
        else:
            # print(f"No data found for team ID {team_id}.")
            return None
    else:
        print(f"Error fetching data for team ID {team_id}: {response.status_code}")
        return None


# Function to add home team and away team coach IDs to the DataFrame
def add_coach_ids(fixtures_df):
    # Create new columns for home_team_coach_id and away_team_coach_id
    api_call_count = 0

    def get_coach_with_limit(team_id):
        nonlocal api_call_count
        api_call_count += 1

        if api_call_count % 300 == 0:
            time.sleep(60)  # Pause for 60 seconds after every 250 API calls

        return get_coaches_by_team(team_id)

    # Get the coach data for each home team and merge with fixtures_df
    home_coach_data_list = []
    for home_team_id in fixtures_df['home_team_id']:
        coach_data = get_coach_with_limit(home_team_id)
        if coach_data is not None:
            home_coach_data_list.append(coach_data)

    # Merge the home coach data back into the fixtures_df
    merged_home_coach_df = pd.concat(home_coach_data_list, ignore_index=True)

    # Merge the home coach data into the original fixtures_df based on home_team_id
    fixtures_df = pd.merge(fixtures_df, merged_home_coach_df[['id', 'team.id']],
                           left_on='home_team_id', right_on='team.id',
                           how='left')

    # Create the home_team_coach_id column
    fixtures_df['home_team_coach_id'] = fixtures_df['id']

    # Drop the temporary 'id' and 'team.id' columns
    fixtures_df = fixtures_df.drop(columns=['id', 'team.id'])

    # Get the coach data for each away team and merge with fixtures_df
    away_coach_data_list = []
    for away_team_id in fixtures_df['away_team_id']:
        coach_data = get_coach_with_limit(away_team_id)
        if coach_data is not None:
            away_coach_data_list.append(coach_data)

    # Merge the away coach data back into the fixtures_df
    merged_away_coach_df = pd.concat(away_coach_data_list, ignore_index=True)

    # Merge the away coach data into the original fixtures_df based on away_team_id
    fixtures_df = pd.merge(fixtures_df, merged_away_coach_df[['id', 'team.id']],
                           left_on='away_team_id', right_on='team.id',
                           how='left')

    # Create the away_team_coach_id column
    fixtures_df['away_team_coach_id'] = fixtures_df['id']

    # Drop the temporary 'id' and 'team.id' columns
    fixtures_df = fixtures_df.drop(columns=['id', 'team.id'])

    return fixtures_df

def fetch_odds_data(dates):
    # API endpoint for odds
    url_odds = "https://api-football-v1.p.rapidapi.com/v3/odds"
    headers = {
        "x-rapidapi-key": "XXXXXXXXX", #Put API Key
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    all_odds = []

    # Fetch odds data for each date
    for date in dates:
        page = 1
        while True:
            querystring = {"date": date, "page": page}
            response = requests.get(url_odds, headers=headers, params=querystring)
            if response.status_code == 200:
                odds_data = response.json().get("response", [])
                if odds_data:
                    all_odds.extend(odds_data)
                    page += 1
                    time.sleep(1)  # Sleep for 1 second to respect the rate limit
                else:
                    #st.write(f"No more odds data found for date {date} on page {page}")
                    break
            else:
                st.write(f"Failed to retrieve odds data for date {date} on page {page}: {response.status_code}")
                break

    return all_odds

def process_odds_data(all_odds):
    # Convert the collected odds data to a pandas DataFrame
    df = pd.json_normalize(all_odds)

    # List of columns to delete
    columns_to_delete = [
        'league.id', 'league.name', 'league.country', 'league.logo', 'league.flag', 'league.season',
        'fixture.timezone', 'fixture.date', 'fixture.timestamp', 'update'
    ]

    # Drop the specified columns
    df.drop(columns=columns_to_delete, inplace=True, errors='ignore')

    # Flatten the 'bookmakers' column
    df = df.explode('bookmakers')
    bookmakers_df = pd.json_normalize(df['bookmakers'])

    # Combine the original DataFrame with the flattened 'bookmakers' DataFrame
    df = df.drop(columns=['bookmakers']).reset_index(drop=True)
    df = pd.concat([df, bookmakers_df], axis=1)

    # Flatten the 'bets' column inside 'bookmakers'
    df = df.explode('bets')
    bets_df = pd.json_normalize(df['bets'])

    # Combine the DataFrame with the flattened 'bets' DataFrame
    df = df.drop(columns=['bets']).reset_index(drop=True)
    df = pd.concat([df, bets_df], axis=1)

    # Identify columns named 'name' and rename them accordingly
    name_columns = [i for i, col in enumerate(df.columns) if col == 'name']  # Find all indices of columns named 'name']

    # Rename the first 'name' column to 'betting_company' and the second 'name' column to 'odd_category'
    if len(name_columns) > 0:
        df.columns.values[name_columns[0]] = 'betting_company'  # Rename the first 'name' column
    if len(name_columns) > 1:
        df.columns.values[name_columns[1]] = 'odd_category'  # Rename the second 'name' column

    # Ensure all column names are unique
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in
                                                         range(sum(cols == dup))]
    df.columns = cols

    # Filter the DataFrame to only include rows where 'odd_category' is 'Match Winner'
    df_filtered = df[df['odd_category'] == 'Match Winner']

    # Flatten the 'values' column inside 'bets'
    if 'values' in df_filtered.columns:
        # Normalize the dictionaries in 'values' column into separate columns
        values_df = pd.json_normalize(df_filtered['values'])

        # Rename columns in values_df to ensure uniqueness
        values_df.columns = [f"values_{col}" for col in values_df.columns]

        # Combine the original DataFrame with the flattened 'values' DataFrame
        df_filtered = df_filtered.drop(columns=['values']).reset_index(drop=True)
        df_filtered = pd.concat([df_filtered, values_df], axis=1)

    # Flatten the 'values_0' column
    if 'values_0' in df_filtered.columns:
        values_0_df = pd.json_normalize(df_filtered['values_0'])
        values_0_df.columns = [f"values_0_{col}" for col in values_0_df.columns]

        # Combine the original DataFrame with the flattened 'values_0' DataFrame
        df_filtered = df_filtered.drop(columns=['values_0']).reset_index(drop=True)
        df_filtered = pd.concat([df_filtered, values_0_df], axis=1)

    # Delete the column 'values_0_value'
    df_filtered.drop(columns=['values_0_value'], inplace=True, errors='ignore')

    # Rename the column 'values_0_odd' to 'home_team_winning_odd'
    df_filtered.rename(columns={'values_0_odd': 'home_team_winning_odd'}, inplace=True)

    # Flatten the 'values_2' column
    if 'values_2' in df_filtered.columns:
        values_2_df = pd.json_normalize(df_filtered['values_2'])
        values_2_df.columns = [f"values_2_{col}" for col in values_2_df.columns]

        # Combine the original DataFrame with the flattened 'values_2' DataFrame
        df_filtered = df_filtered.drop(columns=['values_2']).reset_index(drop=True)
        df_filtered = pd.concat([df_filtered, values_2_df], axis=1)

    # Delete the column 'values_2_value'
    df_filtered.drop(columns=['values_2_value'], inplace=True, errors='ignore')

    # Rename the column 'values_2_odd' to 'away_team_winning_odd'
    df_filtered.rename(columns={'values_2_odd': 'away_team_winning_odd'}, inplace=True)

    # Delete additional specified columns
    columns_to_delete_final = ['id', 'betting_company', 'id_1', 'odd_category', 'values_1']
    df_filtered.drop(columns=columns_to_delete_final, inplace=True, errors='ignore')

    # Ensure 'home_team_winning_odd' and 'away_team_winning_odd' are numeric
    df_filtered['home_team_winning_odd'] = pd.to_numeric(df_filtered['home_team_winning_odd'], errors='coerce')
    df_filtered['away_team_winning_odd'] = pd.to_numeric(df_filtered['away_team_winning_odd'], errors='coerce')

    # Sort by 'fixture.id'
    df_filtered.sort_values(by='fixture.id', inplace=True)

    # Create a new DataFrame with 'fixture.id' and the averages of 'home_team_winning_odd' and 'away_team_winning_odd'
    df_avg = df_filtered.groupby('fixture.id').agg(
        home_team_winning_odd_avg=('home_team_winning_odd', 'mean'),
        away_team_winning_odd_avg=('away_team_winning_odd', 'mean')
    ).reset_index()

    # Rename columns in the final DataFrame
    df_avg.rename(columns={
        'fixture.id': 'fixture_id',
        'home_team_winning_odd_avg': 'home_team_winning_odd',
        'away_team_winning_odd_avg': 'away_team_winning_odd'
    }, inplace=True)

    return df_avg

def add_is_cup_column(fixtures_df):

    # List of keywords that indicate a cup competition
    cup_keywords = ["cup", "cupp", "copa", "cupe", "coppa", "coupe", "cupa", "cupen", "pokal", "Beker"]

    # Create a copy of the relevant columns
    cup_df = fixtures_df[['fixture_id', 'league_name']].copy()

    # Function to check if any keyword is in the league name
    def check_cup(league_name):
        return int(any(keyword in league_name.lower() for keyword in cup_keywords))

    # Apply the function to create 'is_cup' column
    cup_df['is_cup'] = cup_df['league_name'].apply(check_cup)

    # Merge the 'is_cup' column back into the original DataFrame
    fixtures_df = pd.merge(fixtures_df, cup_df[['fixture_id', 'is_cup']], on='fixture_id', how='left')

    return fixtures_df

def add_recent_fixture_info(fixtures_df, combined_df, num_matches=10):


    # Ensure the dates are in datetime format and timezone-naive
    fixtures_df['date'] = pd.to_datetime(fixtures_df['date']).dt.tz_localize(None)
    combined_df['date'] = pd.to_datetime(combined_df['date']).dt.tz_localize(None)

    # Sort the DataFrames by 'home_team_id'/'team_id' and 'date'
    fixtures_df = fixtures_df.sort_values(by=['home_team_id', 'date'])
    combined_df = combined_df.sort_values(by=['team_id', 'date'])

    # Function to get the most recent fixture info
    def get_most_recent_fixture_info(row, df, team_col, prefix, num_matches=10):
        team_id = row[team_col]
        current_date = row['date']
        fixture_info = {}

        # Filter the combined_df for the same team_id and dates before the current row's date
        past_fixtures = df[(df['team_id'] == team_id) & (df['date'] < current_date)]
        past_fixtures = past_fixtures.sort_values(by='date', ascending=False).head(num_matches)

        # Loop through the past fixtures and add relevant information
        for i in range(1, num_matches + 1):
            if i <= len(past_fixtures):
                most_recent_fixture = past_fixtures.iloc[i - 1]
                fixture_info.update({
                    f'{prefix}_history_fixture_id_{i}': most_recent_fixture['fixture_id'],
                    f'{prefix}_history_match_date_{i}': most_recent_fixture['date'],
                    f'{prefix}_history_coach_{i}': most_recent_fixture['coach_id'],
                    f'{prefix}_history_is_play_home_{i}': most_recent_fixture['play_at_home'],
                    f'{prefix}_history_is_cup_{i}': most_recent_fixture['is_cup']
                })

                # Add goals and league information depending on whether the team played at home
                if most_recent_fixture['play_at_home'] == 1:
                    fixture_info.update({
                        f'{prefix}_history_goal_{i}': most_recent_fixture['home_team_goals'],
                        f'{prefix}_history_opponent_goal_{i}': most_recent_fixture['away_team_goals'],
                        f'{prefix}_history_league_id_{i}': most_recent_fixture['league_id'],
                        f'{prefix}_history_winning_odd_{i}': most_recent_fixture['home_team_winning_odd'],
                        f'{prefix}_history_opponent_winning_odd_{i}': most_recent_fixture['away_team_winning_odd']
                    })
                else:
                    fixture_info.update({
                        f'{prefix}_history_goal_{i}': most_recent_fixture['away_team_goals'],
                        f'{prefix}_history_opponent_goal_{i}': most_recent_fixture['home_team_goals'],
                        f'{prefix}_history_league_id_{i}': most_recent_fixture['league_id'],
                        f'{prefix}_history_winning_odd_{i}': most_recent_fixture['away_team_winning_odd'],
                        f'{prefix}_history_opponent_winning_odd_{i}': most_recent_fixture['home_team_winning_odd']
                    })
            else:
                # If there are not enough past matches, fill with None
                fixture_info.update({
                    f'{prefix}_history_fixture_id_{i}': None,
                    f'{prefix}_history_match_date_{i}': None,
                    f'{prefix}_history_coach_{i}': None,
                    f'{prefix}_history_is_play_home_{i}': None,
                    f'{prefix}_history_is_cup_{i}': None,
                    f'{prefix}_history_goal_{i}': None,
                    f'{prefix}_history_opponent_goal_{i}': None,
                    f'{prefix}_history_league_id_{i}': None,
                    f'{prefix}_history_winning_odd_{i}': None,
                    f'{prefix}_history_opponent_winning_odd_{i}': None,
                })

        return pd.Series(fixture_info)

    # Apply the function for home_team_id and away_team_id
    home_additional_info = fixtures_df.apply(lambda row: get_most_recent_fixture_info(row, combined_df, 'home_team_id', 'home_team', num_matches=num_matches), axis=1)
    away_additional_info = fixtures_df.apply(lambda row: get_most_recent_fixture_info(row, combined_df, 'away_team_id', 'away_team', num_matches=num_matches), axis=1)

    # Combine the additional info with the original DataFrame
    fixtures_df = pd.concat([fixtures_df, home_additional_info, away_additional_info], axis=1)

    return fixtures_df

def save_fixtures_to_excel(fixtures_df, file_path):

    # Copy the DataFrame to avoid modifying the original
    new_df = fixtures_df.copy()

    # Convert all datetime columns to object type (string format)
    for col in new_df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
        new_df[col] = new_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save the new DataFrame to an Excel file
    new_df.to_excel(file_path, index=False)


def reduce_mem_usage(df, df_name, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float64)
                else:
                    df[col] = df[col].astype(np.float64)
    # calculate memory after reduction
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        # reduced memory usage in percent
        diff_pst = 100 * (start_mem - end_mem) / start_mem
        msg = f'{df_name} mem. usage decreased to {end_mem:5.2f} Mb ({diff_pst:.1f}% reduction)'
        print(msg)
    return df


def delete_status_extra_column(df):
    # Check if 'status.extra' column exists in the DataFrame
    if 'status.extra' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'status.extra' column.")

    columns_to_drop_test = ['status.extra']
    df.drop(columns=columns_to_drop_test, inplace=True)

    return df

def remove_rows_all_nan_except_specified(df):
    # Specify the columns to exclude from the check
    columns_to_exclude = ['fixture_id', 'target', 'home_team_name', 'away_team_name', 'match_date', 'league_name',
                          'league_id', 'is_cup']
    # Get the list of columns to check
    columns_to_check = [col for col in df.columns if col not in columns_to_exclude]

    # Check if all specified columns have NaN values in any row
    rows_with_all_nan = df[df[columns_to_check].isnull().all(axis=1)]

    # Drop rows with all NaN values in specified columns
    df = df.drop(rows_with_all_nan.index)

    return df

def remove_rows_with_more_than_50_percent_nan(df):
    # Calculate percentage of NaN values in each row
    nan_percentage_per_row = (df.isnull().mean(axis=1) * 100)

    # Find rows where more than 50% of columns are NaN
    rows_to_remove = nan_percentage_per_row[nan_percentage_per_row > 50].index

    # Remove rows with more than 90% NaN values
    df = df.drop(rows_to_remove)

    return df


def delete_history_fixture_id_columns(df):
    for i in range(1, 11):
        home_col_name = f'home_team_history_fixture_id_{i}'
        away_col_name = f'away_team_history_fixture_id_{i}'

        if home_col_name in df.columns:
            df = df.drop(columns=[home_col_name])

        if away_col_name in df.columns:
            df = df.drop(columns=[away_col_name])

    return df


def drop_home_and_away_columns(df):
    columns_to_drop = ['home', 'away']

    # Drop the columns if they exist in the DataFrame
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    return df

def rename_date_column(df):
    # Rename the 'date' column to 'match_date'
    df = df.rename(columns={'date': 'match_date'})
    return df


def rename_history_league_columns(df):
    for i in range(1, 11):
        # Rename home team columns
        old_home_col_name = f'home_team_team_history_league_id_{i}'
        new_home_col_name = f'home_team_history_league_id_{i}'
        if old_home_col_name in df.columns:
            df = df.rename(columns={old_home_col_name: new_home_col_name})

        # Rename away team columns
        old_away_col_name = f'away_team_team_history_league_id_{i}'
        new_away_col_name = f'away_team_history_league_id_{i}'
        if old_away_col_name in df.columns:
            df = df.rename(columns={old_away_col_name: new_away_col_name})

    return df

def fill_nan_values_away_team_coach_id(df):
    # Define the columns to check
    columns_to_check = ['away_team_coach_id',
                        'away_team_history_coach_1',
                        'away_team_history_coach_2',
                        'away_team_history_coach_3',
                        'away_team_history_coach_4',
                        'away_team_history_coach_5',
                        'away_team_history_coach_6',
                        'away_team_history_coach_7',
                        'away_team_history_coach_8',
                        'away_team_history_coach_9',
                        'away_team_history_coach_10']

    for index, row in df.iterrows():
        if all(pd.isna(row[col]) for col in columns_to_check):
            df.loc[index, columns_to_check] = 999999
        else:
            for col in columns_to_check:
                if pd.notna(row[col]):
                    break
            else:
                for col in columns_to_check:
                    if pd.notna(row[col]):
                        break
            for col in columns_to_check:
                if pd.isna(row[col]):
                    if pd.notna(row['away_team_coach_id']):
                        df.loc[index, col] = row['away_team_coach_id']
                    else:
                        for i in range(1, 11):
                            if pd.notna(row[f'away_team_history_coach_{i}']):
                                df.loc[index, col] = row[f'away_team_history_coach_{i}']
                                break
    return df

def fill_nan_values_home_team_coach_id(df):
    columns_to_check = ['home_team_coach_id',
                        'home_team_history_coach_1',
                        'home_team_history_coach_2',
                        'home_team_history_coach_3',
                        'home_team_history_coach_4',
                        'home_team_history_coach_5',
                        'home_team_history_coach_6',
                        'home_team_history_coach_7',
                        'home_team_history_coach_8',
                        'home_team_history_coach_9',
                        'home_team_history_coach_10']

    for index, row in df.iterrows():
        if all(pd.isna(row[col]) for col in columns_to_check):
            df.loc[index, columns_to_check] = 999999
        else:
            for col in columns_to_check:
                if pd.notna(row[col]):
                    break
            else:
                for col in columns_to_check:
                    if pd.notna(row[col]):
                        break
            for col in columns_to_check:
                if pd.isna(row[col]):
                    if pd.notna(row['home_team_coach_id']):
                        df.loc[index, col] = row['home_team_coach_id']
                    else:
                        for i in range(1, 11):
                            if pd.notna(row[f'home_team_history_coach_{i}']):
                                df.loc[index, col] = row[f'home_team_history_coach_{i}']
                                break
    return df

def replace_value_with_row_index(df):
    # List of columns to check
    columns_to_check = [
        'home_team_coach_id',
        'home_team_history_coach_1',
        'home_team_history_coach_2',
        'home_team_history_coach_3',
        'home_team_history_coach_4',
        'home_team_history_coach_5',
        'home_team_history_coach_6',
        'home_team_history_coach_7',
        'home_team_history_coach_8',
        'home_team_history_coach_9',
        'home_team_history_coach_10'
    ]

    # Iterate over each row of the DataFrame
    for index, row in df.iterrows():
        # Iterate over each specified column
        for col in columns_to_check:
            if col in df.columns:  # Check if the column exists in the DataFrame
                # Ensure the value is treated as an integer or string and check for 999999
                if row[col] == 999999 or str(row[col]) == '999999':
                    df.at[index, col] = 999999 + (index + 1)  # Replace with 999999 + (row_index + 1)

    return df


def replace_value_with_row_index_multiplied(df):
    # List of columns to check
    columns_to_check = [
        'away_team_coach_id',
        'away_team_history_coach_1',
        'away_team_history_coach_2',
        'away_team_history_coach_3',
        'away_team_history_coach_4',
        'away_team_history_coach_5',
        'away_team_history_coach_6',
        'away_team_history_coach_7',
        'away_team_history_coach_8',
        'away_team_history_coach_9',
        'away_team_history_coach_10'
    ]

    # Iterate over each row of the DataFrame
    for index, row in df.iterrows():
        # Iterate over each specified column
        for col in columns_to_check:
            if col in df.columns:  # Check if the column exists in the DataFrame
                # Ensure the value is treated as an integer or string and check for 999999
                if row[col] == 999999 or str(row[col]) == '999999':
                    df.at[index, col] = 999999 * (index + 1)  # Replace with 999999 * (row_index + 1)

    return df


def fill_nan_values_cup(df):
    # Replace 'False' with 0 and 'True' with 1 in the 'is_cup' column
    df['is_cup'] = df['is_cup'].replace({False: 0, True: 1})

    # Filter columns containing the word 'cup'
    cup_columns = [col for col in df.columns if 'cup' in col]

    # Iterate over cup columns
    for col in cup_columns:
        # Calculate the ratio of 1 and 0 values
        ratio_1 = (df[col] == 1).mean()
        ratio_0 = 1 - ratio_1

        # Replace NaN values based on the ratio
        nan_count = df[col].isna().sum()
        nan_fill_values = np.random.choice([0, 1], size=nan_count, p=[ratio_0, ratio_1])

        # Create a Series with NaN filled values and fill NaNs in the column
        nan_fill_series = pd.Series(nan_fill_values, index=df[col].index[df[col].isna()])
        df[col].fillna(nan_fill_series, inplace=True)

    return df


def to_date(df):
    # Iterate over the columns
    for col in df.columns:
        # Check if column name contains the word 'date'
        if 'date' in col.lower():
            # Convert the column to datetime and then keep only the date part
            df[col] = pd.to_datetime(df[col]).dt.date

    return df


import pandas as pd
import numpy as np

def away_team_history_match_date(df):
    # Define the columns to calculate days between
    columns_to_calculate = [
        'match_date',
        'away_team_history_match_date_1',
        'away_team_history_match_date_2',
        'away_team_history_match_date_3',
        'away_team_history_match_date_4',
        'away_team_history_match_date_5',
        'away_team_history_match_date_6',
        'away_team_history_match_date_7',
        'away_team_history_match_date_8',
        'away_team_history_match_date_9',
        'away_team_history_match_date_10'
    ]

    # Convert columns to datetime
    for col in columns_to_calculate:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Handle invalid dates as NaT

    # Calculate the differences between consecutive dates
    diff_df = df[columns_to_calculate].apply(lambda row: row.diff().dt.days, axis=1)

    # Calculate the mean difference for each row, ignoring NaN values
    df['mean_diff'] = diff_df.mean(axis=1, skipna=True)

    # Fill NaN values in each date column
    for col in columns_to_calculate[1:]:  # Skip the first column ('match_date')
        # Fill forward with the previous valid date
        df[col] = df[col].fillna(method='ffill')

        # If still NaN, add the mean difference (converted to timedelta)
        df[col] = df[col].fillna(df['match_date'] + pd.to_timedelta(df['mean_diff'], unit='D'))

    # Drop the 'mean_diff' column
    df.drop(columns='mean_diff', inplace=True)

    return df



def home_team_history_match_date(df):
    # Define the columns to calculate days between
    columns_to_calculate = [
        'match_date',
        'home_team_history_match_date_1',
        'home_team_history_match_date_2',
        'home_team_history_match_date_3',
        'home_team_history_match_date_4',
        'home_team_history_match_date_5',
        'home_team_history_match_date_6',
        'home_team_history_match_date_7',
        'home_team_history_match_date_8',
        'home_team_history_match_date_9',
        'home_team_history_match_date_10'
    ]

    # Convert columns to datetime
    for col in columns_to_calculate:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # Handle invalid dates as NaT

    # Calculate the differences between consecutive dates
    diff_df = df[columns_to_calculate].apply(lambda row: row.diff().dt.days, axis=1)

    # Calculate the mean difference for each row, ignoring NaN values
    df['mean_diff'] = diff_df.mean(axis=1, skipna=True)

    # Fill NaN values in each date column
    for col in columns_to_calculate[1:]:  # Skip the first column ('match_date')
        # Fill forward with the previous valid date
        df[col] = df[col].fillna(method='ffill')

        # If still NaN, add the mean difference (converted to timedelta)
        df[col] = df[col].fillna(df['match_date'] + pd.to_timedelta(df['mean_diff'], unit='D'))

    # Drop the 'mean_diff' column
    df.drop(columns='mean_diff', inplace=True)

    return df




def home_team_history_opponent_goal(df):
    # Define the columns to calculate mean average and fill NaN values
    goal_columns = ['home_team_history_opponent_goal_1', 'home_team_history_opponent_goal_2',
                    'home_team_history_opponent_goal_3', 'home_team_history_opponent_goal_4',
                    'home_team_history_opponent_goal_5', 'home_team_history_opponent_goal_6',
                    'home_team_history_opponent_goal_7', 'home_team_history_opponent_goal_8',
                    'home_team_history_opponent_goal_9', 'home_team_history_opponent_goal_10']

    # Replace non-finite values with 999
    df[goal_columns] = df[goal_columns].replace([np.inf, -np.inf, np.nan], 999)

    # Iterate over each row
    for index, row in df.iterrows():
        # Calculate mean average excluding 999
        mean_average = row[goal_columns][row[goal_columns] != 999].mean()
        # Replace NaN with 0
        if np.isnan(mean_average):
            mean_average = 0
        # Replace 999 with mean average in integer format
        df.loc[index, goal_columns] = row[goal_columns].replace(999, int(mean_average))

    return df

def home_team_history_goal(df):
    # Define the columns to calculate mean average and fill NaN values
    goal_columns = ['home_team_history_goal_1', 'home_team_history_goal_2', 'home_team_history_goal_3',
                    'home_team_history_goal_4', 'home_team_history_goal_5', 'home_team_history_goal_6',
                    'home_team_history_goal_7', 'home_team_history_goal_8', 'home_team_history_goal_9',
                    'home_team_history_goal_10']

    # Replace non-finite values with 999
    df[goal_columns] = df[goal_columns].replace([np.inf, -np.inf, np.nan], 999)

    # Iterate over each row
    for index, row in df.iterrows():
        # Calculate mean average excluding 999
        mean_average = row[goal_columns][row[goal_columns] != 999].mean()
        # Replace NaN with 0
        if np.isnan(mean_average):
            mean_average = 0
        # Replace 999 with mean average in integer format
        df.loc[index, goal_columns] = row[goal_columns].replace(999, int(mean_average))

    return df


def away_team_history_goal(df):
    # Define the columns to calculate mean average and fill NaN values
    goal_columns = ['away_team_history_goal_1', 'away_team_history_goal_2', 'away_team_history_goal_3',
                    'away_team_history_goal_4', 'away_team_history_goal_5', 'away_team_history_goal_6',
                    'away_team_history_goal_7', 'away_team_history_goal_8', 'away_team_history_goal_9',
                    'away_team_history_goal_10']

    # Replace non-finite values with 999
    df[goal_columns] = df[goal_columns].replace([np.inf, -np.inf, np.nan], 999)

    # Iterate over each row
    for index, row in df.iterrows():
        # Calculate mean average excluding 999
        mean_average = row[goal_columns][row[goal_columns] != 999].mean()
        # Replace NaN with 0
        if np.isnan(mean_average):
            mean_average = 0
        # Replace 999 with mean average in integer format
        df.loc[index, goal_columns] = row[goal_columns].replace(999, int(mean_average))

    return df


def away_team_history_opponent_goal(df):
    # Define the columns to calculate mean average and fill NaN values
    goal_columns = ['away_team_history_opponent_goal_1', 'away_team_history_opponent_goal_2',
                    'away_team_history_opponent_goal_3',
                    'away_team_history_opponent_goal_4', 'away_team_history_opponent_goal_5',
                    'away_team_history_opponent_goal_6',
                    'away_team_history_opponent_goal_7', 'away_team_history_opponent_goal_8',
                    'away_team_history_opponent_goal_9',
                    'away_team_history_opponent_goal_10']

    # Replace non-finite values with 999
    df[goal_columns] = df[goal_columns].replace([np.inf, -np.inf, np.nan], 999)

    # Iterate over each row
    for index, row in df.iterrows():
        # Calculate mean average excluding 999
        mean_average = row[goal_columns][row[goal_columns] != 999].mean()
        # Replace NaN with 0
        if np.isnan(mean_average):
            mean_average = 0
        # Replace 999 with mean average in integer format
        df.loc[index, goal_columns] = row[goal_columns].replace(999, int(mean_average))

    return df


def home_team_history_is_play_home(df):
    # Filter columns with 'home_team_history_is_play_home_' prefix
    filtered_columns = [col for col in df.columns if 'home_team_history_is_play_home_' in col]

    # Replace NaN values with 999 in the filtered columns
    df[filtered_columns] = df[filtered_columns].fillna(999)

    # Calculate the ratios of 1 and 0 values in each row
    for index, row in df.iterrows():
        # Calculate the counts of 1 and 0 values in the row
        counts = row[filtered_columns].value_counts(normalize=True)
        ratio_1 = counts.get(1, 0)
        ratio_0 = counts.get(0, 0)

        # Normalize the ratios
        total_ratio = ratio_1 + ratio_0
        if total_ratio > 0:
            ratio_1 /= total_ratio
            ratio_0 /= total_ratio
        else:
            ratio_1 = 1
            ratio_0 = 0

        # Replace 999 values with 1 or 0 based on the calculated ratio
        replacement_values = np.random.choice([1, 0], size=len(filtered_columns), p=[ratio_1, ratio_0])

        # Update the row with the replacement values
        df.loc[index, filtered_columns] = replacement_values

    return df


def away_team_history_is_play_home(df):
    # Filter columns with 'away_team_history_is_play_home_' prefix
    filtered_columns = [col for col in df.columns if 'away_team_history_is_play_home_' in col]

    # Replace NaN values with 999 in the filtered columns
    df[filtered_columns] = df[filtered_columns].fillna(999)

    # Calculate the ratios of 1 and 0 values in each row
    for index, row in df.iterrows():
        # Calculate the counts of 1 and 0 values in the row
        counts = row[filtered_columns].value_counts(normalize=True)
        ratio_1 = counts.get(1, 0)
        ratio_0 = counts.get(0, 0)

        # Normalize the ratios
        total_ratio = ratio_1 + ratio_0
        if total_ratio > 0:
            ratio_1 /= total_ratio
            ratio_0 /= total_ratio
        else:
            ratio_1 = 1
            ratio_0 = 0

        # Replace 999 values with 1 or 0 based on the calculated ratio
        replacement_values = np.random.choice([1, 0], size=len(filtered_columns), p=[ratio_1, ratio_0])

        # Update the row with the replacement values
        df.loc[index, filtered_columns] = replacement_values

    return df

def fill_na_home_team_history_league_id_with_most_common(df):
    # Define the range of columns to consider starting from 10 and going to 1
    cols_to_check_home = [f'home_team_history_league_id_{i}' for i in range(10, 0, -1)]
    cols_to_check_away = [f'away_team_history_league_id_{i}' for i in range(10, 0, -1)]

    # Iterate over each row
    for index, row in df.iterrows():
        # Iterate over each column
        for col_home, col_away in zip(cols_to_check_home, cols_to_check_away):
            # If the value in the home column is NaN
            if pd.isna(row[col_home]):
                # Find the most common value among the corresponding away columns
                most_common_values = df.loc[index, cols_to_check_away].mode().values
                if len(most_common_values) > 0:
                    most_common_value = most_common_values[0]
                    # Fill NaN in the current home column with the most common value from away columns
                    df.at[index, col_home] = most_common_value
                else:
                    # If there is no most common value, fill with NaN
                    df.at[index, col_home] = None

    return df

def home_team_history_winning_odd(df):
    # Define the columns to calculate mean average and fill NaN values
    home_team_history_winning_odd_columns = ['home_team_history_winning_odd_1', 'home_team_history_winning_odd_2',
                                             'home_team_history_winning_odd_3', 'home_team_history_winning_odd_4',
                                             'home_team_history_winning_odd_5', 'home_team_history_winning_odd_6',
                                             'home_team_history_winning_odd_7', 'home_team_history_winning_odd_8',
                                             'home_team_history_winning_odd_9', 'home_team_history_winning_odd_10']

    # Replace non-finite values with 999
    df[home_team_history_winning_odd_columns] = df[home_team_history_winning_odd_columns].replace(
        [np.inf, -np.inf, np.nan, ''], 999)

    # Iterate over each row
    for index, row in df.iterrows():
        # Calculate mean average excluding 999
        mean_average = row[home_team_history_winning_odd_columns][
            row[home_team_history_winning_odd_columns] != 999].mean()
        # Replace NaN with 0
        if np.isnan(mean_average):
            mean_average = 0
        # Replace 999 with mean average in integer format
        df.loc[index, home_team_history_winning_odd_columns] = row[home_team_history_winning_odd_columns].replace(999,
                                                                                                                  int(mean_average))

    return df


def home_team_history_opponent_winning_odd(df):
    # Define the columns to calculate mean average and fill NaN values
    home_team_opponent_winning_odd_columns = ['home_team_history_opponent_winning_odd_1',
                                              'home_team_history_opponent_winning_odd_2',
                                              'home_team_history_opponent_winning_odd_3',
                                              'home_team_history_opponent_winning_odd_4',
                                              'home_team_history_opponent_winning_odd_5',
                                              'home_team_history_opponent_winning_odd_6',
                                              'home_team_history_opponent_winning_odd_7',
                                              'home_team_history_opponent_winning_odd_8',
                                              'home_team_history_opponent_winning_odd_9',
                                              'home_team_history_opponent_winning_odd_10']

    # Replace non-finite values with 999
    df[home_team_opponent_winning_odd_columns] = df[home_team_opponent_winning_odd_columns].replace(
        [np.inf, -np.inf, np.nan, ''], 999)

    # Iterate over each row
    for index, row in df.iterrows():
        # Calculate mean average excluding 999
        mean_average = row[home_team_opponent_winning_odd_columns][
            row[home_team_opponent_winning_odd_columns] != 999].mean()
        # Replace NaN with 0
        if np.isnan(mean_average):
            mean_average = 0
        # Replace 999 with mean average in integer format
        df.loc[index, home_team_opponent_winning_odd_columns] = row[home_team_opponent_winning_odd_columns].replace(999,
                                                                                                                    int(mean_average))

    return df


def away_team_history_winning_odd(df):
    # Define the columns to calculate mean average and fill NaN values
    away_team_winning_odd_columns = ['away_team_history_winning_odd_1', 'away_team_history_winning_odd_2',
                                     'away_team_history_winning_odd_3',
                                     'away_team_history_winning_odd_4', 'away_team_history_winning_odd_5',
                                     'away_team_history_winning_odd_6',
                                     'away_team_history_winning_odd_7', 'away_team_history_winning_odd_8',
                                     'away_team_history_winning_odd_9',
                                     'away_team_history_winning_odd_10']

    # Replace non-finite values with 999
    df[away_team_winning_odd_columns] = df[away_team_winning_odd_columns].replace([np.inf, -np.inf, np.nan, ''], 999)

    # Iterate over each row
    for index, row in df.iterrows():
        # Calculate mean average excluding 999
        mean_average = row[away_team_winning_odd_columns][row[away_team_winning_odd_columns] != 999].mean()
        # Replace NaN with 0
        if np.isnan(mean_average):
            mean_average = 0
        # Replace 999 with mean average in integer format
        df.loc[index, away_team_winning_odd_columns] = row[away_team_winning_odd_columns].replace(999,
                                                                                                  int(mean_average))

    return df


def away_team_history_opponent_winning_odd(df):
    # Define the columns to calculate mean average and fill NaN values
    away_team_opponent_winning_odd_columns = ['away_team_history_opponent_winning_odd_1',
                                              'away_team_history_opponent_winning_odd_2',
                                              'away_team_history_opponent_winning_odd_3',
                                              'away_team_history_opponent_winning_odd_4',
                                              'away_team_history_opponent_winning_odd_5',
                                              'away_team_history_opponent_winning_odd_6',
                                              'away_team_history_opponent_winning_odd_7',
                                              'away_team_history_opponent_winning_odd_8',
                                              'away_team_history_opponent_winning_odd_9',
                                              'away_team_history_opponent_winning_odd_10']

    # Replace non-finite values with 999
    df[away_team_opponent_winning_odd_columns] = df[away_team_opponent_winning_odd_columns].replace(
        [np.inf, -np.inf, np.nan, ''], 999)

    # Iterate over each row
    for index, row in df.iterrows():
        # Calculate mean average excluding 999
        mean_average = row[away_team_opponent_winning_odd_columns][
            row[away_team_opponent_winning_odd_columns] != 999].mean()
        # Replace NaN with 0
        if np.isnan(mean_average):
            mean_average = 0
        # Replace 999 with mean average in integer format
        df.loc[index, away_team_opponent_winning_odd_columns] = row[away_team_opponent_winning_odd_columns].replace(999,
                                                                                                                    int(mean_average))

    return df
def fill_na_home_team_history_league_id_with_most_common(df):
    # Define the range of columns to consider starting from 10 and going to 1
    cols_to_check_home = [f'home_team_history_league_id_{i}' for i in range(10, 0, -1)]
    cols_to_check_away = [f'away_team_history_league_id_{i}' for i in range(10, 0, -1)]

    # Iterate over each row
    for index, row in df.iterrows():
        # Iterate over each column
        for col_home, col_away in zip(cols_to_check_home, cols_to_check_away):
            # If the value in the home column is NaN
            if pd.isna(row[col_home]):
                # Find the most common value among the corresponding away columns
                most_common_values = df.loc[index, cols_to_check_away].mode().values
                if len(most_common_values) > 0:
                    most_common_value = most_common_values[0]
                    # Fill NaN in the current home column with the most common value from away columns
                    df.at[index, col_home] = most_common_value
                else:
                    # If there is no most common value, fill with NaN
                    df.at[index, col_home] = None

    return df

def fill_na_away_team_history_league_id_with_most_common(df):
    # Define the range of columns to consider starting from 10 and going to 1
    cols_to_check_away = [f'away_team_history_league_id_{i}' for i in range(10, 0, -1)]
    cols_to_check_home = [f'home_team_history_league_id_{i}' for i in range(10, 0, -1)]

    # Iterate over each row
    for index, row in df.iterrows():
        # Iterate over each column
        for col_away, col_home in zip(cols_to_check_away, cols_to_check_home):
            # If the value in the away column is NaN
            if pd.isna(row[col_away]):
                # Find the most common value among the corresponding home columns
                most_common_values = df.loc[index, cols_to_check_home].mode().values
                if len(most_common_values) > 0:
                    most_common_value = most_common_values[0]
                    # Fill NaN in the current away column with the most common value from home columns
                    df.at[index, col_away] = most_common_value
                else:
                    # If there is no most common value, fill with NaN
                    df.at[index, col_away] = None

    return df


def delete_rows_with_any_nan(df):
    # Drop rows where any column has NaN values
    df.dropna(inplace=True)

    # Return the modified DataFrame
    return df

def check_away_team_coach_change(df):
    df['away_team_changed_coach_since_last_match'] = (df['away_team_coach_id'] != df['away_team_history_coach_1']).astype(int)
    return df

def check_home_team_coach_change(df):
    df['home_team_changed_coach_since_last_match'] = (df['home_team_coach_id'] != df['home_team_history_coach_1']).astype(int)
    return df


def count_matches_with_same_coach_away(df):
    # Extracting the coach ID of the away team
    away_coach_id = df['away_team_coach_id']

    # Filter columns containing the history of coaches for the away team
    away_history_columns = [col for col in df.columns if 'away_team_history_coach_' in col]

    # Counting how many times the same coach ID appears in the history columns for each row
    df['away_team_matches_with_same_coach'] = df.apply(
        lambda row: sum(row[away_history_columns] == row['away_team_coach_id']), axis=1)

    return df


def count_matches_with_same_coach_home(df):
    # Extracting the coach ID of the home team
    home_coach_id = df['home_team_coach_id']

    # Filter columns containing the history of coaches for the home team
    home_history_columns = [col for col in df.columns if 'home_team_history_coach_' in col]

    # Counting how many times the same coach ID appears in the history columns for each row
    df['home_team_matches_with_same_coach'] = df.apply(
        lambda row: sum(row[home_history_columns] == row['home_team_coach_id']), axis=1)

    return df


def add_days_between_columns(df, history_columns, prefix):
    for i in [1, 3, 6, 10]:
        new_column_name = f'{prefix}_days_between_last_{i}_matches'
        history_date_column = f'{prefix}_team_history_match_date_{i}'
        df[new_column_name] = (pd.to_datetime(df['match_date']) - pd.to_datetime(df[history_date_column])).dt.days
    return df


def add_days_between_matches(df):
    home_history_columns = [f'home_team_history_match_date_{i}' for i in [1, 3, 6, 10]]
    away_history_columns = [f'away_team_history_match_date_{i}' for i in [1, 3, 6, 10]]

    df = add_days_between_columns(df, home_history_columns, 'home')
    df = add_days_between_columns(df, away_history_columns, 'away')

    return df


def add_goal_difference_columns(df):
    history_columns = [col for col in df.columns if 'history_goal_' in col]
    for i in range(1, 11):
        # Home team goal difference columns
        home_goal_column = f'home_team_history_goal_{i}'
        home_opponent_goal_column = f'home_team_history_opponent_goal_{i}'
        home_goal_difference_column = f'home_team_history_goal_difference_{i}'

        # Away team goal difference columns
        away_goal_column = f'away_team_history_goal_{i}'
        away_opponent_goal_column = f'away_team_history_opponent_goal_{i}'
        away_goal_difference_column = f'away_team_history_goal_difference_{i}'

        # Calculate goal differences for home team
        df[home_goal_difference_column] = df[home_goal_column] - df[home_opponent_goal_column]

        # Calculate goal differences for away team
        df[away_goal_difference_column] = df[away_goal_column] - df[away_opponent_goal_column]

    return df



def points_per_match(df):
    # Update history_columns list to include the newly added goal difference columns
    home_history_columns = [col for col in df.columns if 'home_team_history_goal_difference' in col]
    away_history_columns = [col for col in df.columns if 'away_team_history_goal_difference' in col]

    # Calculate points per match for home team
    for column in home_history_columns:
        new_column_name = column.replace('goal_difference', 'point')
        df[new_column_name] = df[column].apply(lambda x: 3 if x > 0 else (1 if x == 0 else 0))

    # Calculate points per match for away team
    for column in away_history_columns:
        new_column_name = column.replace('goal_difference', 'point')
        df[new_column_name] = df[column].apply(lambda x: 3 if x < 0 else (1 if x == 0 else 0))

    return df


def mean_average_home_points(df):
    # Initialize an empty list to store mean average home points for each row
    mean_avg_home_points_list = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize variables to store total points and count of home matches for the current row
        total_points = 0
        home_matches_count = 0

        # Iterate over the history columns to calculate total points and count of home matches for the current row
        for i in range(1, 11):
            is_home_match_column = f'home_team_history_is_play_home_{i}'
            points_column = f'home_team_history_point_{i}'

            # Calculate total points and count of home matches for the current row
            total_points += row[is_home_match_column] * row[points_column]
            home_matches_count += row[is_home_match_column]

        # Calculate the mean average home points for the current row and append it to the list
        mean_avg_home_points = total_points / home_matches_count if home_matches_count != 0 else 0
        mean_avg_home_points_list.append(mean_avg_home_points)

    # Add the list of mean average home points as a new column to the DataFrame
    df['home_team_mean_average_home_points'] = mean_avg_home_points_list

    return df


def mean_average_away_points(df):
    # Initialize an empty list to store mean average away points for each row
    mean_avg_away_points_list = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize variables to store total points and count of away matches for the current row
        total_points = 0
        away_matches_count = 0

        # Iterate over the history columns to calculate total points and count of away matches for the current row
        for i in range(1, 11):
            is_away_match_column = f'away_team_history_is_play_home_{i}'
            points_column = f'away_team_history_point_{i}'

            # Check if it's an away match
            if row[is_away_match_column] == 0:
                total_points += row[points_column]
                away_matches_count += 1

        # Calculate the mean average away points for the current row and append it to the list
        mean_avg_away_points = total_points / away_matches_count if away_matches_count != 0 else 0
        mean_avg_away_points_list.append(mean_avg_away_points)

    # Add the list of mean average away points as a new column to the DataFrame
    df['away_team_mean_average_away_points'] = mean_avg_away_points_list

    return df


def mean_average_home_goal_difference(df):
    # Initialize an empty list to store mean average goal difference for each row
    mean_avg_home_goal_difference_list = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize variables to store total goal difference and count of home matches for the current row
        total_goal_difference = 0
        home_matches_count = 0

        # Iterate over the history columns to calculate total goal difference and count of home matches for the current row
        for i in range(1, 11):
            is_home_match_column = f'home_team_history_is_play_home_{i}'
            goal_difference_column = f'home_team_history_goal_difference_{i}'

            # Check if it's a home match
            if row[is_home_match_column] == 1:
                total_goal_difference += row[goal_difference_column]
                home_matches_count += 1

        # Calculate the mean average goal difference for the current row and append it to the list
        mean_avg_goal_difference = total_goal_difference / home_matches_count if home_matches_count != 0 else 0
        mean_avg_home_goal_difference_list.append(mean_avg_goal_difference)

    # Add the list of mean average goal difference as a new column to the DataFrame
    df['home_team_mean_average_goal_difference_at_home'] = mean_avg_home_goal_difference_list

    return df


def mean_average_away_goal_difference(df):
    # Initialize an empty list to store mean average goal difference for each row
    mean_avg_away_goal_difference_list = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize variables to store total goal difference and count of away matches for the current row
        total_goal_difference = 0
        away_matches_count = 0

        # Iterate over the history columns to calculate total goal difference and count of away matches for the current row
        for i in range(1, 11):
            is_home_match_column = f'away_team_history_is_play_home_{i}'
            goal_difference_column = f'away_team_history_goal_difference_{i}'

            # Check if it's an away match
            if row[is_home_match_column] == 0:
                total_goal_difference += row[goal_difference_column]
                away_matches_count += 1

        # Calculate the mean average goal difference for the current row and append it to the list
        mean_avg_goal_difference = total_goal_difference / away_matches_count if away_matches_count != 0 else 0
        mean_avg_away_goal_difference_list.append(mean_avg_goal_difference)

    # Add the list of mean average goal difference as a new column to the DataFrame
    df['away_team_mean_average_goal_difference_at_away'] = mean_avg_away_goal_difference_list

    return df


def std_dev_home_goal_difference(df):
    # Initialize an empty list to store the standard deviation of goal difference for each row
    std_dev_home_goal_difference_list = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize a list to store goal differences for home matches
        home_goal_differences = []

        # Iterate over the history columns to collect goal differences for home matches
        for i in range(1, 11):
            is_home_match_column = f'home_team_history_is_play_home_{i}'
            goal_difference_column = f'home_team_history_goal_difference_{i}'

            # Check if it's a home match
            if row[is_home_match_column] == 1:
                home_goal_differences.append(row[goal_difference_column])

        # Calculate the standard deviation of goal difference for the current row and append it to the list
        std_dev_goal_difference = np.std(home_goal_differences) if home_goal_differences else 0
        std_dev_home_goal_difference_list.append(std_dev_goal_difference)

    # Add the list of standard deviation of goal difference as a new column to the DataFrame
    df['home_team_std_dev_home_goal_difference'] = std_dev_home_goal_difference_list

    return df


def std_dev_away_goal_difference(df):
    # Initialize an empty list to store the standard deviation of goal difference for each row
    std_dev_away_goal_difference_list = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize a list to store goal differences for away matches
        away_goal_differences = []

        # Iterate over the history columns to collect goal differences for away matches
        for i in range(1, 11):
            is_home_match_column = f'away_team_history_is_play_home_{i}'
            goal_difference_column = f'away_team_history_goal_difference_{i}'

            # Check if it's an away match
            if row[is_home_match_column] == 0:
                away_goal_differences.append(row[goal_difference_column])

        # Calculate the standard deviation of goal difference for the current row and append it to the list
        std_dev_goal_difference = np.std(away_goal_differences) if away_goal_differences else 0
        std_dev_away_goal_difference_list.append(std_dev_goal_difference)

    # Add the list of standard deviation of goal difference as a new column to the DataFrame
    df['away_team_std_dev_away_goal_difference'] = std_dev_away_goal_difference_list

    return df


def home_team_winning_losing_draw_ratios(df):
    # Initialize empty lists to store ratios for each row
    home_winning_ratios = []
    home_losing_ratios = []
    home_draw_ratios = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize variables to count wins, losses, and draws for home matches
        home_wins = 0
        home_losses = 0
        home_draws = 0

        # Iterate over the history columns to determine match outcomes for home matches
        for i in range(1, 11):
            is_home_match_column = f'home_team_history_is_play_home_{i}'
            goal_difference_column = f'home_team_history_goal_difference_{i}'

            # Check if it's a home match
            if row[is_home_match_column] == 1:
                goal_difference = row[goal_difference_column]
                if goal_difference > 0:
                    home_wins += 1
                elif goal_difference < 0:
                    home_losses += 1
                else:
                    home_draws += 1

        # Calculate ratios for home wins, losses, and draws for the current row
        total_home_matches = home_wins + home_losses + home_draws
        home_winning_ratio = home_wins / total_home_matches if total_home_matches != 0 else 0
        home_losing_ratio = home_losses / total_home_matches if total_home_matches != 0 else 0
        home_draw_ratio = home_draws / total_home_matches if total_home_matches != 0 else 0

        # Append the ratios to their respective lists
        home_winning_ratios.append(home_winning_ratio)
        home_losing_ratios.append(home_losing_ratio)
        home_draw_ratios.append(home_draw_ratio)

    # Add the lists of ratios as new columns to the DataFrame
    df['home_team_winning_ratio_at_home'] = home_winning_ratios
    df['home_team_losing_ratio_at_home'] = home_losing_ratios
    df['home_team_draw_ratio_at_home'] = home_draw_ratios

    return df


def away_team_winning_losing_draw_ratios(df):
    # Initialize empty lists to store ratios for each row
    away_winning_ratios = []
    away_losing_ratios = []
    away_draw_ratios = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize variables to count wins, losses, and draws for away matches
        away_wins = 0
        away_losses = 0
        away_draws = 0

        # Iterate over the history columns to determine match outcomes for away matches
        for i in range(1, 11):
            is_home_match_column = f'away_team_history_is_play_home_{i}'
            goal_difference_column = f'away_team_history_goal_difference_{i}'

            # Check if it's an away match
            if row[is_home_match_column] == 0:
                goal_difference = row[goal_difference_column]
                if goal_difference > 0:
                    away_wins += 1
                elif goal_difference < 0:
                    away_losses += 1
                else:
                    away_draws += 1

        # Calculate ratios for away wins, losses, and draws for the current row
        total_away_matches = away_wins + away_losses + away_draws
        away_winning_ratio = away_wins / total_away_matches if total_away_matches != 0 else 0
        away_losing_ratio = away_losses / total_away_matches if total_away_matches != 0 else 0
        away_draw_ratio = away_draws / total_away_matches if total_away_matches != 0 else 0

        # Append the ratios to their respective lists
        away_winning_ratios.append(away_winning_ratio)
        away_losing_ratios.append(away_losing_ratio)
        away_draw_ratios.append(away_draw_ratio)

    # Add the lists of ratios as new columns to the DataFrame
    df['away_team_winning_ratio_at_away'] = away_winning_ratios
    df['away_team_losing_ratio_at_away'] = away_losing_ratios
    df['away_team_draw_ratio_at_away'] = away_draw_ratios

    return df


def home_team_average_points_at_home(df):
    # Initialize an empty list to store average points for each row
    home_average_points = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize variables to calculate average points for home matches
        total_points = 0
        total_home_matches = 0

        # Iterate over the history columns to determine match outcomes for home matches
        for i in range(1, 11):
            is_home_match_column = f'home_team_history_is_play_home_{i}'
            goal_difference_column = f'home_team_history_goal_difference_{i}'

            # Check if it's a home match
            if row[is_home_match_column] == 1:
                total_home_matches += 1
                goal_difference = row[goal_difference_column]
                if goal_difference > 0:
                    total_points += 3  # 3 points for a win
                elif goal_difference == 0:
                    total_points += 1  # 1 point for a draw

        # Calculate the average points for the current row
        average_points = total_points / total_home_matches if total_home_matches != 0 else 0

        # Append the average points to the list
        home_average_points.append(average_points)

    # Add the list of average points as a new column to the DataFrame
    df['home_team_average_points_at_home'] = home_average_points

    return df


def away_team_average_points_at_away(df):
    # Initialize an empty list to store average points for each row
    away_average_points = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize variables to calculate average points for away matches
        total_points = 0
        total_away_matches = 0

        # Iterate over the history columns to determine match outcomes for away matches
        for i in range(1, 11):
            is_home_match_column = f'away_team_history_is_play_home_{i}'
            goal_difference_column = f'away_team_history_goal_difference_{i}'

            # Check if it's an away match
            if row[is_home_match_column] == 0:
                total_away_matches += 1
                goal_difference = row[goal_difference_column]
                if goal_difference > 0:
                    total_points += 3  # 3 points for a win
                elif goal_difference == 0:
                    total_points += 1  # 1 point for a draw

        # Calculate the average points for the current row
        average_points = total_points / total_away_matches if total_away_matches != 0 else 0

        # Append the average points to the list
        away_average_points.append(average_points)

    # Add the list of average points as a new column to the DataFrame
    df['away_team_average_points_at_away'] = away_average_points

    return df


def calculate_home_team_goal_metrics_at_home(df):
    # Initialize empty lists to store mean average goals scored, conceded, difference, and their std for each row
    mean_avg_home_goals_scored_list = []
    mean_avg_home_goals_conceded_list = []
    mean_avg_home_goal_difference_list = []
    std_home_goals_scored_list = []
    std_home_goals_conceded_list = []
    std_home_goal_difference_list = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize variables to store total goals scored, conceded, difference, and count of home matches for the current row
        total_goals_scored = 0
        total_goals_conceded = 0
        total_goal_difference = 0
        home_matches_count = 0
        home_goals_scored = []
        home_goals_conceded = []
        home_goal_difference = []

        # Iterate over the history columns to calculate total goals scored, conceded,
        # difference, and count of home matches for the current row
        for i in range(1, 11):
            is_home_match_column = f'home_team_history_is_play_home_{i}'
            goals_scored_column = f'home_team_history_goal_{i}'
            goals_conceded_column = f'home_team_history_opponent_goal_{i}'
            goal_difference_column = f'home_team_history_goal_difference_{i}'

            # Check if it's a home match
            if row[is_home_match_column] == 1:
                total_goals_scored += row[goals_scored_column]
                total_goals_conceded += row[goals_conceded_column]
                total_goal_difference += row[goal_difference_column]
                home_matches_count += 1
                home_goals_scored.append(row[goals_scored_column])
                home_goals_conceded.append(row[goals_conceded_column])
                home_goal_difference.append(row[goal_difference_column])

        # Calculate the mean average goals scored, conceded, difference, and their std for the current row
        mean_avg_goals_scored = total_goals_scored / home_matches_count if home_matches_count != 0 else 0
        mean_avg_goals_conceded = total_goals_conceded / home_matches_count if home_matches_count != 0 else 0
        mean_avg_goal_difference = total_goal_difference / home_matches_count if home_matches_count != 0 else 0
        std_goals_scored = np.std(home_goals_scored) if home_matches_count != 0 else 0
        std_goals_conceded = np.std(home_goals_conceded) if home_matches_count != 0 else 0
        std_goal_difference = np.std(home_goal_difference) if home_matches_count != 0 else 0

        # Append the results to the corresponding lists
        mean_avg_home_goals_scored_list.append(mean_avg_goals_scored)
        mean_avg_home_goals_conceded_list.append(mean_avg_goals_conceded)
        mean_avg_home_goal_difference_list.append(mean_avg_goal_difference)
        std_home_goals_scored_list.append(std_goals_scored)
        std_home_goals_conceded_list.append(std_goals_conceded)
        std_home_goal_difference_list.append(std_goal_difference)

    # Add the lists of mean average goals scored, conceded, difference, and their std as new columns to the DataFrame
    df['home_team_mean_average_goals_scored_at_home'] = mean_avg_home_goals_scored_list
    df['home_team_mean_average_goals_conceded_at_home'] = mean_avg_home_goals_conceded_list
    df['home_team_mean_average_goal_difference_at_home'] = mean_avg_home_goal_difference_list
    df['home_team_std_goals_scored_at_home'] = std_home_goals_scored_list
    df['home_team_std_goals_conceded_at_home'] = std_home_goals_conceded_list
    df['home_team_std_goal_difference_at_home'] = std_home_goal_difference_list

    return df


def calculate_away_team_goal_metrics_at_away(df):
    # Initialize empty lists to store mean average goals scored, conceded, difference, and their std for each row
    mean_avg_away_goals_scored_list = []
    mean_avg_away_goals_conceded_list = []
    mean_avg_away_goal_difference_list = []
    std_away_goals_scored_list = []
    std_away_goals_conceded_list = []
    std_away_goal_difference_list = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Initialize variables to store total goals scored, conceded, difference, and count of away matches for the current row
        total_goals_scored = 0
        total_goals_conceded = 0
        total_goal_difference = 0
        away_matches_count = 0
        away_goals_scored = []
        away_goals_conceded = []
        away_goal_difference = []

        # Iterate over the history columns to calculate total goals scored, conceded,
        # difference, and count of away matches for the current row
        for i in range(1, 11):
            is_away_match_column = f'away_team_history_is_play_home_{i}'
            goals_scored_column = f'away_team_history_goal_{i}'
            goals_conceded_column = f'away_team_history_opponent_goal_{i}'
            goal_difference_column = f'away_team_history_goal_difference_{i}'

            # Check if it's an away match
            if row[is_away_match_column] == 0:
                total_goals_scored += row[goals_scored_column]
                total_goals_conceded += row[goals_conceded_column]
                total_goal_difference += row[goal_difference_column]
                away_matches_count += 1
                away_goals_scored.append(row[goals_scored_column])
                away_goals_conceded.append(row[goals_conceded_column])
                away_goal_difference.append(row[goal_difference_column])

        # Calculate the mean average goals scored, conceded, difference, and their std for the current row
        mean_avg_goals_scored = total_goals_scored / away_matches_count if away_matches_count != 0 else 0
        mean_avg_goals_conceded = total_goals_conceded / away_matches_count if away_matches_count != 0 else 0
        mean_avg_goal_difference = total_goal_difference / away_matches_count if away_matches_count != 0 else 0
        std_goals_scored = np.std(away_goals_scored) if away_matches_count != 0 else 0
        std_goals_conceded = np.std(away_goals_conceded) if away_matches_count != 0 else 0
        std_goal_difference = np.std(away_goal_difference) if away_matches_count != 0 else 0

        # Append the results to the corresponding lists
        mean_avg_away_goals_scored_list.append(mean_avg_goals_scored)
        mean_avg_away_goals_conceded_list.append(mean_avg_goals_conceded)
        mean_avg_away_goal_difference_list.append(mean_avg_goal_difference)
        std_away_goals_scored_list.append(std_goals_scored)
        std_away_goals_conceded_list.append(std_goals_conceded)
        std_away_goal_difference_list.append(std_goal_difference)

    # Add the lists of mean average goals scored, conceded, difference, and their std as new columns to the DataFrame
    df['away_team_mean_average_goals_scored_at_away'] = mean_avg_away_goals_scored_list
    df['away_team_mean_average_goals_conceded_at_away'] = mean_avg_away_goals_conceded_list
    df['away_team_mean_average_goal_difference_at_away'] = mean_avg_away_goal_difference_list
    df['away_team_std_goals_scored_at_away'] = std_away_goals_scored_list
    df['away_team_std_goals_conceded_at_away'] = std_away_goals_conceded_list
    df['away_team_std_goal_difference_at_away'] = std_away_goal_difference_list

    return df


def calculate_mean_average_home_team_goals_scored(df):
    matches = [3, 6, 10]

    for num_matches in matches:
        goal_columns = [f'home_team_history_goal_{i}' for i in
                        range(1, num_matches + 1)]  # Selecting columns for the specified number of matches
        mean_avg_goals_col = f'home_team_mean_avg_goals_scored_in_{num_matches}_matches'

        # Calculate mean average goals scored across the specified number of matches
        df[mean_avg_goals_col] = df[goal_columns].mean(axis=1)

    return df


def calculate_mean_average_home_team_goals_conceded(df):
    matches = [3, 6, 10]

    for num_matches in matches:
        conceded_columns = [f'home_team_history_opponent_goal_{i}' for i in
                            range(1, num_matches + 1)]  # Selecting columns for the specified number of matches
        mean_avg_conceded_col = f'home_team_mean_avg_goals_conceded_in_{num_matches}_matches'

        # Calculate mean average goals conceded across the specified number of matches
        df[mean_avg_conceded_col] = df[conceded_columns].mean(axis=1)

    return df


def calculate_mean_average_home_team_goals_difference(df):
    matches = [3, 6, 10]

    for num_matches in matches:
        difference_columns = [f'home_team_history_goal_difference_{i}' for i in
                              range(1, num_matches + 1)]  # Selecting columns for the specified number of matches
        mean_avg_difference_col = f'home_team_mean_avg_goal_difference_in_{num_matches}_matches'

        # Calculate mean average goal difference across the specified number of matches
        df[mean_avg_difference_col] = df[difference_columns].mean(axis=1)

    return df


def calculate_mean_average_away_team_goals_scored(df):
    matches = [3, 6, 10]

    for num_matches in matches:
        goals_scored_columns = [f'away_team_history_goal_{i}' for i in
                                range(1, num_matches + 1)]  # Selecting columns for the specified number of matches
        mean_avg_goals_col = f'away_team_mean_avg_goals_scored_in_{num_matches}_matches'

        # Calculate mean average goals scored across the specified number of matches
        df[mean_avg_goals_col] = df[goals_scored_columns].mean(axis=1)

    return df


def calculate_mean_average_away_team_goals_difference(df):
    matches = [3, 6, 10]

    for num_matches in matches:
        goals_difference_columns = [f'away_team_history_goal_difference_{i}' for i in
                                    range(1, num_matches + 1)]  # Selecting columns for the specified number of matches
        mean_avg_goals_difference_col = f'away_team_mean_avg_goals_difference_in_{num_matches}_matches'

        # Calculate mean average goal difference across the specified number of matches
        df[mean_avg_goals_difference_col] = df[goals_difference_columns].mean(axis=1)

    return df


def calculate_mean_average_away_team_goals_conceded(df):
    matches = [3, 6, 10]

    for num_matches in matches:
        goals_conceded_columns = [f'away_team_history_opponent_goal_{i}' for i in
                                  range(1, num_matches + 1)]  # Selecting columns for the specified number of matches
        mean_avg_goals_conceded_col = f'away_team_mean_avg_goals_conceded_in_{num_matches}_matches'

        # Calculate mean average goals conceded across the specified number of matches
        df[mean_avg_goals_conceded_col] = df[goals_conceded_columns].mean(axis=1)

    return df


def calculate_mean_average_home_team_points(df):
    matches = [3, 6, 10]

    for num_matches in matches:
        points_columns = [f'home_team_history_point_{i}' for i in
                          range(1, num_matches + 1)]  # Selecting columns for the specified number of matches
        mean_avg_points_col = f'home_team_mean_avg_points_in_{num_matches}_matches'

        # Calculate mean average points across the specified number of matches
        df[mean_avg_points_col] = df[points_columns].mean(axis=1)

    return df


def calculate_mean_average_home_team_points_std(df):
    matches = [3, 6, 10]

    for num_matches in matches:
        points_columns = [f'home_team_history_point_{i}' for i in
                          range(1, num_matches + 1)]  # Selecting columns for the specified number of matches
        std_points_col = f'home_team_std_points_in_{num_matches}_matches'

        # Calculate standard deviation of points across the specified number of matches
        df[std_points_col] = df[points_columns].std(axis=1)

    return df


def calculate_mean_average_away_team_points_std(df):
    matches = [3, 6, 10]

    for num_matches in matches:
        points_columns = [f'away_team_history_point_{i}' for i in
                          range(1, num_matches + 1)]  # Selecting columns for the specified number of matches
        std_points_col = f'away_team_std_points_in_{num_matches}_matches'

        # Calculate standard deviation of points across the specified number of matches
        df[std_points_col] = df[points_columns].std(axis=1)

    return df


def calculate_mean_average_away_team_points(df):
    matches = [3, 6, 10]

    for num_matches in matches:
        points_columns = [f'away_team_history_point_{i}' for i in
                          range(1, num_matches + 1)]  # Selecting columns for the specified number of matches
        mean_avg_points_col = f'away_team_mean_avg_points_in_{num_matches}_matches'

        # Calculate mean average points across the specified number of matches
        df[mean_avg_points_col] = df[points_columns].mean(axis=1)

    return df


def calculate_league_average_goal(df):
    unique_league_ids = df['league_id'].unique()
    # unique_league_ids = [636, 752]

    combined_df = generate_combined_df(df, unique_league_ids)
    combined_df1 = generate_combined_df1(df, unique_league_ids)
    # Concatenate the two DataFrames
    final_combined_df = pd.concat([combined_df, combined_df1], ignore_index=True)

    # Calculate mean averages per league_id
    mean_averages = final_combined_df.groupby('league_id').mean().reset_index()

    # Merge mean averages with df1 on league_id
    df_merged = pd.merge(df, mean_averages, on='league_id', how='left')

    df_merged.rename(columns={'home_team_history_goal': 'average_home_team_goal_scored_by_league',
                              'away_team_history_goal': 'average_away_team_goal_scored_by_league',
                              'total_goals_per_game': 'average_total_goals_scored_by_league'}, inplace=True)

    return df_merged



def create_new_df(filtered_df, suffix):
    # Initialize an empty list to store rows (faster than appending rows one by one to a DataFrame)
    rows_list = []

    # Iterate over the filtered DataFrame
    for index, row in filtered_df.iterrows():
        if row[f'home_team_history_is_play_home_{suffix}'] == 1:
            new_row = {
                'league_id': row[f'home_team_history_league_id_{suffix}'],
                'home_team_history_goal': row[f'home_team_history_goal_{suffix}'],
                'away_team_history_goal': row[f'home_team_history_opponent_goal_{suffix}']
            }
        else:
            new_row = {
                'league_id': row[f'home_team_history_league_id_{suffix}'],
                'home_team_history_goal': row[f'home_team_history_opponent_goal_{suffix}'],
                'away_team_history_goal': row[f'home_team_history_goal_{suffix}']
            }

        # Append the new row to the list
        rows_list.append(new_row)

    # Convert the list of rows to a DataFrame
    new_df = pd.DataFrame(rows_list, columns=['league_id', 'home_team_history_goal', 'away_team_history_goal'])

    # Calculate the total goals per game
    new_df['total_goals_per_game'] = new_df['home_team_history_goal'] + new_df['away_team_history_goal']

    return new_df



def generate_combined_df(df, league_ids):
    combined_dfs = []
    for league_id in league_ids:
        combined_df = pd.DataFrame(
            columns=['league_id', 'home_team_history_goal', 'away_team_history_goal', 'total_goals_per_game'])
        for i in range(1, 11):
            filtered_df = df[df[f'home_team_history_league_id_{i}'] == league_id]
            new_df = create_new_df(filtered_df, i)
            combined_df = pd.concat([combined_df, new_df], ignore_index=True)
        combined_dfs.append(combined_df)
    return pd.concat(combined_dfs, ignore_index=True)


def create_new_df1(filtered_df, suffix):
    # Initialize an empty list to store rows (faster than appending rows one by one to a DataFrame)
    rows_list = []

    # Iterate over the filtered DataFrame
    for index, row in filtered_df.iterrows():
        if row[f'away_team_history_is_play_home_{suffix}'] == 1:
            new_row = {
                'league_id': row[f'away_team_history_league_id_{suffix}'],
                'home_team_history_goal': row[f'away_team_history_goal_{suffix}'],
                'away_team_history_goal': row[f'away_team_history_opponent_goal_{suffix}']
            }
        else:
            new_row = {
                'league_id': row[f'away_team_history_league_id_{suffix}'],
                'home_team_history_goal': row[f'away_team_history_opponent_goal_{suffix}'],
                'away_team_history_goal': row[f'away_team_history_goal_{suffix}']
            }

        # Append the new row to the list
        rows_list.append(new_row)

    # Convert the list of rows to a DataFrame
    new_df1 = pd.DataFrame(rows_list, columns=['league_id', 'home_team_history_goal', 'away_team_history_goal'])

    # Calculate the total goals per game
    new_df1['total_goals_per_game'] = new_df1['home_team_history_goal'] + new_df1['away_team_history_goal']

    return new_df1



def generate_combined_df1(df, league_ids):
    combined_dfs1 = []
    for league_id in league_ids:
        combined_df1 = pd.DataFrame(
            columns=['league_id', 'home_team_history_goal', 'away_team_history_goal', 'total_goals_per_game'])
        for i in range(1, 11):
            filtered_df = df[df[f'away_team_history_league_id_{i}'] == league_id]
            new_df = create_new_df1(filtered_df, i)
            combined_df1 = pd.concat([combined_df1, new_df], ignore_index=True)
        combined_dfs1.append(combined_df1)
    return pd.concat(combined_dfs1, ignore_index=True)


def calculate_team_strengths_and_expected_goals(df):
    # Calculate home team attack strength
    df['home_team_league_attack_strength'] = df['home_team_mean_avg_goals_scored_in_10_matches'] / df[
        'average_total_goals_scored_by_league']

    # Calculate home team defense strength
    df['home_team_league_defense_strength'] = df['home_team_mean_avg_goals_conceded_in_10_matches'] / df[
        'average_total_goals_scored_by_league']

    # Calculate away team attack strength
    df['away_team_league_attack_strength'] = df['away_team_mean_avg_goals_scored_in_10_matches'] / df[
        'average_total_goals_scored_by_league']

    # Calculate away team defense strength
    df['away_team_league_defense_strength'] = df['away_team_mean_avg_goals_conceded_in_10_matches'] / df[
        'average_total_goals_scored_by_league']

    # Calculate home team expected goals
    df['home_team_league_expected_goal'] = df['average_home_team_goal_scored_by_league'] * df[
        'home_team_league_attack_strength'] * df['away_team_league_defense_strength']

    # Calculate away team expected goals
    df['away_team_league_expected_goal'] = df['average_away_team_goal_scored_by_league'] * df[
        'away_team_league_attack_strength'] * df['home_team_league_defense_strength']

    return df




# Set up logging
logging.basicConfig(level=logging.INFO)


def calculate_coach_average_goal_scored_and_conceded(df):
    logging.info("Starting 'calculate_coach_average_goal_scored_and_conceded' function.")

    if 'home_team_coach_id' not in df.columns or 'away_team_coach_id' not in df.columns:
        raise ValueError("DataFrame must contain 'home_team_coach_id' and 'away_team_coach_id' columns.")

    coach_id = list(set(df['home_team_coach_id'].tolist() + df['away_team_coach_id'].tolist()))

    try:
        combined_df10 = generate_combined_df_coach(df, coach_id)
        combined_df110 = generate_combined_df1_coach(df, coach_id)
    except Exception as e:
        logging.error(f"An error occurred while generating combined DataFrames: {e}")
        raise

    final_combined_df10 = pd.concat([combined_df10, combined_df110], ignore_index=True)

    mean_averages = final_combined_df10.groupby('coach_id').mean().reset_index()

    df_merged10 = pd.merge(df, mean_averages, left_on='home_team_coach_id', right_on='coach_id', how='left')
    df_merged10 = pd.merge(df_merged10, mean_averages, left_on='away_team_coach_id', right_on='coach_id', how='left')

    df_merged10.drop(columns=['coach_id_x', 'coach_id_y'], inplace=True)

    df_merged10.rename(columns={
        'home_team_history_goal_x': 'home_team_coach_average_goal_scored',
        'away_team_history_goal_x': 'home_team_coach_average_goal_conceded',
        'home_team_history_goal_y': 'away_team_coach_average_goal_scored',
        'away_team_history_goal_y': 'away_team_coach_average_goal_conceded'
    }, inplace=True)

    logging.info("Completed 'calculate_coach_average_goal_scored_and_conceded' function.")
    return df_merged10


def create_new_df_coach(filtered_df10, suffix):
    new_rows = []

    for index, row in filtered_df10.iterrows():
        if row[f'home_team_history_is_play_home_{suffix}'] == 1:
            new_row = {
                'coach_id': row[f'home_team_history_coach_{suffix}'],
                'home_team_history_goal': row[f'home_team_history_goal_{suffix}'],
                'away_team_history_goal': row[f'home_team_history_opponent_goal_{suffix}']
            }
        else:
            new_row = {
                'coach_id': row[f'home_team_history_coach_{suffix}'],
                'home_team_history_goal': row[f'home_team_history_opponent_goal_{suffix}'],
                'away_team_history_goal': row[f'home_team_history_goal_{suffix}']
            }
        new_rows.append(new_row)

    new_df10 = pd.DataFrame(new_rows, columns=['coach_id', 'home_team_history_goal', 'away_team_history_goal'])
    return new_df10


def generate_combined_df_coach(df, coach_id):
    combined_dfs10 = []

    for coach in coach_id:
        combined_df10 = pd.DataFrame(columns=['coach_id', 'home_team_history_goal', 'away_team_history_goal'])
        for i in range(1, 11):
            filtered_df10 = df[df[f'home_team_history_coach_{i}'] == coach]
            new_df10 = create_new_df_coach(filtered_df10, i)
            combined_df10 = pd.concat([combined_df10, new_df10], ignore_index=True)
        combined_dfs10.append(combined_df10)

    return pd.concat(combined_dfs10, ignore_index=True)


def create_new_df_coach_1(filtered_df10, suffix):
    new_rows = []

    for index, row in filtered_df10.iterrows():
        if row[f'away_team_history_is_play_home_{suffix}'] == 1:
            new_row = {
                'coach_id': row[f'away_team_history_coach_{suffix}'],
                'home_team_history_goal': row[f'away_team_history_goal_{suffix}'],
                'away_team_history_goal': row[f'away_team_history_opponent_goal_{suffix}']
            }
        else:
            new_row = {
                'coach_id': row[f'away_team_history_coach_{suffix}'],
                'home_team_history_goal': row[f'away_team_history_opponent_goal_{suffix}'],
                'away_team_history_goal': row[f'away_team_history_goal_{suffix}']
            }
        new_rows.append(new_row)

    new_df110 = pd.DataFrame(new_rows, columns=['coach_id', 'home_team_history_goal', 'away_team_history_goal'])
    return new_df110


def generate_combined_df1_coach(df, coach_id):
    combined_dfs110 = []

    for coach in coach_id:
        combined_df110 = pd.DataFrame(columns=['coach_id', 'home_team_history_goal', 'away_team_history_goal'])
        for i in range(1, 11):
            filtered_df10 = df[df[f'away_team_history_coach_{i}'] == coach]
            new_df110 = create_new_df_coach_1(filtered_df10, i)
            combined_df110 = pd.concat([combined_df110, new_df110], ignore_index=True)
        combined_dfs110.append(combined_df110)

    return pd.concat(combined_dfs110, ignore_index=True)



# Set up logging
logging.basicConfig(level=logging.INFO)

def create_new_dfs(df):
    logging.info("Starting 'create_new_dfs' function.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Initialize lists to hold new rows
    new_rows1 = []
    new_rows2 = []

    for index, row in df.iterrows():
        for i in range(1, 11):
            try:
                new_row1 = {
                    'coach_id': row[f'home_team_history_coach_{i}'],
                    'league_id': row[f'home_team_history_league_id_{i}'],
                    'play_at_home': row[f'home_team_history_is_play_home_{i}'],
                    'home_team_history_goal': row[f'home_team_history_goal_{i}'],
                    'away_team_history_goal': row[f'home_team_history_opponent_goal_{i}']
                }
                new_rows1.append(new_row1)

                new_row2 = {
                    'coach_id': row[f'away_team_history_coach_{i}'],
                    'league_id': row[f'away_team_history_league_id_{i}'],
                    'play_at_home': row[f'away_team_history_is_play_home_{i}'],
                    'home_team_history_goal': row[f'away_team_history_goal_{i}'],
                    'away_team_history_goal': row[f'away_team_history_opponent_goal_{i}']
                }
                new_rows2.append(new_row2)
            except KeyError as e:
                logging.error(f"Missing key for index {index} and i={i}: {e}")
                continue  # Skip to the next iteration if a key is missing

    # Create DataFrames from lists
    new_df1 = pd.DataFrame(new_rows1)
    new_df2 = pd.DataFrame(new_rows2)

    # Concatenate the two DataFrames
    combined_df = pd.concat([new_df1, new_df2], ignore_index=True)

    # Splitting into two DataFrames based on 'play_at_home' column
    home_df = combined_df[combined_df['play_at_home'] == 1].drop(columns=['play_at_home'])
    away_df = combined_df[combined_df['play_at_home'] == 0].drop(columns=['play_at_home'])

    # Calculate mean averages
    home_df['home_team_coach_average_goal_scored_by_league'] = home_df.groupby(['coach_id', 'league_id'])[
        'home_team_history_goal'].transform('mean')
    home_df['home_team_coach_average_goal_conceded_by_league'] = home_df.groupby(['coach_id', 'league_id'])[
        'away_team_history_goal'].transform('mean')

    away_df['away_team_coach_average_goal_scored_by_league'] = away_df.groupby(['coach_id', 'league_id'])[
        'home_team_history_goal'].transform('mean')
    away_df['away_team_coach_average_goal_conceded_by_league'] = away_df.groupby(['coach_id', 'league_id'])[
        'away_team_history_goal'].transform('mean')

    logging.info("Completed 'create_new_dfs' function.")
    return home_df, away_df



def merge_home_df_to_original_df(df, home_df):
    for index, row in df.iterrows():
        mask = (home_df['league_id'] == row['league_id']) & (home_df['coach_id'] == row['home_team_coach_id'])
        if mask.any():
            values = home_df.loc[mask, ['home_team_coach_average_goal_scored_by_league',
                                        'home_team_coach_average_goal_conceded_by_league']].values
            # Extracting the values from a list within the values variable
            values = values[0] if len(values) > 0 else [None, None]
            # Assign the merged values to the original DataFrame
            df.at[index, 'home_team_coach_average_goal_scored_by_league'] = values[0]
            df.at[index, 'home_team_coach_average_goal_conceded_by_league'] = values[1]

    return df


def merge_away_df_to_original_df(df, away_df):
    for index, row in df.iterrows():
        mask = (away_df['league_id'] == row['league_id']) & (away_df['coach_id'] == row['away_team_coach_id'])
        if mask.any():
            values = away_df.loc[mask, ['away_team_coach_average_goal_scored_by_league',
                                        'away_team_coach_average_goal_conceded_by_league']].values
            # Extracting the values from a list within the values variable
            values = values[0] if len(values) > 0 else [None, None]
            # Assign the merged values to the original DataFrame
            df.at[index, 'away_team_coach_average_goal_scored_by_league'] = values[0]
            df.at[index, 'away_team_coach_average_goal_conceded_by_league'] = values[1]

    return df


def calculate_coach_strengths_and_expected_goals_by_league(df):
    df['home_team_coach_attack_strength_by_league'] = df['home_team_coach_average_goal_scored_by_league'] / df[
        'average_total_goals_scored_by_league']
    df['home_team_coach_defence_strength_by_league'] = df['home_team_coach_average_goal_conceded_by_league'] / df[
        'average_total_goals_scored_by_league']
    df['away_team_coach_attack_strength_by_league'] = df['away_team_coach_average_goal_scored_by_league'] / df[
        'average_total_goals_scored_by_league']
    df['away_team_coach_defence_strength_by_league'] = df['away_team_coach_average_goal_conceded_by_league'] / df[
        'average_total_goals_scored_by_league']
    df['home_team_coach_expected_goal_by_league'] = df['average_home_team_goal_scored_by_league'] * df[
        'home_team_coach_attack_strength_by_league'] * df['away_team_coach_defence_strength_by_league']
    df['away_team_coach_expected_goal_by_league'] = df['average_away_team_goal_scored_by_league'] * df[
        'away_team_coach_attack_strength_by_league'] * df['home_team_coach_defence_strength_by_league']

    return df

def home_team_difference_winning_odd_history_columns(df):
    num_cols = 10
    for i in range(1, num_cols + 1):
        home_winning_odd_col = f'home_team_history_winning_odd_{i}'
        opponent_winning_odd_col = f'home_team_history_opponent_winning_odd_{i}'
        diff_col = f'home_team_difference_winning_odd_history_{i}'
        df[diff_col] = df.apply(lambda row: row[home_winning_odd_col] - row[opponent_winning_odd_col] if not pd.isna(row[home_winning_odd_col]) and not pd.isna(row[opponent_winning_odd_col]) else np.nan, axis=1)
    return df

def away_team_difference_winning_odd_history_columns(df):
    num_cols = 10
    for i in range(1, num_cols + 1):
        away_winning_odd_col = f'away_team_history_winning_odd_{i}'
        opponent_winning_odd_col = f'away_team_history_opponent_winning_odd_{i}'
        diff_col = f'away_team_difference_winning_odd_history_{i}'
        df[diff_col] = df.apply(lambda row: row[away_winning_odd_col] - row[opponent_winning_odd_col] if not pd.isna(row[away_winning_odd_col]) and not pd.isna(row[opponent_winning_odd_col]) else np.nan, axis=1)
    return df


def calculate_home_team_winning_odd_difference_at_home(df):
    new_rows = []
    for i in range(1, 11):  # Assuming there are 10 columns to check
        mask = df[f'home_team_history_is_play_home_{i}'] == 1
        filtered_df = df[mask]
        new_rows.extend(
            filtered_df[['fixture_id', f'home_team_difference_winning_odd_history_{i}']].dropna().values.tolist())

    new_df = pd.DataFrame(new_rows, columns=['fixture_id', 'home_team_difference_winning_odd_history'])
    grouped_df = new_df.groupby('fixture_id')['home_team_difference_winning_odd_history']
    avg_winning_odd_diff = grouped_df.mean().rename('average_home_team_winning_odd_difference_at_home').reset_index()
    median_winning_odd_diff = grouped_df.median().rename(
        'median_home_team_winning_odd_difference_at_home').reset_index()
    std_winning_odd_diff = grouped_df.std().rename('std_home_team_winning_odd_difference_at_home').reset_index()

    merged_df = df.merge(avg_winning_odd_diff, on='fixture_id', how='left')
    merged_df = merged_df.merge(median_winning_odd_diff, on='fixture_id', how='left')
    merged_df = merged_df.merge(std_winning_odd_diff, on='fixture_id', how='left')

    return merged_df


def calculate_away_team_winning_odd_difference_at_away(df):
    new_rows = []
    for i in range(1, 11):  # Assuming there are 10 columns to check
        mask = df[f'away_team_history_is_play_home_{i}'] == 0
        filtered_df = df[mask]
        new_rows.extend(
            filtered_df[['fixture_id', f'away_team_difference_winning_odd_history_{i}']].dropna().values.tolist())

    new_df = pd.DataFrame(new_rows, columns=['fixture_id', 'away_team_difference_winning_odd_history'])
    grouped_df = new_df.groupby('fixture_id')['away_team_difference_winning_odd_history']
    avg_winning_odd_diff = grouped_df.mean().rename('average_away_team_winning_odd_difference_away').reset_index()
    median_winning_odd_diff = grouped_df.median().rename('median_away_team_winning_odd_difference_away').reset_index()
    std_winning_odd_diff = grouped_df.std().rename('std_away_team_winning_odd_difference_away').reset_index()

    merged_df = df.merge(avg_winning_odd_diff, on='fixture_id', how='left')
    merged_df = merged_df.merge(median_winning_odd_diff, on='fixture_id', how='left')
    merged_df = merged_df.merge(std_winning_odd_diff, on='fixture_id', how='left')

    return merged_df


def calculate_home_team_winning_odd_statistics(df):
    indices = [3, 6, 10]
    for i in indices:
        column_name = f'home_team_difference_winning_odd_history_{i}'
        avg_column_name = f'home_team_average_difference_winning_odd_on_{i}_games'
        median_column_name = f'home_team_median_difference_winning_odd_on_{i}_games'
        std_column_name = f'home_team_std_difference_winning_odd_on_{i}_games'

        df[avg_column_name] = df[column_name].rolling(window=i).mean()
        df[median_column_name] = df[column_name].rolling(window=i).median()
        df[std_column_name] = df[column_name].rolling(window=i).std()

        # Fill NaN values with 0 if there are less than i values available for calculation
        df[avg_column_name].fillna(0, inplace=True)
        df[median_column_name].fillna(0, inplace=True)
        df[std_column_name].fillna(0, inplace=True)

    return df


def calculate_away_team_winning_odd_statistics(df):
    indices = [3, 6, 10]
    for i in indices:
        column_name = f'away_team_difference_winning_odd_history_{i}'
        avg_column_name = f'away_team_average_difference_winning_odd_on_{i}_games'
        median_column_name = f'away_team_median_difference_winning_odd_on_{i}_games'
        std_column_name = f'away_team_std_difference_winning_odd_on_{i}_games'

        df[avg_column_name] = df[column_name].rolling(window=i).mean()
        df[median_column_name] = df[column_name].rolling(window=i).median()
        df[std_column_name] = df[column_name].rolling(window=i).std()

        # Fill NaN values with 0 if there are less than i values available for calculation
        df[avg_column_name].fillna(0, inplace=True)
        df[median_column_name].fillna(0, inplace=True)
        df[std_column_name].fillna(0, inplace=True)

    return df


def calculate_elo_ratings(df):
    num_cols = 10  # Assuming there are 10 historical match columns

    # Elo for historical matches (i=1 to 10)
    for i in range(1, num_cols + 1):
        home_team_winning_odd_col = f'home_team_history_winning_odd_{i}'
        home_team_opponent_winning_odd_col = f'home_team_history_opponent_winning_odd_{i}'

        # Replace missing values with a default value (e.g., 0) for Elo winning_odd calculation
        home_team_winning_odd = df[home_team_winning_odd_col].fillna(0)
        home_team_opponent_winning_odd = df[home_team_opponent_winning_odd_col].fillna(0)

        # Calculate Elo winning_odd for the home team
        df[f'home_team_history_elo_winning_odd_{i}'] = 1 / (
                    1 + 10 ** ((home_team_opponent_winning_odd - home_team_winning_odd) / 10))

        away_team_winning_odd_col = f'away_team_history_winning_odd_{i}'
        away_team_opponent_winning_odd_col = f'away_team_history_opponent_winning_odd_{i}'

        # Replace missing values with a default value (e.g., 0) for Elo winning_odd calculation
        away_team_winning_odd = df[away_team_winning_odd_col].fillna(0)
        away_team_opponent_winning_odd = df[away_team_opponent_winning_odd_col].fillna(0)

        # Calculate Elo winning_odd for the away team
        df[f'away_team_history_elo_winning_odd_{i}'] = 1 / (
                    1 + 10 ** ((away_team_opponent_winning_odd - away_team_winning_odd) / 10))

    # Elo for the current match (i=0)
    home_team_winning_odd = df['home_team_winning_odd']  # Current home team winning odds
    away_team_winning_odd = df['away_team_winning_odd']  # Current away team winning odds

    # Calculate Elo for the home team based on current match odds
    df['home_team_elo_winning_odd'] = 1 / (1 + 10 ** ((away_team_winning_odd - home_team_winning_odd) / 10))

    # Calculate Elo for the away team based on current match odds
    df['away_team_elo_winning_odd'] = 1 / (1 + 10 ** ((home_team_winning_odd - away_team_winning_odd) / 10))

    return df

def fill_na_and_inf_with_zero(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf values with NaN
    df.fillna(0, inplace=True)  # Fill NaN values with 0
    return df


# Streamlit UI for selecting one date
def run():
    # Set up logging to display messages in the console
    import logging
    import time
    import pandas as pd
    import streamlit as st
    from sqlalchemy import create_engine, text

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    st.header("Build Test Set")

    # API endpoints and headers
    url_fixtures = "https://api-football-v1.p.rapidapi.com/v3/fixtures"

    headers = {
        "X-RapidAPI-Key": "XXXXXXXXX", #Put API Key
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }

    # Single date picker
    picked_date = st.date_input("Pick a date", pd.to_datetime("today"))
    # Save the selected date in 'date' as a string in "YYYY-MM-DD" format
    date = picked_date.strftime("%Y-%m-%d")

    # Display the selected date
    st.write("Selected Date:", date)

    # Button to fetch data
    if st.button("Select Date for the Football Matches"):
        all_fixtures = []
        start_time = time.time()  # Start the timer

        # Fetch fixture data for the selected date
        fixture_data = fetch_fixtures_by_date(date, url_fixtures, headers)
        if fixture_data:
            all_fixtures.extend(fixture_data)

        # Convert fixture data to DataFrame
        fixtures_df = pd.DataFrame(all_fixtures)

        combined_df = pd.read_csv(
            'TEST_COMBINED_DF_TEMP.csv')

        if not fixtures_df.empty:
            # Process the fixture data
            fixtures_df = process_fixtures(fixtures_df)
            fixtures_df = remove_friendlies(fixtures_df)
            fixtures_df = remove_unwanted_status(fixtures_df)

            # Add coach IDs
            fixtures_df = add_coach_ids(fixtures_df)
            fixtures_df = remove_duplicates(fixtures_df)

            # all_odds = fetch_odds_data(date)
            # df_avg = process_odds_data(all_odds)

            db_url = 'postgresql://postgres:Kinito23!@localhost/Football_Matches'

            # Create a SQLAlchemy engine
            engine = create_engine(db_url)

            # # Insert the new data from df_avg into the 'odds' table
            # try:
            #     df_avg.to_sql('odds', con=engine, if_exists='append', index=False)
            # except Exception as e:
            #     st.write(f"An error occurred while updating the database: {e}")

            with engine.connect() as connection:
                query = text("SELECT * FROM odds")
                odds_df = pd.read_sql(query, con=connection)

            odds_df = remove_duplicates(odds_df)

            fixtures_df = fixtures_df.merge(
                odds_df[['fixture_id', 'home_team_winning_odd', 'away_team_winning_odd']],
                on='fixture_id',
                how='left'
            )

            fixtures_df['date'] = pd.to_datetime(fixtures_df['date'])
            fixtures_df = add_is_cup_column(fixtures_df)
            fixtures_df = add_recent_fixture_info(fixtures_df, combined_df, num_matches=10)

            # Test DataFrame
            test = fixtures_df
            test = reduce_mem_usage(test, "test")
            test = delete_status_extra_column(test)
            test = remove_rows_all_nan_except_specified(test)
            test = remove_rows_with_more_than_50_percent_nan(test)
            test = delete_history_fixture_id_columns(test)
            test = drop_home_and_away_columns(test)

            test = rename_date_column(test)
            test = rename_history_league_columns(test)
            test = fill_nan_values_away_team_coach_id(test)
            test = fill_nan_values_home_team_coach_id(test)
            test = replace_value_with_row_index(test)
            test = replace_value_with_row_index_multiplied(test)
            test = fill_nan_values_cup(test)
            test = to_date(test)

            test = away_team_history_match_date(test)
            test = home_team_history_match_date(test)
            test = home_team_history_opponent_goal(test)
            test = home_team_history_goal(test)
            test = away_team_history_goal(test)
            test = away_team_history_opponent_goal(test)
            test = home_team_history_is_play_home(test)
            test = away_team_history_is_play_home(test)

            test = home_team_history_winning_odd(test)
            test = home_team_history_opponent_winning_odd(test)
            test = away_team_history_winning_odd(test)
            test = away_team_history_opponent_winning_odd(test)
            test = fill_na_home_team_history_league_id_with_most_common(test)
            test = fill_na_away_team_history_league_id_with_most_common(test)
            test = delete_rows_with_any_nan(test)

            test = check_home_team_coach_change(test)
            test = check_away_team_coach_change(test)
            test = count_matches_with_same_coach_away(test)
            test = count_matches_with_same_coach_home(test)
            test = add_days_between_matches(test)
            test = add_goal_difference_columns(test)
            test = points_per_match(test)

            test = mean_average_home_points(test)
            test = mean_average_away_points(test)
            test = mean_average_home_goal_difference(test)
            test = mean_average_away_goal_difference(test)
            test = std_dev_home_goal_difference(test)
            test = std_dev_away_goal_difference(test)

            test = home_team_winning_losing_draw_ratios(test)
            test = away_team_winning_losing_draw_ratios(test)
            test = calculate_home_team_goal_metrics_at_home(test)
            test = calculate_away_team_goal_metrics_at_away(test)
            test = calculate_mean_average_home_team_goals_scored(test)
            test = calculate_mean_average_home_team_goals_conceded(test)
            test = calculate_mean_average_home_team_goals_difference(test)

            test = calculate_mean_average_away_team_goals_scored(test)
            test = calculate_mean_average_away_team_goals_conceded(test)
            test = calculate_mean_average_away_team_goals_difference(test)
            test = calculate_mean_average_home_team_points(test)
            test = calculate_mean_average_away_team_points(test)
            test = calculate_mean_average_home_team_points_std(test)
            test = calculate_mean_average_away_team_points_std(test)

            test = calculate_league_average_goal(test)
            test = calculate_team_strengths_and_expected_goals(test)
            test = calculate_coach_average_goal_scored_and_conceded(test)
            home_df, away_df = create_new_dfs(test)
            test = merge_home_df_to_original_df(test, home_df)
            test = merge_away_df_to_original_df(test, away_df)

            test = calculate_coach_strengths_and_expected_goals_by_league(test)
            test = home_team_difference_winning_odd_history_columns(test)
            test = away_team_difference_winning_odd_history_columns(test)
            test = calculate_home_team_winning_odd_difference_at_home(test)
            test = calculate_away_team_winning_odd_difference_at_away(test)
            test = calculate_home_team_winning_odd_statistics(test)
            test = calculate_away_team_winning_odd_statistics(test)
            test = calculate_elo_ratings(test)
            test = fill_na_and_inf_with_zero(test)
            test = reduce_mem_usage(test, "test")

            st.write("Please proceed to Final Predictions Page to receive the matches predictions")

            save_fixtures_to_excel(test, 'TEST_TEMP.xlsx')

            end_time = time.time()  # End the timer
            execution_time = end_time - start_time  # Calculate the execution time

            st.write(f"Execution time: {execution_time:.2f} seconds")  # Print the execution time
            st.dataframe(test)
        else:
            st.warning("No fixtures data found for the selected dates.")


