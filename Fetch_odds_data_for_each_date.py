import streamlit as st
import requests
import pandas as pd
import time
from sqlalchemy import create_engine

def fetch_odds_data(date):
    # API endpoint for odds
    url_odds = "https://api-football-v1.p.rapidapi.com/v3/odds"
    headers = {
        "x-rapidapi-key": "XXXXXXXXX", #Put API Key
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }

    all_odds = []
    page = 1

    # Fetch odds data for the selected date
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
                # st.write(f"No more odds data found for date {date} on page {page}")
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

    # Display the new DataFrame
    df_avg

    return df_avg

def run():
    st.title("Fetch Odds Data")

    # Date selection
    date = st.date_input("Pick a date", pd.to_datetime("today"))
    date_str = date.strftime("%Y-%m-%d")

    if st.button("Fetch Odds Data"):
        all_odds = fetch_odds_data(date_str)
        if all_odds:
            df_avg = process_odds_data(all_odds)

            # Define your database connection (adjust the URL to match your database)
            db_url = 'postgresql://postgres:Kinito23!@localhost/Football_Matches'

            # Create a SQLAlchemy engine
            engine = create_engine(db_url)

            # Insert the new data from df_avg into the 'odds' table
            try:
                df_avg.to_sql('odds', con=engine, if_exists='append', index=False)
                st.write("Odds data fetched and appended to the database successfully.")
                st.dataframe(df_avg)
            except Exception as e:
                st.write(f"An error occurred while updating the database: {e}")
        else:
            st.write("No odds data found for the selected date.")


