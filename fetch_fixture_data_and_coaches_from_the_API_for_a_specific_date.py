import streamlit as st
import requests
import pandas as pd
import time
from sqlalchemy import create_engine

# Function to fetch fixture data from the API for a specific date
def fetch_fixtures_by_date(date, url, headers):
    querystring = {"date": date}
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json().get("response", [])
    else:
        st.error(f"Failed to retrieve data for date {date}: {response.status_code}")
        return []

# Function to fetch lineup data for a specific fixture
def fetch_lineup_data(fixture_id, url, headers):
    querystring = {"fixture": fixture_id}
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json().get("response", [])
    else:
        st.error(f"Failed to retrieve lineup data for fixture {fixture_id}: {response.status_code}")
        return []

# API endpoints and headers
url_fixtures = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
url_lineups = "https://api-football-v1.p.rapidapi.com/v3/fixtures/lineups"
headers = {
    "X-RapidAPI-Key": "XXXXXXXXX", #Put API Key
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

# Streamlit UI
def run():
    st.title("Fetch Fixture Data and Coaches from the API for a Specific Date")

    # Date input
    date = st.date_input("Pick a date", pd.to_datetime("today"))
    date_str = date.strftime("%Y-%m-%d")

    if st.button("Fetch Data"):
        all_fixtures = []

        # Fetch fixture data for the selected date
        fixture_data = fetch_fixtures_by_date(date_str, url_fixtures, headers)
        if fixture_data:
            all_fixtures.extend(fixture_data)

        # Convert all_fixtures to DataFrame
        fixtures_df = pd.DataFrame(all_fixtures)

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
            fixtures_df.rename(columns={'id': 'league_id', 'name': 'league_name'}, inplace=True)
            league_columns_to_drop = ['logo', 'flag']
            fixtures_df.drop(columns=league_columns_to_drop, inplace=True, errors='ignore')

        # Flatten the 'teams' column
        if 'teams' in fixtures_df.columns:
            teams_df = pd.json_normalize(fixtures_df['teams'])
            fixtures_df = pd.concat([fixtures_df.drop(columns=['teams']), teams_df], axis=1)
            fixtures_df.rename(columns={
                'home.id': 'home_team_id', 'home.name': 'home_team_name',
                'away.id': 'away_team_id', 'away.name': 'away_team_name'
            }, inplace=True)
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

        # Filter fixtures where the status is 'Match Finished'
        def filter_fixtures_match_finished(fixtures_df):
            if 'status.long' in fixtures_df.columns:
                fixtures_df = fixtures_df[fixtures_df['status.long'] == 'Match Finished']
            return fixtures_df

        fixtures_df = filter_fixtures_match_finished(fixtures_df)

        # Check if 'fixture_id' column exists before proceeding
        if 'fixture_id' in fixtures_df.columns:
            fixture_ids = fixtures_df['fixture_id'].unique()
            total_fixtures = len(fixture_ids)
            st.write(f"Total fixtures to process: {total_fixtures}")
        else:
            st.error("Column 'fixture_id' is missing in the fixtures_df DataFrame.")
            return  # Exit the function if 'fixture_id' is not present

        # Fetch lineup data
        batch_size = 300
        pause_time = 60
        all_lineups = []
        for i in range(0, total_fixtures, batch_size):
            batch_fixture_ids = fixture_ids[i:i + batch_size]

            for fixture_id in batch_fixture_ids:
                lineup_data = fetch_lineup_data(fixture_id, url_lineups, headers)
                if lineup_data:
                    for lineup in lineup_data:
                        lineup['fixture_id'] = fixture_id
                    all_lineups.extend(lineup_data)

            if i + batch_size < total_fixtures:
                st.write(
                    f"Processed {i + batch_size} of {total_fixtures} fixtures. Pausing for {pause_time} seconds.")
                time.sleep(pause_time)

        # Convert all_lineups to DataFrame
        lineups_df = pd.DataFrame(all_lineups)

        # Flatten the 'coach' column
        if 'coach' in lineups_df.columns:
            lineups_df = pd.concat([lineups_df.drop(['coach'], axis=1), lineups_df['coach'].apply(pd.Series)], axis=1)

        # Drop specified columns
        columns_to_drop = ['team', 'formation', 'startXI', 'substitutes', 'name', 'photo']
        lineups_df.drop(columns=columns_to_drop, inplace=True)

        # Group by fixture_id and get coach ids
        def get_coach_ids(x):
            if len(x) >= 2:
                return pd.Series([x.iloc[0], x.iloc[1]])
            elif len(x) == 1:
                return pd.Series([x.iloc[0], None])  # Only one coach, set second as None
            else:
                return pd.Series([None, None])  # No coaches found

        coaches_df = lineups_df.groupby('fixture_id')['id'].apply(get_coach_ids).unstack().reset_index()
        coaches_df.columns = ['fixture_id', 'home_team_coach_id', 'away_team_coach_id']

        # Merge coaches_df with fixtures_df
        merged_df = pd.merge(fixtures_df, coaches_df, on='fixture_id', how='left')

        if "status.extra" in merged_df.columns:
            merged_df = merged_df.drop(columns=["status.extra"])

        # Rename columns for database compatibility
        merged_df = merged_df.rename(columns={'date': 'match_date', 'status.long': 'status'})

        # Define your database connection
        db_url = 'postgresql://postgres:Kinito23!@localhost/Football_Matches'
        engine = create_engine(db_url)

        # Save to database
        merged_df.to_sql('all_fixtures', con=engine, if_exists='append', index=False)

        st.write("Data successfully appended to the 'all_fixtures' table.")
        st.dataframe(merged_df)

if __name__ == "__main__":
    run()
