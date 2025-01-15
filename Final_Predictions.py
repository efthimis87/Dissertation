import streamlit as st
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt




def final_predictions(desired_percentage):


    train = pd.read_csv(
        'TRAIN_TEMP.csv')  # Reading the csv file

    test = pd.read_excel(
        'TEST_TEMP.xlsx')

    test1 = pd.read_excel(
        'TEST_TEMP.xlsx')



    columns_to_drop_train = ['fixture_id', 'timezone', 'match_date', 'status.long', 'league_id', 'league_name',
                             'country', 'season',
                             'round', 'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name',
                             'home_team_coach_id',
                             'away_team_coach_id', 'home_team_history_match_date_1', 'home_team_history_match_date_2',
                             'home_team_history_match_date_3', 'home_team_history_match_date_4',
                             'home_team_history_match_date_5',
                             'home_team_history_match_date_6', 'home_team_history_match_date_7',
                             'home_team_history_match_date_8',
                             'home_team_history_match_date_9', 'home_team_history_match_date_10',
                             'home_team_history_coach_1',
                             'home_team_history_coach_2', 'home_team_history_coach_3', 'home_team_history_coach_4',
                             'home_team_history_coach_5', 'home_team_history_coach_6', 'home_team_history_coach_7',
                             'home_team_history_coach_8', 'home_team_history_coach_9', 'home_team_history_coach_10',
                             'home_team_history_league_id_1', 'home_team_history_league_id_2',
                             'home_team_history_league_id_3',
                             'home_team_history_league_id_4', 'home_team_history_league_id_5',
                             'home_team_history_league_id_6',
                             'home_team_history_league_id_7', 'home_team_history_league_id_8',
                             'home_team_history_league_id_9',
                             'home_team_history_league_id_10', 'away_team_history_match_date_1',
                             'away_team_history_match_date_2',
                             'away_team_history_match_date_3', 'away_team_history_match_date_4',
                             'away_team_history_match_date_5',
                             'away_team_history_match_date_6', 'away_team_history_match_date_7',
                             'away_team_history_match_date_8',
                             'away_team_history_match_date_9', 'away_team_history_match_date_10',
                             'away_team_history_coach_1',
                             'away_team_history_coach_2', 'away_team_history_coach_3', 'away_team_history_coach_4',
                             'away_team_history_coach_5',
                             'away_team_history_coach_6', 'away_team_history_coach_7', 'away_team_history_coach_8',
                             'away_team_history_coach_9',
                             'away_team_history_coach_10', 'away_team_history_league_id_2',
                             'away_team_history_league_id_3', 'away_team_history_league_id_4',
                             'away_team_history_league_id_5',
                             'away_team_history_league_id_6', 'away_team_history_league_id_7',
                             'away_team_history_league_id_8',
                             'away_team_history_league_id_9',
                             'away_team_history_league_id_10']  # Add all column names you want to drop to this list

    columns_to_drop_test = ['timezone', 'match_date', 'status.long', 'league_id', 'league_name', 'country', 'season',
                            'round', 'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name',
                            'home_team_coach_id',
                            'away_team_coach_id', 'home_team_history_match_date_1', 'home_team_history_match_date_2',
                            'home_team_history_match_date_3', 'home_team_history_match_date_4',
                            'home_team_history_match_date_5',
                            'home_team_history_match_date_6', 'home_team_history_match_date_7',
                            'home_team_history_match_date_8',
                            'home_team_history_match_date_9', 'home_team_history_match_date_10',
                            'home_team_history_coach_1',
                            'home_team_history_coach_2', 'home_team_history_coach_3', 'home_team_history_coach_4',
                            'home_team_history_coach_5', 'home_team_history_coach_6', 'home_team_history_coach_7',
                            'home_team_history_coach_8', 'home_team_history_coach_9', 'home_team_history_coach_10',
                            'home_team_history_league_id_1', 'home_team_history_league_id_2',
                            'home_team_history_league_id_3',
                            'home_team_history_league_id_4', 'home_team_history_league_id_5',
                            'home_team_history_league_id_6',
                            'home_team_history_league_id_7', 'home_team_history_league_id_8',
                            'home_team_history_league_id_9',
                            'home_team_history_league_id_10', 'away_team_history_match_date_1',
                            'away_team_history_match_date_2',
                            'away_team_history_match_date_3', 'away_team_history_match_date_4',
                            'away_team_history_match_date_5',
                            'away_team_history_match_date_6', 'away_team_history_match_date_7',
                            'away_team_history_match_date_8',
                            'away_team_history_match_date_9', 'away_team_history_match_date_10',
                            'away_team_history_coach_1',
                            'away_team_history_coach_2', 'away_team_history_coach_3', 'away_team_history_coach_4',
                            'away_team_history_coach_5',
                            'away_team_history_coach_6', 'away_team_history_coach_7', 'away_team_history_coach_8',
                            'away_team_history_coach_9',
                            'away_team_history_coach_10', 'away_team_history_league_id_2',
                            'away_team_history_league_id_3', 'away_team_history_league_id_4',
                            'away_team_history_league_id_5',
                            'away_team_history_league_id_6', 'away_team_history_league_id_7',
                            'away_team_history_league_id_8',
                            'away_team_history_league_id_9',
                            'away_team_history_league_id_10']  # Add all column names you want to drop to this list

    train.drop(columns=columns_to_drop_train, inplace=True)
    test.drop(columns=columns_to_drop_test, inplace=True)



    # Delete 'target' column from test DataFrame if it exists
    if 'target' in test.columns:
        test.drop('target', axis=1, inplace=True)

    # Ensure the features in the test set match the features in the train set
    common_features = set(train.columns).intersection(set(test.columns))
    X_train = train[list(common_features)]
    X_test = test[list(common_features)]

    # Extract the target variable from the train DataFrame
    y_train = train['target']

    # Encode the categorical labels into numerical values
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Perform feature selection
    selector = SelectKBest(score_func=f_classif, k=50)  # Select top 50 features
    X_train_selected = selector.fit_transform(X_train, y_train_encoded)
    X_test_selected = selector.transform(X_test)



    # Identify and remove non-numeric columns from the training data
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train_numeric = X_train[numeric_cols]

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler to the training data
    scaler.fit(X_train_numeric)

    # Transform the training and testing data
    X_train_scaled = scaler.transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test[numeric_cols])

    # Calculate the percentage of each class in the original dataset
    class_counts = train['target'].value_counts(normalize=True) * 100

    # Define the desired percentages for each class after SMOTE
    # desired_percentage = {
    #     'home': 503.00,
    #     'away': 451.00,
    #     'draw': 601.00
    # }

    # Calculate the number of samples needed for each class after SMOTE
    total_samples = len(train)
    desired_samples = {label_encoder.transform([label])[0]: max(int(total_samples * (percentage / 100)), count) for
                       label, percentage, count in
                       zip(desired_percentage.keys(), desired_percentage.values(), class_counts)}

    # Define SMOTE with adjusted sampling strategy
    smote = SMOTE(sampling_strategy=desired_samples, random_state=42)

    # Apply SMOTE to balance the class distribution
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_encoded)

    # Define parameters for CatBoost
    catboost_params = {
        'iterations': 100,
        'learning_rate': 0.1,
        'random_seed': 42,
        'logging_level': 'Silent'
    }

    # Create the CatBoostClassifier model
    model = CatBoostClassifier(**catboost_params)

    # Train the model
    model.fit(X_train_resampled, y_train_resampled)

    # Make predictions (probabilities) on the test set
    y_pred_prob = model.predict_proba(X_test_scaled)

    # # Create submission DataFrame
    # submission = pd.DataFrame(y_pred_prob,
    #                           columns=['away', 'draw', 'home'])  # Assuming the order of classes is away, draw, home
    # submission['fixture_id'] = test.sort_values(by='fixture_id').reset_index()['fixture_id']
    # submission = submission[['fixture_id', 'home', 'away', 'draw']]  # Reorder columns to match submission format

    # Create test_with_predictions DataFrame with only 'id' and 'target' columns
    test_with_predictions = test[['fixture_id']]
    test_with_predictions['target'] = label_encoder.inverse_transform(np.argmax(y_pred_prob, axis=1))

    # Calculate the second prediction
    y_pred_prob_sorted = np.sort(y_pred_prob, axis=1)[:, ::-1]
    test_with_predictions['second_prediction'] = label_encoder.inverse_transform(np.argsort(y_pred_prob, axis=1)[:, -2])


    # Step 1: Count occurrences of each value
    value_counts = test_with_predictions['target'].value_counts()

    # Step 2: Divide by the total number of entries
    total_counts = test_with_predictions['target'].count()
    percentages = (value_counts / total_counts) * 100

    # Display the percentages
    st.dataframe(percentages)
    #print(percentages)

    # Assuming test_with_predictions and test1 are your DataFrames
    columns_to_merge = ['fixture_id', 'match_date', 'league_name', 'country', 'round', 'home_team_name', 'away_team_name']

    # Perform the left merge on fixture_id
    merged_df = test_with_predictions.merge(test1[columns_to_merge], on='fixture_id', how='left')
    merged_df['match_date'] = pd.to_datetime(merged_df['match_date']).dt.date

    # # Visualize top 10 features
    # feature_scores = selector.scores_
    feature_scores = model.feature_importances_
    top_10_indices = np.argsort(feature_scores)[-10:][::1]
    top_10_features = X_train.columns[top_10_indices]
    top_10_scores = feature_scores[top_10_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(top_10_features, top_10_scores, color='skyblue')
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Top 10 Features")
    st.pyplot(plt)



    return merged_df

def run():
    import time

    st.title("Get Predictions")

    # Allow manual input of desired percentages
    st.subheader("Set Desired Nodes for Predictions")
    desired_percentage = {
        'home': st.number_input("Enter Desired Number of Nodes for 'Home':", min_value=0.0, value=503.00),
        'away': st.number_input("Enter Desired Number of Nodes for 'Away':", min_value=0.0, value=451.00),
        'draw': st.number_input("Enter Desired Number of Nodes for 'Draw':", min_value=0.0, value=601.00)
    }

    st.write("Current Desired Number of Nodes:")
    st.write(desired_percentage)

    st.write("Click the button below to get the Predictions.")

    # Button to trigger prediction generation
    if st.button("Get Predictions"):
        with st.spinner("Getting Predictions..."):
            start_time = time.time()

            # Pass the desired_percentage as an argument to the function
            try:
                merged_df = final_predictions(desired_percentage)
                end_time = time.time()  # End the timer
                execution_time = end_time - start_time  # Calculate the execution time

                st.success(f"Final Predictions generated successfully in {execution_time:.2f} seconds.")

                # Display the resulting DataFrame
                st.dataframe(merged_df)

                # Option to download the result
                output_file = 'Final_Predictions.xlsx'
                merged_df.to_excel(output_file, index=False)
                with open(output_file, "rb") as file:
                    st.download_button(
                        label="Download Predictions",
                        data=file,
                        file_name=output_file,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"An error occurred: {e}")

