import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_bet_win_vs_bet_lose(df):
    # Calculate the percentage of each outcome for 'Bet Win 1 Prediction'
    bet_win_1_and_2_count = df['Bet Win 1 Prediction'].value_counts(normalize=True).get('Bet win', 0) * 100
    bet_lose_1_and_2_count = df['Bet Win 1 Prediction'].value_counts(normalize=True).get('Bet lose', 0) * 100

    # Data for plotting
    labels = ['Bet Win', 'Bet Lose']
    counts = [bet_win_1_and_2_count, bet_lose_1_and_2_count]
    colors = ['tab:green', 'tab:red']

    # Create the bar chart with a larger figure size
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the size here (width, height)
    bars = ax.bar(labels, counts, color=colors)

    # Set labels and title
    ax.set_ylabel('Percentage (%)')
    ax.set_title("Percentage of 'Bet Win' and 'Bet Lose' with model's First Prediction")

    # Add percentage labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12)

    # Display the plot in Streamlit
    st.pyplot(fig)


def plot_bet_win_vs_bet_lose_1_and_2(df):
    # Calculate the percentage of each outcome for 'Bet Win 1 and 2 Prediction'
    bet_win_1_and_2_count = df['Bet Win 1 and 2 Prediction'].value_counts(normalize=True).get('Bet win', 0) * 100
    bet_lose_1_and_2_count = df['Bet Win 1 and 2 Prediction'].value_counts(normalize=True).get('Bet lose', 0) * 100

    # Data for plotting
    labels = ['Bet Win', 'Bet Lose']
    counts = [bet_win_1_and_2_count, bet_lose_1_and_2_count]
    colors = ['tab:green', 'tab:red']

    # Create the bar chart with a larger figure size
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the size here (width, height)
    bars = ax.bar(labels, counts, color=colors)

    # Set labels and title
    ax.set_ylabel('Percentage (%)')
    ax.set_title("Percentage of 'Bet Win' and 'Bet Lose' with model's First and Second Prediction")

    # Add percentage labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12)

    # Display the plot in Streamlit
    st.pyplot(fig)


def plot_confusion_matrix(df):
    # Generate confusion matrix
    cm = confusion_matrix(df['Real Result'], df['1 Prediction'], labels=['home', 'away', 'draw'])

    # Convert confusion matrix to DataFrame for better readability
    cm_df = pd.DataFrame(cm, index=['True Home', 'True Away', 'True Draw'],
                         columns=['Pred Home', 'Pred Away', 'Pred Draw'])

    # Calculate percentages for each cell in the confusion matrix
    cm_percentage = cm_df / cm_df.sum().sum() * 100

    # Plot the confusion matrix using imshow (no seaborn)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(cm_percentage, interpolation='nearest', cmap='coolwarm')

    # Add colorbar
    fig.colorbar(cax)

    # Add labels to each cell
    for i in range(cm_percentage.shape[0]):
        for j in range(cm_percentage.shape[1]):
            ax.text(j, i, f'{cm_percentage.iloc[i, j]:.2f}%', ha='center', va='center', fontsize=12)

    # Set labels and title
    ax.set_title("Confusion Matrix (Percentage)", fontsize=16)
    ax.set_ylabel('True Labels', fontsize=12)
    ax.set_xlabel('Predicted Labels', fontsize=12)

    # Set tick labels
    ax.set_xticks(range(len(cm_percentage.columns)))
    ax.set_xticklabels(cm_percentage.columns, fontsize=12)
    ax.set_yticks(range(len(cm_percentage.index)))
    ax.set_yticklabels(cm_percentage.index, fontsize=12)

    # Display the plot in Streamlit
    st.pyplot(fig)


def analyze_and_plot_bet_win(df):
    # Group by 'country league_name' and calculate appearances
    group_df = df.groupby('country league_name').size().reset_index(name='appearances')

    # Filter leagues with at least 30 appearances
    filtered_leagues = group_df[group_df['appearances'] >= 30]
    filtered_df = df[df['country league_name'].isin(filtered_leagues['country league_name'])]

    # Calculate percentages of 'Bet win' in both columns
    percentages = filtered_df.groupby('country league_name').apply(
        lambda x: pd.Series({
            'Bet Win 1 Prediction %': (x['Bet Win 1 Prediction'] == 'Bet win').mean() * 100,
            'Bet Win 1 and 2 Prediction %': (x['Bet Win 1 and 2 Prediction'] == 'Bet win').mean() * 100
        })
    ).reset_index()

    # Sort and pick top 20 and lowest 20 for both columns
    top_20_bet_win_1 = percentages.nlargest(20, 'Bet Win 1 Prediction %')
    lowest_20_bet_win_1 = percentages.nsmallest(20, 'Bet Win 1 Prediction %')
    top_20_bet_win_1_and_2 = percentages.nlargest(20, 'Bet Win 1 and 2 Prediction %')
    lowest_20_bet_win_1_and_2 = percentages.nsmallest(20, 'Bet Win 1 and 2 Prediction %')

    # Helper function to plot horizontal bar charts
    def plot_bar_chart(data, column, title, color):
        fig, ax = plt.subplots(figsize=(16, 14))
        bars = ax.barh(data['country league_name'], data[column], color=color, alpha=0.7)
        ax.set_xlabel('Percentage (%)', fontsize=12)
        ax.set_ylabel('League', fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.invert_yaxis()  # Reverse the order for better readability

        # Add percentage labels to the bars
        for bar, percentage in zip(bars, data[column]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f'{percentage:.1f}%', va='center', fontsize=10)

        st.pyplot(fig)

    # Plot top 20 and lowest 20 for 'Bet Win 1 Prediction %'
    plot_bar_chart(top_20_bet_win_1, 'Bet Win 1 Prediction %', 'Top 20 Leagues: Bet Winning Percentage with First Prediction', 'blue')
    plot_bar_chart(lowest_20_bet_win_1, 'Bet Win 1 Prediction %', 'Lowest 20 Leagues: Bet Winning Percentage with First Prediction', 'red')

    # Plot top 20 and lowest 20 for 'Bet Win 1 and 2 Prediction %'
    plot_bar_chart(top_20_bet_win_1_and_2, 'Bet Win 1 and 2 Prediction %', 'Top 20 Leagues: Bet Winning Percentage with First and Second Prediction', 'green')
    plot_bar_chart(lowest_20_bet_win_1_and_2, 'Bet Win 1 and 2 Prediction %', 'Lowest 20 Leagues: Bet Winning Percentage with First and Second Prediction', 'orange')




def run():
    st.title("Metrics")

    # Read the Excel file
    df = pd.read_excel('TOTAL_PREDICTIONS.xlsx')

    # Generate the plots
    plot_bet_win_vs_bet_lose(df)
    plot_bet_win_vs_bet_lose_1_and_2(df)
    plot_confusion_matrix(df)
    analyze_and_plot_bet_win(df)




