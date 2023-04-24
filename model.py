import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# GUI
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt
# Set custom font
font = QFont()
font.setPointSize(12)
font.setBold(True)

def downloadData():
    # Set the URL for the desired data
    url = 'https://www.basketball-reference.com'

    # Set the parameters for the desired data
    seasons = range(1980, 2023)  # Seasons to scrape data for

    # Create an empty list to hold the data
    data = []

    # Loop through the seasons and scrape the game data
    for season in seasons:
        print('Scraping season {}'.format(season))
        # Build the URL for the desired data
        season_url = url + '/leagues/NBA_{}_games.html'.format(season)

        # Send a GET request to the URL
        page = requests.get(season_url)

        # Use BeautifulSoup to parse the HTML content of the page
        soup = BeautifulSoup(page.content, 'html.parser')

        # Find the table containing the game data and extract the rows
        table = soup.find('table', {'id': 'schedule'})
        if table is None:
            continue  # Skip if the table doesn't exist for this season
        if table.tbody is not None:
            rows = table.tbody.find_all('tr')
        else:
            rows = table.find_all('tr')[1:]  # Skip the header row if there is no tbody

        # Loop through the rows and extract the data
        for row in rows:
            # Extract the date, teams, and scores
            date = row.th.get_text()
            teams = [td.a.get_text() for td in row.find_all('td', {'data-stat': 'visitor_team_name'})]
            teams.extend([td.a.get_text() for td in row.find_all('td', {'data-stat': 'home_team_name'})])
            scores = [td.get_text() for td in row.find_all('td', {'data-stat': 'visitor_pts'})]
            scores.extend([td.get_text() for td in row.find_all('td', {'data-stat': 'home_pts'})])
            location = row.find('td', {'data-stat': 'game_location'})
            if location is None:
                location = ''
            else:
                location = location.get_text()
            
            # Combine the data into a dictionary and append to the list
            game_data = {'season': season, 'date': date, 'team_1': teams[0], 'team_2': teams[1], 
                        'score_1': scores[0], 'score_2': scores[1], 'location': location}
            data.append(game_data)

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(data)
    
    # save to csv
    df.to_csv('data/nba_scores.csv', index=False)
    
    return df

# df = downloadData()
df = pd.read_csv('data/nba_scores.csv')

# Print the first 5 rows of the DataFrame
# print(df)

# Create a list of all unique team names
team_names = list(set(df['team_1']).union(set(df['team_2']))) + ['Unknown Team']

# Create one-hot encoded features for each team
for team in team_names:
    df[team] = np.where((df['team_1'] == team) | (df['team_2'] == team), 1, 0)

# Create a column for unknown teams
df['Unknown Team'] = np.where(df[team_names].sum(axis=1) == 0, 1, 0)

# Define the target variable as a binary column indicating which team won
df['team_1_win'] = np.where(df['score_1'] > df['score_2'], 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[team_names], df['team_1_win'], test_size=0.2)

# Train a logistic regression model on the training data
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Use the model to make predictions on the testing data
y_pred = lr.predict_proba(X_test)

# Print the predicted probabilities for the first 10 testing examples
# print(y_pred[:10])

def predict_winner(team_1, team_2):
    # Create a DataFrame with one row containing the one-hot encoded features for the two teams
    team_data = pd.DataFrame(columns=team_names)
    team_data.loc[0] = [1 if team == team_1 else 0 for team in team_names]
    team_data.loc[1] = [1 if team == team_2 else 0 for team in team_names]

    # Use the global logistic regression model to make predictions on the team data
    y_pred = lr.predict_proba(team_data)

    # Return the predicted probabilities for each team
    return {team_1: y_pred[0][1], team_2: y_pred[1][1]}

def print_winner_probs(team_1, team_2):
    # Predict the winner using the predict_winner function
    winner_probs = predict_winner(team_1, team_2)

    # Print the probabilities
    print('{}: {:.2f}%'.format(team_1, winner_probs[team_1] * 100))
    print('{}: {:.2f}%'.format(team_2, winner_probs[team_2] * 100))

    # Print the predicted winner
    if winner_probs[team_1] > winner_probs[team_2]:
        print('Predicted winner: {}'.format(team_1))
    else:
        print('Predicted winner: {}'.format(team_2))
        
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NBA Predictions')
        self.team1_label = QLabel('Team 1:', self)
        self.team1_label.setFont(font)
        self.team2_label = QLabel('Team 2:', self)
        self.team2_label.setFont(font)
        self.team1_combo = QComboBox(self)
        self.team2_combo = QComboBox(self)
        self.predict_button = QPushButton('Predict', self)
        self.predict_button.setFont(font)
        self.prob1_label = QLabel('Prob1:', self)
        self.prob1_label.setFont(font)
        self.prob2_label = QLabel('Prob2:', self)
        self.prob2_label.setFont(font)
        self.winner_label = QLabel('Winner:', self)
        self.winner_label.setFont(font)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.team1_label)
        self.layout.addWidget(self.team1_combo)
        self.layout.addWidget(self.team2_label)
        self.layout.addWidget(self.team2_combo)
        self.layout.addWidget(self.predict_button)
        self.layout.addWidget(self.prob1_label)
        self.layout.addWidget(self.prob2_label)
        self.layout.addWidget(self.winner_label)
        # sort team names alphabetically
        self.team1_combo.addItems(sorted(team_names))
        self.team2_combo.addItems(sorted(team_names))
        self.predict_button.clicked.connect(self.predict_winner)
        self.setStyleSheet("""
            QLabel {
                color: white;
            }
            QComboBox {
                background-color: #555;
                color: white;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 5px;
                border-radius: 5px;
            }
        """)

    def predict_winner(self):
        team_1 = self.team1_combo.currentText()
        team_2 = self.team2_combo.currentText()
        winner_probs = predict_winner(team_1, team_2)
        prob1 = winner_probs[team_1] * 100
        prob2 = winner_probs[team_2] * 100
        self.prob1_label.setText('{}: {:.2f}%'.format(team_1, prob1))
        self.prob2_label.setText('{}: {:.2f}%'.format(team_2, prob2))
        if prob1 > prob2:
            self.winner_label.setText('Winner: {}'.format(team_1))
            self.winner_label.setStyleSheet("color: green;")
        else:
            self.winner_label.setText('Winner: {}'.format(team_2))
            self.winner_label.setStyleSheet("color: green;")

# Set app style
app = QApplication(sys.argv)
app.setStyle('Fusion')
palette = QPalette()
palette.setColor(QPalette.Window, QColor(30, 30, 30))
palette.setColor(QPalette.WindowText, Qt.white)
palette.setColor(QPalette.HighlightedText, QColor(200, 200, 200))
app.setPalette(palette)

# Set window style
window = MainWindow()
window.setFixedSize(500, 300)
window.setStyleSheet("""
    background-color: #333;
""")
window.show()

sys.exit(app.exec_())