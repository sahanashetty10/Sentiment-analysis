# Sentiment-analysis
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_csv('linkedin-reviews1.csv')
data.head()
#Pre-process Data
def preprocess_data(data):
    # Remove package name as it's not relevant
    data = data.drop('package_name', axis=1)
    
    # Convert text to lowercase
    data['Review'] = data['Review'].str.strip().str.lower()
    return data
    data
    # Split into training and testing data
x = data[['Review', 'Date']]  # Use double brackets to select multiple columns
y = data['Rating']
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)
import matplotlib.pyplot as plt

# Count the occurrences of each sentiment category
sentiment_counts = data['Sentiment'].value_counts()

# Plot bar chart
sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red'])

# Add labels and title
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Analysis Results')
plt.xticks(rotation=0)  # Rotate x-axis labels if needed

# Show plot
plt.show()
