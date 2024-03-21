import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the dataset without specifying column names
df = pd.read_csv('car evaluation_with.csv', header=None)

# Extract features and target variable
X = df.iloc[:, :-1]  # All columns except the last one as features
y = df.iloc[:, -1]   # Last column as target variable

# Use LabelEncoder to convert categorical target variable into numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# One-hot encode the categorical features
one_hot_encoder = OneHotEncoder()
X_encoded = one_hot_encoder.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Fit the model
knn.fit(X_train, y_train)

# Save the trained model
joblib.dump(knn, 'knn_model.pkl')
