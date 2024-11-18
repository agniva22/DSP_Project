import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier


# Load datasets for four classes
real = pd.read_csv('./Feature_data/real.csv')
easy = pd.read_csv('./Feature_data/easy.csv')
hard = pd.read_csv('./Feature_data/hard.csv')
mid = pd.read_csv('./Feature_data/mid.csv')  

real['label'] = 0
easy['label'] = 1
hard['label'] = 2
mid['label'] = 3

# Combine the datasets
data = np.vstack((real, easy, hard, mid))
data = list(data) 
data = pd.DataFrame(data)
print("Data loaded successfully.")

# Features and labels
X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

# Preprocess X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize LazyClassifier
lazy_classifier = LazyClassifier()

# Fit and evaluate models
models = lazy_classifier.fit(X_train, X_test, y_train, y_test)
print(models)
