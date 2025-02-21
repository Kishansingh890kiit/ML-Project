# ML-Project
Box office revenue prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('boxoffice.csv', encoding='latin-1')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe().T)

# We will be predicting domestic_revenue
# No need to remove world_revenue and opening_revenue

# Check for null values
print(df.isnull().sum() * 100 / df.shape[0])

# Handle null values
for col in ['MPAA', 'genres', 'budget']:
    df[col] = df[col].fillna(df[col].mode()[0])

df.dropna(inplace=True)

# Clean and convert numeric columns
numeric_cols = ['domestic_revenue', 'world_revenue', 'opening_revenue', 'opening_theaters', 'budget', 'release_days']
for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')

# EDA
plt.figure(figsize=(10, 5))
sb.countplot(df['MPAA'])
plt.show()

print(df.groupby('MPAA')['domestic_revenue'].mean())

features = ['domestic_revenue', 'world_revenue', 'opening_revenue', 'opening_theaters', 'budget', 'release_days']
plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()

# Log transform skewed features
for col in features:
    df[col] = df[col].apply(lambda x: np.log10(x) if x > 0 else 0)

# Create features from genres
vectorizer = CountVectorizer()
genre_features = vectorizer.fit_transform(df['genres']).toarray()
genre_names = vectorizer.get_feature_names_out()
for i, name in enumerate(genre_names):
    df[name] = genre_features[:, i]

df.drop('genres', axis=1, inplace=True)

# Remove rare genres
for col in df.columns[df.columns.get_loc('action'):]:
    if (df[col] == 0).mean() > 0.95:
        df.drop(col, axis=1, inplace=True)

# Encode categorical variables
for col in ['distributor', 'MPAA']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Correlation heatmap
plt.figure(figsize=(12, 10))
sb.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

# Prepare data for modeling
features = df.drop(['title', 'domestic_revenue'], axis=1)
target = df['domestic_revenue'].values

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)
print(X_train.shape, X_val.shape)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train and evaluate model
from sklearn.metrics import mean_absolute_error as mae
model = XGBRegressor()
model.fit(X_train, Y_train)

train_preds = model.predict(X_train)
print('Training Error : ', mae(Y_train, train_preds))

val_preds = model.predict(X_val)
print('Validation Error : ', mae(Y_val, val_preds))
