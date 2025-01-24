# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# scaling and train test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder

# creating a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras import models, layers, losses, optimizers, metrics
import os


# Wrangling function
def wrangle(filename):
    # Reading the dataset
    df = pd.read_csv(filename)

    # Convert 'Price' to numeric, coercing errors to NaN
    df["Price"] = pd.to_numeric(df["Price"], errors='coerce')

    # Drop rows with NaN values in 'Price'
    df = df.dropna(subset=["Price"])

    # Split the 'Room' column into separate living room and bedroom info
    df["Room"] = df["Room"].str.split("+", expand=True)[0]  # Keeping only the bedroom count

    # Clean numerical columns and convert to float
    df["Area"] = df["Area"].replace(r'[^\d.]', '', regex=True).astype(float)
    df["Age"] = df["Age"].replace(r'[^\d.]', '', regex=True).astype(float)
    df["Floor"] = df["Floor"].replace(r'[^\d.]', '', regex=True).astype(float)
    df["Room"] = df["Room"].replace(r'[^\d.]', '', regex=True).astype(float)

    # Drop outliers using quantiles for 'Area', 'Price', and 'Age'
    low, high = df["Area"].quantile([0.1, 0.9])
    mask_area = df["Area"].between(low, high)

    low, high = df["Price"].quantile([0.1, 0.9])
    mask_price = df["Price"].between(low, high)

    low, high = df["Age"].quantile([0.1, 0.9])
    mask_age = df["Age"].between(low, high)

    # Keep rows that are not outliers in all three columns
    df = df[mask_area & mask_price & mask_age]

    # Return cleaned DataFrame
    return df


# Apply wrangling function to your dataset
df = wrangle("C:/Users/sila/Desktop/konut/archive/istanbul_satilik_evler_2023.csv")

# Display DataFrame info
df.info()

# Display the last few rows
df.tail()

# Describe 'Area' and 'Price' columns
print(df[["Area", "Price"]].describe())

# Check for null values
print(df.isnull().sum())
