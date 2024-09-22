# Import necessary libraries
import numpy as np  # For numerical operations, such as arrays and matrix handling
import pandas as pd  # For handling and manipulating data structures (like DataFrames)
from sklearn.model_selection import train_test_split  # For splitting the data into training and test sets
from sklearn.metrics import mean_absolute_error, r2_score  # For evaluating the model's performance
from catboost import CatBoostRegressor  # The CatBoost machine learning model for regression
import warnings  # To handle warnings that could interrupt execution

# Ignore warnings to keep the output clean
warnings.filterwarnings("ignore")

# Outlier handling function
def outlier_threshold(dataframe, variable):
    Q1 = dataframe[variable].quantile(0.01)  # First percentile (1%)
    Q3 = dataframe[variable].quantile(0.99)  # 99th percentile
    IQR = Q3 - Q1  # Interquartile range (difference between Q3 and Q1)
    up_limit = Q3 + 1.5 * IQR  # Upper limit beyond which values are considered outliers
    low_limit = Q1 - 1.5 * IQR  # Lower limit for outliers
    return low_limit, up_limit  # Return both the lower and upper limits

# Function to replace outliers with threshold values
def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)  # Get outlier limits
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit  # Replace values below the lower limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit  # Replace values above the upper limit

# Data preprocessing function
def pre_process(df):
    # Convert categorical variable 'imbalance_buy_sell_flag' into dummy/indicator variables
    df = pd.get_dummies(df, columns=['imbalance_buy_sell_flag'], prefix='imbalance_flag', drop_first=True)
    # Create a new feature: the imbalance ratio (the ratio of imbalance size to matched size)
    df['imbalance_ratio'] = df['imbalance_size'] / df['matched_size']
    return df  # Return the modified DataFrame

# Feature engineering function
def engineered_features(df):
    # Lists of columns related to prices and sizes
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    
    # Add a new feature: total volume, calculated as the sum of bid size and ask size
    df["volume"] = df.eval("ask_size + bid_size")
    # Add another feature: the mid-price (the average of ask and bid prices)
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    
    # More feature engineering based on size ratios and differences
    df["bid_ask_size_ratio"] = df["bid_size"] / df["ask_size"]
    df["imbalance_bid_size_ratio"] = df["imbalance_size"] / df["bid_size"]
    df["imbalance_ask_size_ratio"] = df["imbalance_size"] / df["ask_size"]
    df["matched_size_ratio"] = df["matched_size"] / (df["bid_size"] + df["ask_size"])
    df["ref_wap_difference"] = df["reference_price"] - df["wap"]
    df["bid_ask_spread"] = df["ask_price"] - df["bid_price"]
    
    # Replace any infinite values resulting from divisions by zero with 0
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df  # Return the DataFrame with new features

# Load the dataset
df = pd.read_csv("data/train.csv")  # Load the CSV file containing the training data into a pandas DataFrame

# Drop any rows where the 'ask_price' is missing
df = df.dropna(subset=["ask_price"], axis=0)

# Handle outliers in the 'ask_price' column
replace_with_threshold(df, "ask_price")

# Fill any other missing values in the dataset with 0
df.fillna(0, inplace=True)

# Apply feature engineering to the dataset
df = engineered_features(df)

# Prepare input features (X) and the target variable (y)
X = df.drop(["target", "row_id"], axis=1)  # X is all columns except 'target' and 'row_id'
y = df["target"]  # y is the target variable we want to predict

# Split the data into training and testing sets (67% train, 33% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44)

# Initialize the CatBoost model for regression
catboost_model = CatBoostRegressor(verbose=0)  # Verbose=0 means the model will run silently

# Train the model on the training data
catboost_model.fit(X_train, y_train)

# Save the trained model
catboost_model.save_model("catboost_model.cbm")  # Save the model to a file

# Make predictions on the test set
predictions = catboost_model.predict(X_test)

# Evaluate the model using Mean Absolute Error (MAE) and R² score
mae = mean_absolute_error(y_test, predictions)  # MAE measures how far predictions are from actual values
r2 = r2_score(y_test, predictions)  # R² measures how well the model explains variance in the target variable

# Print the evaluation metrics
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')