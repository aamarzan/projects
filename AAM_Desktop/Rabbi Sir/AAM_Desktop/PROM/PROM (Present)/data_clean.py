import pandas as pd
import numpy as np
import os

# --- Define File Paths ---
# Set the working directory to your specified path.
# Using raw string (r"...") to handle backslashes correctly in Windows paths.
working_directory = r"C:\Users\User\OneDrive\2. Personal\AAM\Desktop\PROM\PROM (Present)"
input_file_path = os.path.join(working_directory, "PROM (corrected data).xlsx - Corrected Data.csv")
output_file_path = os.path.join(working_directory, "cleaned_prom_data.csv")


# --- Load the Dataset ---
try:
    df = pd.read_csv(input_file_path)
    print(f"✅ Successfully loaded the dataset from: {input_file_path}")
except FileNotFoundError:
    print(f"❌ Error: The file was not found at the specified path: {input_file_path}")
    print("Please ensure the file name and directory are correct.")
    exit()

print("\n--- Initial Data Overview ---")
print(f"Original number of rows: {df.shape[0]}")
print(f"Original number of columns: {df.shape[1]}")


# --- Handle Missing Values ---
print("\n--- Missing Value Analysis ---")

# Calculate the percentage of missing values for each column
missing_percentage = df.isnull().sum() / len(df) * 100

# Identify columns with more than 40% missing values
cols_to_drop = missing_percentage[missing_percentage > 40].index.tolist()
print(f"Found {len(cols_to_drop)} columns with more than 40% missing values. These will be removed.")

# Remove the identified columns
df_cleaned = df.drop(columns=cols_to_drop)
print(f"Number of columns after removal: {df_cleaned.shape[1]}")


# Impute remaining missing values
print("\n--- Imputing Remaining Missing Values ---")
for col in df_cleaned.columns:
    if df_cleaned[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            median_val = df_cleaned[col].median()
            df_cleaned[col].fillna(median_val, inplace=True)
        else:
            mode_val = df_cleaned[col].mode()[0]
            df_cleaned[col].fillna(mode_val, inplace=True)
print("Imputation complete. Numeric columns filled with median, categorical columns with mode.")


# --- Save the Cleaned Dataset ---
try:
    df_cleaned.to_csv(output_file_path, index=False)
    print("\n--- Success! ---")
    print(f"✅ The cleaned dataset has been successfully saved to: {output_file_path}")
except Exception as e:
    print(f"\n❌ An error occurred while saving the file: {e}")