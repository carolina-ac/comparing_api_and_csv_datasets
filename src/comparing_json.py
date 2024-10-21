# =============================================================================
# Packages
# =============================================================================
#%%
from config import config
import requests
import pandas as pd  
from io import StringIO
# =============================================================================
# Root Directory
# =============================================================================
root_directory = 'F:/open_git/redcap' # change to your pathway
# =============================================================================
# Function to export data from REDCap
# =============================================================================
def export_redcap_data(api_url, api_token, export_rawOrlabel, export_checkbox_label):
    """
    Function to export data from REDCap based on the configuration of 'exportCheckboxLabel' and 'rawOrLabel'.
    
    Parameters:
    - api_url: str - REDCap API URL.
    - api_token: str - REDCap API authentication token.
    - export_rawOrlabel: str - 'raw' or 'label' for the data configuration.
    - export_checkbox_label: str - 'true' or 'false' for the exportCheckboxLabel configuration.

    Returns:
    - pd.DataFrame - DataFrame with the exported data.
    """
    data = {
        'token': api_token,
        'content': 'record',
        'action': 'export',
        'format': 'json',
        'type': 'flat',
        'csvDelimiter': '',
        'rawOrLabel': 'raw',
        'rawOrLabelHeaders': 'raw',
        'exportCheckboxLabel':'true',
        'exportSurveyFields': 'false',
        'exportDataAccessGroups': 'true',
        'returnFormat': 'json'
    }

    # Make the API request
    r = requests.post(api_url, data=data)
    
    # Check the status of the request
    if r.status_code == 200:
        # Convert the JSON response to a DataFrame
        df = pd.read_json(StringIO(r.text))
        return df
    else:
        print(f"Error: {r.status_code}")
        return None
#%%
# =============================================================================
# Export all combinations of data
# =============================================================================
# Export four DataFrames with the combinations of 'raw/label' and 'true/false'
df_raw_true = export_redcap_data(config['api_url'], config['api_token'], 'raw', 'true')
df_raw_false = export_redcap_data(config['api_url'], config['api_token'], 'raw', 'false')
df_label_true = export_redcap_data(config['api_url'], config['api_token'], 'label', 'true')
df_label_false = export_redcap_data(config['api_url'], config['api_token'], 'label', 'false')
#%%
# =============================================================================
# Function to compare DataFrames
# =============================================================================
def compare_json_dataframes(df1, df2, key_column='record_id'):
    """
    Function to compare two DataFrames and return the differences.
    
    Parameters:
    - df1: pd.DataFrame - First DataFrame for comparison.
    - df2: pd.DataFrame - Second DataFrame for comparison.
    - key_column: str - Key column for comparison ('record_id').
    
    Returns:
    - tuple: (differences in values, record_ids with differences)
    """
    # Convert the key column to the same format
    df1[key_column] = df1[key_column].astype(str)
    df2[key_column] = df2[key_column].astype(str)

    # Merge the two DataFrames based on the key column
    merged_df = pd.merge(df1, df2, on=key_column, how='outer', suffixes=('_df1', '_df2'), indicator=True)

    # Find the differences
    df_differences = merged_df[merged_df['_merge'] != 'both']  # Columns with differences
    record_ids_with_differences = df_differences[key_column].unique()

    return df_differences, record_ids_with_differences
#%%
# =============================================================================
# Compare the DataFrames pairwise
# =============================================================================
def compare_all_combinations(df_raw_true, df_raw_false, df_label_true, df_label_false):
    """
    Compares all possible combinations between the exported DataFrames.
    """
    # Define the combinations to be compared
    combinations = [
        (df_raw_true, df_raw_false, "raw_true vs raw_false"),
        (df_raw_true, df_label_true, "raw_true vs label_true"),
        (df_raw_true, df_label_false, "raw_true vs label_false"),
        (df_raw_false, df_label_true, "raw_false vs label_true"),
        (df_raw_false, df_label_false, "raw_false vs label_false"),
        (df_label_true, df_label_false, "label_true vs label_false")
    ]
    
    # Compare each combination and display results
    for df1, df2, description in combinations:
        if df1 is not None and df2 is not None:
            df_differences, record_ids_with_differences = compare_json_dataframes(df1, df2)
            print(f"\nDifferences between {description}:")
            print(df_differences)
            print(f"Record IDs with differences: {record_ids_with_differences}")
        else:
            print(f"Error in one of the exports: {description}")
#%%
# =============================================================================
# Call the function to compare the combinations
# =============================================================================
compare_all_combinations(df_raw_true, df_raw_false, df_label_true, df_label_false)

#%%
# =============================================================================
# Count the missing values
# =============================================================================
missing_df_raw_true = df_raw_true.isnull().sum().sum()
missing_df_raw_false = df_raw_false.isnull().sum().sum()
missing_df_label_true = df_label_true.isnull().sum().sum()
missing_df_label_false = df_label_false.isnull().sum().sum()

