# =============================================================================
# Packages
# =============================================================================
from config import config
import requests
import pandas as pd  
from io import StringIO
import numpy as np
#%%
# =============================================================================
# Root Directory
# =============================================================================
root_directory = 'F:/open_git/comparing_databases' 
#%%
# =============================================================================
# DataFrame 1 - exporting records
# =============================================================================
# Initial data
data = {
    'token': config['api_token'],
    'content': 'record',
    'action': 'export',
    'format': 'csv',
    'type': 'flat',
    'csvDelimiter': '',
    'rawOrLabel': 'raw',
    'rawOrLabelHeaders': 'raw',
    'exportCheckboxLabel': 'true',
    'exportSurveyFields': 'true',
    'exportDataAccessGroups': 'false',
    'returnFormat': 'json'
}

# Making the request to the API
r = requests.post(config['api_url'], data=data)

# Check request status
if r.status_code != 200:
    print(f"Error accessing the API: {r.status_code} - {r.text}")
else:
    try:
        # Load the CSV from the response
        df_api = pd.read_csv(StringIO(r.text), delimiter=',', low_memory=False)

#        # Specify the variables you want to exclude
#        exclude_fields = [
#            'name'  # Example of column to exclude, adjust as needed
#        ]

        # Exclude the specified variables
#        df_api = df_api.drop(columns=exclude_fields, errors='ignore')

        # Display the first few rows of the DataFrame for verification
#        print(df_api.head())

        # Optional: Save the DataFrame to a CSV file
#        df.to_csv('dados_redcap.csv', index=False, sep=';', encoding='utf-8')
#        print("Data exported and saved to 'dados_redcap.csv'")

    except Exception as e:
        print("Error processing the data:", e)
      
# =============================================================================
# Dataframe df_veg 
# =============================================================================
df_veg = pd.read_csv (f"{root_directory}/data/df_veg_DATA_2024-09-26_0930.csv")
#colunas = list(df2.columns)
#%%
# =============================================================================
# Dataframe df_mast
# =============================================================================
df_mast_1 = pd.read_csv (f"{root_directory}/data/df_api-DadosParadf2Parte_DATA_2024-09-19_1614.csv")
df_mast_2 = pd.read_csv (f"{root_directory}/data/df_api-DadosParadf2SOUP_DATA_2024-09-19_1614.csv")
df_mast = pd.merge(df_mast_1, df_mast_2, on='record_id', how='outer')

#%%
# =============================================================================
# DataFrames
# =============================================================================
# Creating three DataFrames to test some of following functions

data1 = {
    'A': [1, 2, 3, 4, 5],
    'B': [4, 5, 6, 7, 8],
    'C': [7, 8, 9, 10, 11],
    'record_id': [1, 2, 3, 4, 6]  
}

data2 = {
    'B': [4, 5, 6, 7, 8],
    'C': [7, 8, 9, 10, 12],
    'D': [10, 11, 12, 13, 14],
    'record_id': [1, 2, 3, 4, 5]  
}

data3 = {
    'C': [1, 2, 3, 6, 7],
    'D': [4, 5, 6, 7, 8],
    'E': [11, 12, 13, 14, 15],  
    'record_id': [3, 6, 9, 1, 10]  
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)


df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)

#%%
# =============================================================================
# Comparison between variables of 3 dataFrames 
# =============================================================================
def compare_and_save_columns(df1, df2, df3, name1='df1', name2='df2', name3='df3', output_directory='./results'):
    """
    Compares the columns of three DataFrames, identifying common and exclusive columns,
    and saves the results as CSV files.

    Parameters:
    - df1, df2, df3: DataFrames to be compared.
    - name1, name2, name3: Strings - Reference names for the DataFrames.
    - output_directory: String - Path to save the results.

    Returns:
    - dict: Containing common and exclusive columns for each pair of DataFrames and among all three.
    """
    # Get columns from each DataFrame as sets
    columns_df1 = set(df1.columns)
    columns_df2 = set(df2.columns)
    columns_df3 = set(df3.columns)

    # Columns common to all DataFrames
    common_all = columns_df1 & columns_df2 & columns_df3

    # Columns exclusive to each DataFrame
    exclusive_df1 = columns_df1 - (columns_df2 | columns_df3)
    exclusive_df2 = columns_df2 - (columns_df1 | columns_df3)
    exclusive_df3 = columns_df3 - (columns_df1 | columns_df2)

    # Prepare the results for saving and returning
    result = {
        "common_all": list(common_all),
        f"exclusive_{name1}": list(exclusive_df1),
        f"exclusive_{name2}": list(exclusive_df2),
        f"exclusive_{name3}": list(exclusive_df3),
    }
    
    # Save the results as CSV files
    for key, columns in result.items():
        pd.Series(columns).to_csv(f'{output_directory}/{key}.csv', index=False, header=False)

    return result

# Calling the comparison and save function
comparison_result = compare_and_save_columns(df1, df2, df3, 'df1', 'df2', 'df3', f'{root_directory}/results')

# Unpacking the results
common_all = comparison_result["common_all"]
exclusive_df1 = comparison_result["exclusive_df1"]
exclusive_df2 = comparison_result["exclusive_df2"]
exclusive_df3 = comparison_result["exclusive_df3"]

# Displaying the results
for category, columns in comparison_result.items():
    print(f"{category}: {columns}")
    
#%%
# =============================================================================
# Venn Diagram
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# Defining the area sizes according to the provided numbers
# We define 3 sets: df_api, df_veg, and df_mat
venn_counts = {
    '100': 3982,    # Exclusive to df_api
    '010': 1700,    # Exclusive to df_veg
    '001': 4,       # Exclusive to df_mat
    '110': 220,     # In df_api and df_veg, but not in df_mat
    '101': 26,      # In df_api and df_mat, but not in df_veg
    '011': 0,       # In df_veg and df_mat, but not in df_api
    '111': 1633     # In all three
}

# Creating the Venn diagram
plt.figure(figsize=(8, 8))
# plt.title("Venn Diagram for Variables between df_api, df_veg, and df_mat")
venn = venn3(subsets=venn_counts, set_labels=('df_api', 'df_veg', 'df_mat'))

# Adding borders to all sets
for subset in ('100', '010', '001', '110', '101', '011', '111'):
    patch = venn.get_patch_by_id(subset)
    if patch:  # Check if the subset exists in the diagram
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

# Force the display of the value '4' in the '001' set
venn.get_label_by_id('001').set_text('4')

# Display the diagram
plt.show()

#%%
# =============================================================================
# Comparison between DataFrames - pairwise
# =============================================================================
def compare_dataframe_columns(df1, df2, name1='DataFrame 1', name2='DataFrame 2'):
    """
    Compares the columns of two DataFrames and displays the common columns, 
    and those that are unique to each DataFrame.

    Parameters:
    - df1: pd.DataFrame - The first DataFrame to be compared.
    - df2: pd.DataFrame - The second DataFrame to be compared.
    - name1: str - Name of the first DataFrame (used for display).
    - name2: str - Name of the second DataFrame (used for display).

    Returns:
    - tuple - Three sets: common columns, unique to df1, and unique to df2.
    """
    # Get the columns of both DataFrames as sets
    columns_df1 = set(df1.columns)
    columns_df2 = set(df2.columns)
    
    # Columns that are in both DataFrames
    common = columns_df1.intersection(columns_df2)
#    print(f"Columns in both {name1} and {name2}:")
#    print(common_columns)    
    # Columns that are only in the first DataFrame
    unique_df1 = columns_df1.difference(columns_df2)
#    print(f"\nColumns unique to {name1}:")
#    print(unique_columns_df1)    
    # Columns that are only in the second DataFrame
    unique_df2 = columns_df2.difference(columns_df1)
#    print(f"\nColumns unique to {name2}:")
#    print(unique_columns_df2)
    
    return common, unique_df1, unique_df2
#%%
# =============================================================================
# Calling the previous function and saving the results
# =============================================================================

def compare_and_save_columns(df1, df2, name1, name2, output_directory, compare_func):
    """
    Compares columns between two DataFrames, saves the unique and common columns as CSV files,
    and returns the sets of columns for visualization.

    Parameters:
    - df1, df2: DataFrames to be compared.
    - name1, name2: Strings - Reference names of the DataFrames.
    - output_directory: String - Path to save the results.
    - compare_func: Function that compares the DataFrames and returns (common_columns, unique_columns_df1, unique_columns_df2).

    Returns:
    - tuple: (common_columns, unique_columns_df1, unique_columns_df2).
    """
    # Execute the comparison function
    common, unique_df1, unique_df2 = compare_func(df1, df2, name1, name2)

    # Save the unique and common columns to CSV files
    pd.Series(list(unique_df1)).to_csv(f'{output_directory}/unique_{name1}_vs_{name2}.csv', index=False, header=False)
    pd.Series(list(unique_df2)).to_csv(f'{output_directory}/unique_{name2}_vs_{name1}.csv', index=False, header=False)
    pd.Series(list(common)).to_csv(f'{output_directory}/common_{name1}_vs_{name2}.csv', index=False, header=False)

    # Return the sets for inspection
    return common, unique_df1, unique_df2

# Setting the output directory
output_directory = f'{root_directory}/results'

# Calling the function
common_columns, unique_columns_df1, unique_columns_df2 = compare_and_save_columns(df1, df2, 'df1', 'df2', output_directory, compare_dataframe_columns)

# Displaying the results
print(f"Common columns: {common_columns}")
print(f"Unique to df1: {unique_columns_df1}")
print(f"Unique to df2: {unique_columns_df2}")
#%%
# =============================================================================
# Overlap and differences between the previously created sets
# =============================================================================

def save_overlap_and_difference(set1, set2, output_directory, name1='Set1', name2='Set2'):
    """
    Calculates the overlap and difference between two sets and saves the results to CSV files.

    Parameters:
    - set1: set - The first set.
    - set2: set - The second set.
    - output_directory: str - Path to save the CSV files.
    - name1: str - Name of the first set for file naming.
    - name2: str - Name of the second set for file naming.
    
    Returns:
    - dict: Contains the overlap and difference between the sets.
    """
    # Calculate the overlap and the differences
    overlap = set1.intersection(set2)
    difference_set1 = set1 - set2
    difference_set2 = set2 - set1

    # Save the results to CSV files
    pd.Series(list(overlap)).to_csv(f'{output_directory}/overlap_{name1}_vs_{name2}.csv', index=False, header=False)
    pd.Series(list(difference_set1)).to_csv(f'{output_directory}/difference_{name1}_not_in_{name2}.csv', index=False, header=False)
    pd.Series(list(difference_set2)).to_csv(f'{output_directory}/difference_{name2}_not_in_{name1}.csv', index=False, header=False)

    # Return a dictionary with the results
    return {
        'overlap': overlap,
        'difference_set1': difference_set1,
        'difference_set2': difference_set2
    }
#%%
# Test
output_directory = f"{root_directory}/results"  
set1 = {'A', 'B', 'C', 'D'}
set2 = {'B', 'C', 'E'}

results = save_overlap_and_difference(set1, set2, output_directory, name1='Set1', name2='Set2')

print("Overlap:", results['overlap'])
print("Difference in Set1 but not in Set2:", results['difference_set1'])
print("Difference in Set2 but not in Set1:", results['difference_set2'])
#%%
# =============================================================================
# Create a DataFrame from variables that are in the set
# =============================================================================

def create_subset_dataframe(df, variable_set, key_column='record_id'):
    """
    Creates a new DataFrame based on a set of variables that exist in a DataFrame,
    automatically including the key column, if available.

    Parameters:
    - df: pd.DataFrame - The original DataFrame.
    - variable_set: set - Set of variable names to include in the new DataFrame.
    - key_column: str - The name of the key column that should be included in the new DataFrame.

    Returns:
    - pd.DataFrame - DataFrame containing only the specified variables and the key column.
    """
    # Add the key column to the set of variables
    if key_column in df.columns:
        variable_set = variable_set.union({key_column})
    
    # Filter the variables from the set that are present in the DataFrame's columns
    selected_columns = [col for col in variable_set if col in df.columns]
    
    # Create the new DataFrame with only the selected columns
    new_df = df[selected_columns].copy()
    
    return new_df

# Example set of variables
variable_set = {'A', 'B', 'Z'}  # Replace with the names of the variables you want

# Creating a subset DataFrame
subset_df = create_subset_dataframe(df1, variable_set, key_column='record_id')

# Saving the DataFrame
subset_df.to_csv(f'{output_directory}/subset_dataframe.csv', index=False, sep=';', encoding='utf-8')


#%%
# =============================================================================
# Filter Variables
# =============================================================================
  
def filter_dataframe_columns(df, columns_of_interest):
    """
    Filters a DataFrame to keep only the columns of interest that are present.

    Parameters:
    - df: pd.DataFrame - The DataFrame to be filtered.
    - columns_of_interest: list - List of columns you want to keep.

    Returns:
    - pd.DataFrame - The filtered DataFrame with present columns or an informative message.
    """
    # Checking which columns are missing
    missing_columns = [col for col in columns_of_interest if col not in df.columns]
    if missing_columns:
        print("Missing columns:", missing_columns)
    else:
        print("All columns are present.")

    # Filter only the columns that are present in the DataFrame
    present_columns = [col for col in columns_of_interest if col in df.columns]

    # Create the DataFrame with the present columns
    df_mod = df[present_columns]

    # Check and display the resulting DataFrame
    if not df_mod.empty:
        return df_mod
    else:
        print("No matching columns found.")
        return None

# DataFrame df1
columns_of_interest_df1 = [
    'column_id', 'column2', 'column3',  
    'column4', 'column5', 'column6', 
    'column7'
]

# Add variables column8_1 to column8_66, for each j from 0 to 2
for i in range(1, 67):
    for j in range(3):  # j ranges from 0 to 2
        field_name = f'column8_{i}___{j}'
        columns_of_interest_df1.append(field_name)
    
# Add variables column9 from 1 to 10
for i in range(1, 11):
    field_name = f'column9___{i}'
    columns_of_interest_df1.append(field_name)

# Display the result to verify
print(columns_of_interest_df1)

# Calling the function
df1_mod = filter_dataframe_columns(df1, columns_of_interest_df1)

#%%
# ====================================================================================
# Are there any observations in record_id that are present in both df1 and df2?
# ====================================================================================

def compare_key_column(df1, df2, key_column):
    """
    Compares the key column between two DataFrames and identifies the common key values,
    those unique to the first DataFrame, and those unique to the second.

    Parameters:
    - df1: pd.DataFrame - The first DataFrame.
    - df2: pd.DataFrame - The second DataFrame.
    - key_column: str - The name of the key column for comparison.

    Returns:
    - dict: A dictionary containing lists of common keys, keys unique to df1, and keys unique to df2.
    """
    # Extract key columns as sets
    keys_df1 = set(df1[key_column].astype(str))
    keys_df2 = set(df2[key_column].astype(str))

    # Identify common keys, keys unique to df1, and keys unique to df2
    common_keys = keys_df1.intersection(keys_df2)
    unique_to_df1 = keys_df1 - keys_df2
    unique_to_df2 = keys_df2 - keys_df1

    # Return the results in a dictionary
    return {
        "common_keys": list(common_keys),
        "unique_to_df1": list(unique_to_df1),
        "unique_to_df2": list(unique_to_df2)
    }

# Comparing the key column
result = compare_key_column(df1, df2, 'record_id')

# Displaying the results
print("Common Keys:", result["common_keys"])
print("Keys Unique to df1:", result["unique_to_df1"])
print("Keys Unique to df2:", result["unique_to_df2"])
#%%
# ==================================================================================== 
# Is there any variable in df2 that corresponds to record_id in df1? 
# ====================================================================================

def find_columns_with_matching_record_id(df1, df2, key_column, columns_to_check):
    """
    Checks which columns in `df2` contain matches for the values of `key_column` in `df1`.

    Parameters:
    - df1: pd.DataFrame - DataFrame containing the key column.
    - df2: pd.DataFrame - DataFrame containing the columns to check.
    - key_column: str - Name of the key column in `df1` ('record_id').
    - columns_to_check: list - List of columns in `df2` to check for matches.

    Returns:
    - dict: Dictionary with columns that have matches ('matched_columns') and columns that don't ('unmatched_columns').
    """
    # Convert the key column to a set for quick match checking
    record_ids = set(df1[key_column].astype(str))
    
    # Lists to store columns with and without matches
    matched_columns = []
    unmatched_columns = []

    # Check each column in columns_to_check
    for col in columns_to_check:
        if col in df2.columns:
            # Convert the column to str and check intersection with `record_ids`
            matches = record_ids.intersection(set(df2[col].astype(str)))
            if matches:
                matched_columns.append(col)
            else:
                unmatched_columns.append(col)
    
    return {
        "matched_columns": matched_columns,
        "unmatched_columns": unmatched_columns
    }


# Creating one more df to test
data4 = {
    'record_id_2': ['3', '6', '7', '8', '9'],  # Not all values match
    'record_id_3': ['10', '2', '11', '12', '13'],  # '2' will match with df1
    'recordcreation_date': ['2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05']
}

df4 = pd.DataFrame(data4)

# Specifying the columns in df4 to check for matching values with 'record_id' in df1
columns_to_check = ['record_id_2', 'record_id_3', 'recordcreation_date']

# Testing the function to find matching columns
result = find_columns_with_matching_record_id(df1, df4, 'record_id', columns_to_check)

# Displaying the results
print("Columns with matches:", result["matched_columns"])
print("Columns without matches:", result["unmatched_columns"])

#%%
# ====================================================================================
# Are the observations for the same record_id for corresponding variables different?
# ====================================================================================
def compare_variable_values(df1, df2, record_id_col1, record_id_col2, col1, col2):
    """
    Compares the values of two columns in different DataFrames based on the record_id,
    after renaming the columns to avoid conflicts.

    Parameters:
    - df1: pd.DataFrame - The first DataFrame.
    - df2: pd.DataFrame - The second DataFrame.
    - record_id_col1: str - The name of the column containing the record_id in df1.
    - record_id_col2: str - The name of the column containing the record_id in df2.
    - col1: str - The name of the column in the first DataFrame to be compared.
    - col2: str - The name of the column in the second DataFrame to be compared.

    Returns:
    - pd.DataFrame - A DataFrame with the record_id, renamed values of col1 and col2, and an equality indicator.
    """
    # Rename columns in df1 and df2 to avoid conflicts
    df1_renamed = df1.rename(columns={col1: f"{col1}_df1"})
    df2_renamed = df2.rename(columns={col2: f"{col2}_df2"})

    # Performing an inner join between df1 and df2
    merged_df = pd.merge(df1_renamed[[record_id_col1, f"{col1}_df1"]], 
                         df2_renamed[[record_id_col2, f"{col2}_df2"]], 
                         left_on=record_id_col1, right_on=record_id_col2, how='inner')

    # Add a column to indicate whether the values are equal
    merged_df[f"{col1}_are_equal"] = merged_df[f"{col1}_df1"] == merged_df[f"{col2}_df2"]
    
    # Remove `record_id_col2` after the merge
    merged_df.drop(columns=record_id_col2, inplace=True)

    return merged_df
##
# Dataframes for testing
data1 = {
    'record_id': [1, 2, 3, 4],
    'coluna_a': ['A', 'B', 'C', 'D']
}
data2 = {
    'record_id_2': [1, 2, 3, 5],
    'coluna_b': ['A', 'B', 'X', 'D']
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Testing the function
result = compare_variable_values(df1, df2, 'record_id', 'record_id_2', 'coluna_a', 'coluna_b')
#%%
# ====================================================================================
# Concatenating the comparison results
# ====================================================================================
from functools import reduce

def batch_compare_variable_values(df1, df2, record_id_col1, record_id_col2, column_pairs, how='inner'):
    """
    Compares the values of specified column pairs in different DataFrames based on record_id,
    and returns a concatenated DataFrame with all results.

    Parameters:
    - df1: pd.DataFrame - The first DataFrame.
    - df2: pd.DataFrame - The second DataFrame.
    - record_id_col1: str - The name of the column containing the record_id in df1.
    - record_id_col2: str - The name of the column containing the record_id in df2.
    - column_pairs: list of tuples - List of tuples where each tuple contains a pair of columns (col_df1, col_df2) to be compared.
    - how: str - Type of join to use (default 'inner').

    Returns:
    - pd.DataFrame - DataFrame with concatenated results.
    """
    # List to store the resulting DataFrames from each comparison
    dataframes = []
    
    # Loop through each column pair to compare and store in dataframes
    for col1, col2 in column_pairs:
        result = compare_variable_values(df1, df2, record_id_col1, record_id_col2, col1, col2)
        dataframes.append(result)
    
    # Merge all DataFrames in the list into a single DataFrame
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=record_id_col1, how=how), dataframes)

    return merged_df

# Defining the column pairs to compare
column_pairs = [
    ('gender', 'gender_id'),
    ('age', 'age'),
    ('monthly_income', 'monthly_income'),
    ('marital_status', 'marital_status'),
    ('work_hours', 'work_hours')
]

# Usage
concatenated_results = batch_compare_variable_values(
    df1, df2, 'record_id', 'record_id_2', column_pairs
)

# Saving the final DataFrame
concatenated_results.to_csv(f'{root_directory}/results/comparison_variables_df1_vs_df2.csv', index=False, sep=';', encoding='utf-8')
#%%
# ====================================================================================
# There are any other difference between dataframes?
# ====================================================================================

def generate_diff_dataframe(df1, df2, key_column):
    """
    Generates a DataFrame containing only the variables with differences between two DataFrames.
    Missing values (NaN) in both columns are considered equivalent.

    Parameters:
    - df1: pd.DataFrame - The first DataFrame.
    - df2: pd.DataFrame - The second DataFrame.
    - key_column: str - The name of the column containing the IDs for comparison.

    Returns:
    - pd.DataFrame: A DataFrame containing only the variables with differences.
    """

    # Ensure that 'key_column' types are the same in both DataFrames
    df1[key_column] = df1[key_column].astype(str)
    df2[key_column] = df2[key_column].astype(str)
    
    # Remove duplicates based on the key_column
    df1_unique = df1.drop_duplicates(subset=key_column)
    df2_unique = df2.drop_duplicates(subset=key_column)
    
    # Merge the two DataFrames based on the 'key_column'
    merged_df = pd.merge(df1_unique, df2_unique, on=key_column, suffixes=('_df1', '_df2'), how='inner')
    
    # Function to compare a single variable and return the columns with differences
    def compare_column(df, column):
        mask_diff = (
            (df[f'{column}_df1'] != df[f'{column}_df2']) & 
            ~(df[f'{column}_df1'].isna() & df[f'{column}_df2'].isna())
        )
        if mask_diff.any():
            return df[[f'{column}_df1', f'{column}_df2']]
        else:
            return pd.DataFrame()

    # Use reduce to aggregate DataFrames with differences across all columns
    diff_df = reduce(
        lambda left, right: pd.concat([left, right], axis=1),
        [compare_column(merged_df, col) for col in df1.columns if col != key_column]
    )

    # Add key_column to the result if there are differences
    if not diff_df.empty:
        diff_df.insert(0, key_column, merged_df[key_column])

    return diff_df


# Test
data1 = {
    'record_id': ['1', '2', '3', '4', '5', '7'],
    'var_a': [10, 20, 30, 40, None, None],
    'var_b': ['A', 'B', 'C', 'D', 'E', 'F'],
    'var_c': [10, 2.5, 3.5, None, 5.5, 5],
    'var_d': [None, 'cha', 'almoço', None, None, None]
}

data2 = {
    'record_id': ['1', '2', '3', '4', '6', '7'],
    'var_a': [10, 25, None, 40, 50, 60],
    'var_b': ['A', 'B', 'C', 'X', 'E', 'F'],
    'var_c': [1.5, 2.8, 3.5, 4.5, None, None],
    'var_d': [None, 'cha', 'almoço', None, 1, 1]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Call the function
diff_df = generate_diff_dataframe(df1, df2, 'record_id')

# Display the result
print(diff_df)

#%%
def compare_dataframes(df1, df2, key_column):
    """
    Compares two DataFrames based on a key column and finds the differences between them.
    Missing values (NaN) in both columns are considered equivalent.

    Parameters:
    - df1: pd.DataFrame - The first DataFrame.
    - df2: pd.DataFrame - The second DataFrame.
    - key_column: str - The name of the column containing the IDs for comparison.

    Returns:
    - tuple: (list_of_variables_with_differences, list_of_record_ids_with_differences)
    """
    
    # Ensure that 'key_column' types are the same in both DataFrames
    df1[key_column] = df1[key_column].astype(str)
    df2[key_column] = df2[key_column].astype(str)
    
    # Remove duplicates based on the key_column (keeping the first occurrence)
    df1_unique = df1.drop_duplicates(subset=key_column)
    df2_unique = df2.drop_duplicates(subset=key_column)
    
    # Merge the two DataFrames based on the 'key_column'
    merged_df = pd.merge(df1_unique, df2_unique, on=key_column, suffixes=('_df1', '_df2'), how='inner')

    # Lists to store the variables and record_ids with differences
    list_of_variables_with_differences = []
    list_of_record_ids_with_differences = set()  # Use a set to ensure no duplicates

    # Loop through each column and check for differences, ignoring the key_column
    for column in df1.columns:
        if column == key_column:
            continue  # Skip the key column
        
        # Create a mask to find differences
        mask_diff = (
            (merged_df[f'{column}_df1'] != merged_df[f'{column}_df2']) &
            ~(merged_df[f'{column}_df1'].isna() & merged_df[f'{column}_df2'].isna())
        )

        # If there are differences, add the column to the list
        if mask_diff.any():
            list_of_variables_with_differences.append(column)

            # Add the record_ids with differences to the list
            list_of_record_ids_with_differences.update(merged_df.loc[mask_diff, key_column].tolist())

    # Convert the set of record_ids to a sorted list
    list_of_record_ids_with_differences = sorted(list_of_record_ids_with_differences)

    return list_of_variables_with_differences, list_of_record_ids_with_differences

# Call the function
variables_with_differences, record_ids_with_differences = compare_dataframes(df1, df2, 'record_id')

# Results
print("Variables with differences:", variables_with_differences)
print("Record IDs with differences:", record_ids_with_differences)

#%%
# =============================================================================
# Comparison between variables of 2 dataFrames 
# =============================================================================
def generate_difference_report(df1, df2, key_column, variables_with_differences, record_ids_with_differences):
    """
    Creates a DataFrame based on the variables and record_ids that show differences between two DataFrames.
    For each pair of compared variables, a new column is created indicating whether the values are equal or different.
    Missing values (NaN) are temporarily transformed to 999 to facilitate comparison.

    Parameters:
    - df1: pd.DataFrame - The first DataFrame.
    - df2: pd.DataFrame - The second DataFrame.
    - key_column: str - The name of the column containing the IDs for comparison.
    - variables_with_differences: list - List of variables with differences between the DataFrames.
    - record_ids_with_differences: list - List of record_ids with differences between the DataFrames.

    Returns:
    - pd.DataFrame: DataFrame with the compared variables and a column indicating True (equal) or False (different).
    """

    # Filter the DataFrames based on the record_ids that have differences
    df1_filtered = df1[df1[key_column].isin(record_ids_with_differences)]
    df2_filtered = df2[df2[key_column].isin(record_ids_with_differences)]

    # Initialize a new DataFrame with the key_column
    df_comparison = pd.DataFrame({key_column: df1_filtered[key_column]})

    # For each variable with differences, add the columns from df1, df2, and a comparison column
    for var in variables_with_differences:
        df_comparison[f'{var}_df1'] = df1_filtered[var].values
        df_comparison[f'{var}_df2'] = df2_filtered[var].values
        df_comparison[f'{var}_equal'] = df_comparison[f'{var}_df1'] == df_comparison[f'{var}_df2']

    return df_comparison

#%%
# Call the previous comparison function
variables_with_differences, record_ids_with_differences = compare_dataframes(df1, df2, 'record_id')

# Call the new function to create the comparison DataFrame
df_comparison = generate_difference_report(df1, df2, 'record_id', variables_with_differences, record_ids_with_differences)

# Display the final DataFrame
print(df_comparison)

# Save the comparison DataFrame
df_comparison.to_csv(f'{root_directory}/results/comparison_api_raw_check_true.csv', index=False, encoding='utf-8')
#%%
# ====================================================================================
# Merging dataframes based on key column
# ====================================================================================

def merge_and_filter_common_columns(df1, df2, key_col):
    """
    Merges two DataFrames based on a key column, keeping all observations from df1
    and only the common columns between the two DataFrames without duplicating them.

    Parameters:
    - df1: pd.DataFrame - The first DataFrame that contains the 'included' column.
    - df2: pd.DataFrame - The second DataFrame to merge.
    - key_col: str - The common key column for merging.
            
    Returns:
    - pd.DataFrame: Resulting DataFrame after the merge, containing only the common columns from df1.
    """
    
    # Filter df1 where 'included' is 1 and 'gender' is 'f'
    df1_filtered = df1[(df1['included'] == 1) & (df1['gender'] == 'f')]
    
    # Find common columns between the two DataFrames
    common_cols = df1_filtered.columns.intersection(df2.columns).tolist()
    
    # Perform the merge, keeping only the common columns from df1
    merged_df = pd.merge(
        df1_filtered[common_cols], df2[[key_col]], 
        on=key_col, how='left'
    )

    return merged_df

# Test data for df1 and df2
data1 = {
    'record_id': [1, 2, 3, 4, 5],
    'included': [1, 1, 0, 1, 0],
    'gender': ['f', 'm', 'f', 'f', 'm'],
    'age': [23, 34, 45, 29, 40],
    'height': [160, 175, 170, 165, 180]
}

data2 = {
    'record_id': [1, 2, 3, 4, 6],
    'gender': ['f', 'm', 'f', 'f', 'f'],
    'weight': [55, 75, 65, 60, 58],
    'height': [160, 175, 170, 165, 168]
}

# Create DataFrames
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Test the function
df1_merged_df2 = merge_and_filter_common_columns(df1, df2, 'record_id')

# Display the result
df1_merged_df2
#%%
# ====================================================================================
# Checking the merge quality
# ====================================================================================
def diagnose_merge(df1, df2, key_column):
    """
    Diagnoses the result of merging two DataFrames based on a key column.
    
    Parameters:
    - df1: pd.DataFrame - The first DataFrame.
    - df2: pd.DataFrame - The second DataFrame.
    - key_column: str - The name of the key column for merging.
    
    Returns:
    - pd.DataFrame: The result of the merge and details of any potential issues.
    """
    # Perform the merge
    merged_df = pd.merge(df1, df2, on=key_column, how='outer', indicator=True)
    
    # Filter cases where the merge failed (present only in one of the DataFrames)
    diff_df = merged_df[merged_df['_merge'] != 'both']
    
    # Display the result of the merge and problematic rows
    print(f"Total records in the merge: {len(merged_df)}")
    print(f"Total discrepancies (present only in one of the DataFrames): {len(diff_df)}")
    
    if not diff_df.empty:
        print("Rows with discrepancies in the merge (present in one DataFrame but not the other):")
        print(diff_df[[key_column, '_merge']])
    else:
        print("No discrepancies found in the merge.")
    
    return merged_df

# Usage
merged_diagnosis = diagnose_merge(df1, df2, 'record_id')

# View the result of the merge
print(merged_diagnosis.head(6))

# Analyzing the merge status
merge_status_column = merged_diagnosis[['record_id', '_merge']]
left_only_count = (merged_diagnosis['_merge'] == 'left_only').sum()
right_only_count = (merged_diagnosis['_merge'] == 'right_only').sum()
both_count = (merged_diagnosis['_merge'] == 'both').sum()

print(f"Records only in df1: {left_only_count}")
print(f"Records only in df2: {right_only_count}")
print(f"Records in both DataFrames: {both_count}")
