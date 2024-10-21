# DataFrame Comparison

This repository contains a set of functions designed to compare three DataFrames together and perform pairwise comparisons as well. Additionally, it includes filtering, merging, and saving functions for specific conditions, as well as key column handling, testing, and validation features. JSON comparisons are also supported.

### Key Features:
- **DataFrame Comparison**: Simultaneous comparison of three DataFrames, with the option for pairwise comparisons. The comparison is based on key columns shared between the DataFrames.
- **Key Column Handling**: Functions leverage key columns to align and merge the DataFrames efficiently.
- **Filtering Functions**: Filter rows and columns based on specific conditions, ensuring that only the relevant data is processed.
- **Merge Functionality**: Merge DataFrames based on a common key column, keeping shared columns and handling any discrepancies between the data.
- **Saving Results**: Save the comparison and filtering results into CSV files for further analysis.
- **JSON Comparisons**: Support for comparing data stored in JSON format.
- **Testing and Validation**: Includes test functions and data validation to check for inconsistencies, such as missing or mismatched data across DataFrames.

### Important:
- **Do not forget to add the `config.py` file to your `.gitignore`** to avoid exposing sensitive information like API tokens. 
- An example configuration file (`config.py`) is provided in the directory for reference.

These functions simplify the comparison of various data sources, such as CSV files, APIs, and JSON data, with **Pandas** as the main data processing library.
