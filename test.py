import pandas as pd


def preprocess_data(attendance_df):
    # Print the original columns to check the current structure
    print("Original columns:", attendance_df.columns.tolist())

    # Calculate the number of months and columns
    num_months = 6
    expected_columns = num_months * 4  # Each month has 4 columns: Month, Id, Total Leaves, Empty

    # Generate unique column names
    new_columns = [f'Month_{i + 1}' for i in range(num_months)] + \
                  [f'Id_{i + 1}' for i in range(num_months)] + \
                  [f'Total_Leaves_{i + 1}' for i in range(num_months)] + \
                  [f'Empty_{i + 1}' for i in range(num_months)]

    # Adjust if there are extra columns
    if len(attendance_df.columns) > len(new_columns):
        new_columns += [f'Extra_{i + 1}' for i in range(len(attendance_df.columns) - len(new_columns))]
    elif len(attendance_df.columns) < len(new_columns):
        new_columns = new_columns[:len(attendance_df.columns)]

    # Rename columns
    attendance_df.columns = new_columns

    # Melt the DataFrame to long-form (tidy)
    long_df = pd.DataFrame()
    for i in range(num_months):
        temp_df = attendance_df[[f'Month_{i + 1}', f'Id_{i + 1}', f'Total_Leaves_{i + 1}']].copy()
        temp_df.columns = ['Month', 'Id', 'Total_Leaves']
        temp_df['Month_Number'] = i + 1
        long_df = pd.concat([long_df, temp_df])

    # Keep only the rows corresponding to the last month (Month_6)
    last_month_df = long_df[long_df['Month_Number'] == 6]
    last_month_ids = last_month_df['Id'].unique()

    # Filter out employees not present in the last month
    long_df = long_df[long_df['Id'].isin(last_month_ids)]

    # Calculate the leave history for each employee
    long_df['leaves_last_month'] = long_df.groupby('Id')['Total_Leaves'].shift(1).fillna(0)
    long_df['leaves_last_2_months'] = long_df.groupby('Id')['Total_Leaves'] \
        .rolling(window=2).sum().reset_index(level=0, drop=True).shift(1).fillna(0)
    long_df['leaves_last_6_months'] = long_df.groupby('Id')['Total_Leaves'] \
        .rolling(window=6).sum().reset_index(level=0, drop=True).shift(1).fillna(0)

    return long_df, 'Total_Leaves_6'

# Sample usage
attendance_df = pd.read_excel('testdata.xlsx')
long_df, last_month_col = preprocess_data(attendance_df)
