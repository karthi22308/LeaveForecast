import io

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import datetime
import base64



def preprocess_data(attendance_df, holidays_df):
    # Convert holiday dates to datetime and extract the month
    holidays_df['holiday_date'] = pd.to_datetime(holidays_df['holiday_date'])
    holidays_df['month'] = holidays_df['holiday_date'].dt.month

    # Calculate leaves for the last month, last 2 months, and last 6 months
    attendance_df['leaves_last_month'] = attendance_df.groupby('employeeid')['leavedays'].shift(1).fillna(0)

    attendance_df['leaves_last_2_months'] = attendance_df.groupby('employeeid')['leavedays'] \
        .rolling(window=2).sum().reset_index(level=0, drop=True).shift(1).fillna(0)

    attendance_df['leaves_last_6_months'] = attendance_df.groupby('employeeid')['leavedays'] \
        .rolling(window=6).sum().reset_index(level=0, drop=True).shift(1).fillna(0)

    # Count the number of holidays in each month
    holiday_counts = holidays_df.groupby('month').size().reset_index(name='holiday_count')

    # Merge holiday counts with attendance data
    attendance_df = attendance_df.merge(holiday_counts, on='month', how='left').fillna(0)

    return attendance_df



def train_model(attendance_df):

    features = ['leaves_last_month', 'leaves_last_2_months', 'leaves_last_6_months', 'holiday_count']
    target = 'leavedays'

    X = attendance_df[features]
    y = attendance_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    #st.write(f'Mean Absolute Error: {mae}')

    return model


def construct_dataframe( x):
    df = pd.read_excel('Dataset.xlsx', engine='openpyxl')  # Load your data
    # Adjust x to zero-indexed for Python
    x = x - 1

    # Define a list to hold the rows for the new DataFrame
    rows = []

    # Calculate the number of months in the DataFrame
    num_months = len(df.columns) // 3

    # Check if x is within the valid range
    if x + 5 >= num_months:
        raise ValueError("The range exceeds the available months in the dataset.")

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        for i in range(6):  # We are taking x, x+1, ..., x+5 (total 6 months)
            month_index = x + i
            month_col = df.columns[month_index * 3]
            id_col = df.columns[month_index * 3 + 1]
            leaves_col = df.columns[month_index * 3 + 2]

            # Extract the necessary data
            month = row[month_col]
            employeeid = row[id_col]
            leavedays = row[leaves_col]

            # Append the data to the rows list
            rows.append([employeeid, month, leavedays])

    # Create the new DataFrame
    new_df = pd.DataFrame(rows, columns=['employeeid', 'month', 'leavedays'])

    return new_df
def create_excel_file(df1, df2):
    """Create an Excel file with two sheets and return it as a bytes buffer."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)
    buffer.seek(0)  # Move to the beginning of the buffer
    return buffer
def predict_leaves(model, next_month_features):
    # Predict leaves and round to the nearest integer
    predicted_leaves = model.predict(next_month_features)
    predicted_leaves = np.round(predicted_leaves).astype(int)
    return predicted_leaves


def getrealdata(month_number):
    df = pd.read_excel('Dataset.xlsx', engine='openpyxl')  # Load your data
    # Convert month_number to zero-based index for calculation
    month_index = month_number - 1

    # Calculate the number of sets of 'Month', 'Id', 'Total Leaves'
    num_sets = len(df.columns) // 3

    # Check if month_index is within valid range
    if month_index >= num_sets:
        raise ValueError("The specified month number is out of bounds.")

    # Define a list to hold the rows for the new DataFrame
    rows = []

    # Extract columns based on month_index
    month_col = df.columns[month_index * 3]  # Month column
    id_col = df.columns[month_index * 3 + 1]  # Id column
    leaves_col = df.columns[month_index * 3 + 2]  # Total Leaves column

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        employeeid = row[id_col]
        leavedays = row[leaves_col]

        # Append the data to the rows list
        rows.append([employeeid, month_number, leavedays])

    # Create the new DataFrame
    new_df = pd.DataFrame(rows, columns=['employeeid', 'month', 'leavedays'])

    return new_df

def get_table_download_link(df):

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_leaves.csv">Download predicted leaves as CSV</a>'
    return href



st.title('Employee Leave Forecast')




# Define the available months
months = ['July', 'August', 'September', 'October', 'November', 'December']
month_to_int = {month: index +1 for index, month in enumerate(months)}

# Select start month using a selectbox
selected_month = st.selectbox('Select month for prediction', months)
month_number = month_to_int[selected_month]
if st.button('Submit'):

    # Get the corresponding integer value for the selected month


    attendance_df = construct_dataframe(month_number)
    holidays_df = pd.read_excel('holidays.xlsx')


    attendance_df = preprocess_data(attendance_df, holidays_df)

    # Determine the highest (last) month present in the data
    last_month = attendance_df['month'].max()

    # Filter attendance data to include only employees present in the last month
    last_month_ids = attendance_df[attendance_df['month'] == last_month]['employeeid'].unique()
    attendance_df = attendance_df[attendance_df['employeeid'].isin(last_month_ids)]


    model = train_model(attendance_df)


    current_month = datetime.datetime.now().month
    next_month = (current_month % 6) + 1
    next_month_features = attendance_df[attendance_df['month'] == next_month][
    ['leaves_last_month', 'leaves_last_2_months', 'leaves_last_6_months', 'holiday_count']]


    if not next_month_features.empty:
        predicted_leaves = predict_leaves(model, next_month_features)
        predicted_leaves_df = attendance_df[attendance_df['month'] == next_month][['employeeid']].copy()
        predicted_leaves_df['predicted_leaves'] = predicted_leaves
        grouped_predictions = predicted_leaves_df.groupby('employeeid').sum().reset_index()
        real = getrealdata(month_number+6)
        filtered_df = real[real['employeeid'].isin(grouped_predictions['employeeid'])]

        st.write('Predicted Leaves for Next Month')
        excel_buffer = create_excel_file(grouped_predictions, filtered_df)

        # Provide the download link
        st.download_button(
            label='Download Excel File',
            data=excel_buffer,
            file_name='comparison_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        st.write(grouped_predictions)

        file_path = 'output.xlsx'




    else:
        st.write('No data available for the next month in the dataset.')
