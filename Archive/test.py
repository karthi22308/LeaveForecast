import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import datetime
import base64

def preprocess_data(attendance_df, holidays_df):
    holidays_df['holiday_date'] = pd.to_datetime(holidays_df['holiday_date'])
    holidays_df['month'] = holidays_df['holiday_date'].dt.month

    attendance_df['leaves_last_month'] = attendance_df.groupby('employeeid')['leavedays'].shift(1).fillna(0)
    attendance_df['leaves_last_2_months'] = attendance_df.groupby('employeeid')['leavedays'] \
        .rolling(window=2).sum().reset_index(level=0, drop=True).shift(1).fillna(0)
    attendance_df['leaves_last_6_months'] = attendance_df.groupby('employeeid')['leavedays'] \
        .rolling(window=6).sum().reset_index(level=0, drop=True).shift(1).fillna(0)

    holiday_counts = holidays_df.groupby('month').size().reset_index(name='holiday_count')
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
    print(f'Mean Absolute Error: {mae}')

    return model

def construct_dataframe(x):
    df = pd.read_excel('Dataset.xlsx', engine='openpyxl')
    x = x - 1

    rows = []
    num_months = len(df.columns) // 3

    if x + 5 >= num_months:
        raise ValueError("The range exceeds the available months in the dataset.")

    for _, row in df.iterrows():
        for i in range(6):
            month_index = x + i
            month_col = df.columns[month_index * 3]
            id_col = df.columns[month_index * 3 + 1]
            leaves_col = df.columns[month_index * 3 + 2]

            month = row[month_col]
            employeeid = row[id_col]
            leavedays = row[leaves_col]

            rows.append([employeeid, month, leavedays])

    new_df = pd.DataFrame(rows, columns=['employeeid', 'month', 'leavedays'])

    return new_df

def create_excel_file(df1, df2):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet1', index=False, startcol=3)
    buffer.seek(0)
    with open('comparison_data.xlsx', 'wb') as f:
        f.write(buffer.read())
    print("Excel file 'comparison_data.xlsx' created successfully.")

def predict_leaves(model, next_month_features):
    predicted_leaves = model.predict(next_month_features)
    predicted_leaves = np.round(predicted_leaves).astype(int)
    return predicted_leaves

def get_real_data(month_number):
    df = pd.read_excel('Dataset.xlsx', engine='openpyxl')
    month_index = month_number - 1
    num_sets = len(df.columns) // 3

    if month_index >= num_sets:
        raise ValueError("The specified month number is out of bounds.")

    rows = []
    month_col = df.columns[month_index * 3]
    id_col = df.columns[month_index * 3 + 1]
    leaves_col = df.columns[month_index * 3 + 2]

    for _, row in df.iterrows():
        employeeid = row[id_col]
        leavedays = row[leaves_col]
        rows.append([employeeid, month_number, leavedays])

    new_df = pd.DataFrame(rows, columns=['employeeid', 'month', 'leavedays'])

    return new_df

def main():
    months = ['July', 'August', 'September', 'October', 'November', 'December']
    month_to_int = {month: index + 1 for index, month in enumerate(months)}

    print("Available months:", ", ".join(months))
    selected_month = 'November'
    if selected_month not in months:
        print("Invalid month selected. Exiting...")
        return

    month_number = month_to_int[selected_month]

    attendance_df = construct_dataframe(month_number)
    holidays_df = pd.read_excel('holidays.xlsx')

    attendance_df = preprocess_data(attendance_df, holidays_df)
    last_month = attendance_df['month'].max()
    last_month_ids = attendance_df[attendance_df['month'] == last_month]['employeeid'].unique()
    attendance_df = attendance_df[attendance_df['employeeid'].isin(last_month_ids)]

    model = train_model(attendance_df)
    next_month = month_number

    next_month_features = attendance_df[attendance_df['month'] == next_month][
        ['leaves_last_month', 'leaves_last_2_months', 'leaves_last_6_months', 'holiday_count']]

    if not next_month_features.empty:
        predicted_leaves = predict_leaves(model, next_month_features)
        predicted_leaves_df = attendance_df[attendance_df['month'] == next_month][['employeeid']].copy()
        predicted_leaves_df['predicted_leaves'] = predicted_leaves
        grouped_predictions = predicted_leaves_df.groupby('employeeid').sum().reset_index()
        employee_ids_in_df1 = grouped_predictions['employeeid'].unique()

        real = get_real_data(month_number + 6)
        #filtered_df = real[real['employeeid'].isin(employee_ids_in_df1)]
        filtered_rows = []

        # Loop through each row in df2
        for _, row in real.iterrows():
            # Check if the employeeid in the current row of df2 is in df1
            if row['employeeid'] in employee_ids_in_df1:
                # If it is, add the row to the filtered_rows list
                filtered_rows.append(row)

        # Convert the list of filtered rows back into a DataFrame
        print (filtered_rows)
        filtered_df = pd.DataFrame(filtered_rows)
        print(filtered_df)

        print("Predicted Leaves for Next Month:")


        create_excel_file(grouped_predictions, filtered_df)
    else:
        print("No data available for the next month in the dataset.")

if __name__ == "__main__":
    main()
