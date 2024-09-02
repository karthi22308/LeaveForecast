import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import datetime
import base64



def preprocess_data(attendance_df, holidays_df):

    attendance_df['leave_date'] = pd.to_datetime(attendance_df['leave_date'])
    holidays_df['holiday_date'] = pd.to_datetime(holidays_df['holiday_date'])


    attendance_df['month'] = attendance_df['leave_date'].dt.month
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
    #st.write(f'Mean Absolute Error: {mae}')

    return model



def predict_leaves(model, next_month_features):
    predicted_leaves = model.predict(next_month_features)
    return predicted_leaves



def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="predicted_leaves.csv">Download predicted leaves as CSV</a>'
    return href



st.title('Employee Leave Forecast')


attendance_file = st.file_uploader('Upload Attendance Excel', type='xlsx')
holidays_file = st.file_uploader('Upload Holidays Excel', type='xlsx')

if attendance_file and holidays_file:

    attendance_df = pd.read_excel(attendance_file)
    holidays_df = pd.read_excel(holidays_file)


    attendance_df = preprocess_data(attendance_df, holidays_df)


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


        st.write('Predicted Leaves for Next Month')
        st.write(grouped_predictions)

        st.markdown(get_table_download_link(grouped_predictions), unsafe_allow_html=True)
    else:
        st.write('No data available for the next month in the dataset.')
