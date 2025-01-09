
import streamlit as st
months = ['July', 'August', 'September', 'October', 'November', 'December']
month_to_int = {month: index +1 for index, month in enumerate(months)}

# Select start month using a selectbox
selected_month = 'September'

# Get the corresponding integer value for the selected month
month_number = month_to_int[selected_month]

print(month_number)