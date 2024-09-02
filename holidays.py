import pandas as pd
import numpy as np
from faker import Faker
import datetime

# Initialize Faker
fake = Faker()

# Parameters
months = [1, 2, 3, 4, 5, 6]  # Last 6 months
num_holidays_per_month = 2  # Average number of holidays per month

# Generate holiday data
holiday_data = []

for month in months:
    for _ in range(num_holidays_per_month):
        holiday_date = fake.date_between_dates(
            date_start=datetime.date(2024, month, 1),
            date_end=datetime.date(2024, month, 28)
        )
        holiday_data.append([holiday_date])

# Convert to DataFrame
holidays_df = pd.DataFrame(holiday_data, columns=['holiday_date'])

# Save to Excel
holidays_df.to_excel('holidays.xlsx', index=False)
