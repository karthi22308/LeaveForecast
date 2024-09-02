import pandas as pd
import numpy as np
from faker import Faker
import random
import datetime

# Initialize Faker
fake = Faker()

# Parameters
num_employees = 80
months = [1, 2, 3, 4, 5, 6]  # Last 6 months
num_holidays_per_month = 2  # Average number of holidays per month

# Generate employee IDs
employee_ids = [f'EMP{str(i).zfill(3)}' for i in range(1, num_employees + 1)]

# Generate attendance data
attendance_data = []

for employee_id in employee_ids:
    for month in months:
        num_leaves = random.randint(0, 5)  # Random number of leave days between 0 and 5
        for _ in range(num_leaves):
            leave_date = fake.date_between_dates(
                date_start=datetime.date(2024, month, 1),
                date_end=datetime.date(2024, month, 28)
            )
            attendance_data.append([employee_id, month, 1, leave_date])

# Convert to DataFrame
attendance_df = pd.DataFrame(attendance_data, columns=['employeeid', 'month', 'leavedays', 'leave_date'])

# Save to Excel
attendance_df.to_excel('attendance.xlsx', index=False)

# Generate holiday data
holiday_data = []
months = [1, 2, 3, 4, 5, 6,7]
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
