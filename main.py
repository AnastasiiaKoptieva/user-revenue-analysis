import pandas as pd
import matplotlib.pyplot as plt


xls = pd.ExcelFile('Когортний аналіз.xlsx')
df = xls.parse('Revenue cohort')
df = df.iloc[1:].copy()
df.columns = ['Cohort Week', 'Users'] + list(range(len(df.columns) - 2))
df['Users'] = pd.to_numeric(df['Users'], errors='coerce')

for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')


plt.figure(figsize = (14, 8))

for _, row in df.iterrows():
    cohort = row['Cohort Week']
    users = row['Users']

    if pd.notna(users) and users > 0:
        revenue = row.iloc[2:]
        arpu_by_week = revenue.cumsum() / users
        plt.plot(arpu_by_week.index, arpu_by_week.values, label=f'cohort {int(cohort)}')


plt.xlabel('Week')
plt.ylabel('APPU')

plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize='small')

plt.grid(True)
plt.tight_layout()
plt.show()