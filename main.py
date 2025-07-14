import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
        revenue = row.iloc[2:].dropna().values
        arpu = np.cumsum(revenue) / users
        x = np.arange(len(arpu))

        if len(arpu) >= 4:
            try:
                model = ExponentialSmoothing(arpu, trend="add", seasonal=None)
                fit = model.fit()
                forecast_horizon = 52 - len(arpu)
                forecast = fit.forecast(forecast_horizon)
                forecast = np.clip(forecast, arpu[-1], None)

                arpu_full = np.concatenate([arpu, forecast])
                x_full = np.arange(len(arpu_full))
                plt.plot(x_full, arpu_full, label=f'Cohort {int(cohort)}')

            except Exception as e:
                print(f'error {cohort}: {e}')
                continue

plt.xlabel('Week')
plt.ylabel('ARPU')
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize='small')

plt.grid(True)
plt.tight_layout()
plt.show()