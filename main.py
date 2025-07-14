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

checkpoint_weeks = [12, 24, 52]
cohort_labels = []
checkpoint_data = {week: [] for week in checkpoint_weeks}

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


                cohort_labels.append(str(int(cohort)))
                for week in checkpoint_weeks:
                    idx = week - 1
                    if idx < len(arpu_full):
                        checkpoint_data[week].append(arpu_full[idx])
                    else:
                        checkpoint_data[week].append(np.nan)

            except Exception as e:
                print(f'error {cohort}: {e}')
                continue

plt.xlabel('Week')
plt.ylabel('ARPU')
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize='small')

plt.grid(True)
plt.tight_layout()
plt.show()


checkpoint_df = pd.DataFrame({'Cohort': cohort_labels})
for week in checkpoint_weeks:
    checkpoint_df[f'ARPU@{week}w'] = checkpoint_data[week]


fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

for i, week in enumerate(checkpoint_weeks):
    ax = axes[i]
    values = checkpoint_df[f'ARPU@{week}w']
    ax.bar(checkpoint_df['Cohort'], values, color='skyblue')
    ax.set_ylabel(f'ARPU @ {week} week')
    ax.set_title(f'ARPU after {week // 4} months')

axes[-1].set_xlabel('Cohort')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
