import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data.csv")

# -------- NORMALIZATION --------
# group by user and normalize spending
df['amount_norm'] = df.groupby('user_id')['amount'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# -------- MODEL --------
model = IsolationForest(contamination=0.2, random_state=42)
df['anomaly'] = model.fit_predict(df[['amount_norm']])

# Convert -1 to anomaly
df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)

print("\nDetected Anomalies:\n")
print(df[df['anomaly'] == 1])

# -------- VISUALIZATION --------
plt.scatter(df['time'], df['amount'], c=df['anomaly'])
plt.xlabel("Time")
plt.ylabel("Amount")
plt.title("Anomaly Detection")
plt.show()
print("\nExplanation:")
for index, row in df[df['anomaly'] == 1].iterrows():
    print(f"User {row['user_id']} spent unusually high amount {row['amount']}")