"""
Analyze the UPI Fraud - Feature engineering(the process
of selecting, manipulating, and transforming 
raw data into meaningful inputs)
"""
# Input : transaction_base.csv
# Output: Transaction_features.csv
# Install: pip install pandas numpy scikit-learn scipy

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
print("Perform various manipulation to identify UPI fraud\n ")
df=pd.read_csv("transactions_base.csv",parse_dates=["created_at"])             
#Converts it into pandas DataFrame, convert created)at column to proper date time format
df=df.sort_values("created_at").reset_index(drop=True)

print(f"Rows loaded:    {len(df):,}")
print(f"Fraud rows : {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")


# Category 1- To find how many transaction from same upi id in the last 15 minutes
print("\n To find how many transaction from same upi id in 15 min and 1 hour")
# Sorting transaction process in time order per sender
df=df.sort_values(["sender_vpa","created_at"]).reset_index(drop=True)
"""
Calculate the number of messages or actions a sender has intiated within a specific time window using merge-asof approach
Count how many txns the same sender had in last 5 mmins and last 1 hour(to get sustained suspcious activity)
"""
def rolling_sender_count(df,window_minutes,col_name):
    """ Count transaction per sender in the last N minutes."""
    counts=[]
    for vpa, group in df.groupby("sender_vpa"):
        group=group.sort_values("created_at")           # group all transaction of that sender
        times=group["created_at"].values.astype("int64")//1_000_000 # answers in milliseconds(2023-01-01 10:00 → 1672567200000)
        window_ms=window_minutes*60*100 # convert it to minutes
        cnt=[]
        for i, t in enumerate(times):
            # Count how many of the previous transactions fall wiithin in the window
            # (exclude current row by not taking i)
            cnt.append(np.sum((times[:1]>=t-window_ms)&(times[:1]<t)))
        counts.extend(cnt)
    return counts
def rolling_sender_amount(df, window_minutes, col_name):
    """ Sum transaction amounts per sender in the last N minutes"""
    sums=[]
    for vpa, group in df.groupby("sender_vpa"):
        group=group.sort_values("created_at")
        times=group["created_at"].values.astype("int64")//1_000_000
        amts=group["amount"].values
        window_ms=window_minutes*60*1000
        s=[]
        for i, t in enumerate(times):
            mask=(times[:i]>=t-window_ms)&(times[:i]<t)
            s.append(amts[:i][mask].sum() if mask.any() else 0.0)
        sums.extend(s)
    return sums

# Transaction speed over 15 minute window
df["txn_count_last_15min"]=rolling_sender_count(df,15,"txn_count_last_15min")

# Transaction speed over 1 hour window
df["txn_count_last_1hr"]=rolling_sender_count(df,60,"txn_count_last_1hr")

# amount sum over 1 hour window
df["amount_sum_last_1hr"]  = rolling_sender_amount(df, 60, "amount_sum_last_1hr")

#  Burst flag:sender made 3+ txns over 15 min
df["is_velocity_burst"]=(df["txn_count_last_15min"]>=3).astype(int)

print(f"transaction count in last 15 min - max:{df['txn_count_last_15min'].max()}")
print(f"transaction count last 1 hr  - max:  {df['txn_count_last_1hr'].max()}")
print(f"No. of txns with is 3+ over 15 min   - flagged:   {df['is_velocity_burst'].sum():,}")


# Category 2 - time based risk features
# Fraud check at nigh between 1 pm - 4am, weekends, and around holidays
print("\n Time based risk analysis")
df["hour_of_day"]       =df["created_at"].dt.hour       #Convert the time into 0-23 scale
df["days_of_week"]      =df["created_at"].dt.dayofweek  # convert the date into 0-6 scale(0-mon,6-sun)
df["is_wekend"]         =(df["days_of_week"]>=5).astype(int)

# night window:1am-4am is the highest risk window
df["is_night_txn"]   =df["hour_of_day"].between(1,4).astype(int)

# late evening: 10pm-midnight, also elevated risk
df["is_late_evening"]=df["hour_of_day"].between(22,23).astype(int)

#time risk score:0(day),1(late evening),2(night)
df["time_risk_score"]=df["is_night_txn"]*2 +df["is_late_evening"]

print(f"Night transaction       :{df['is_night_txn'].sum():,}({df['is_night_txn'].mean()*100:.1f}%)")
print(f"Weekend transactions    :{df['is_wekend']}  ({df['is_wekend'].mean()*100:.1f}")

#fraud rate by time - for EDA report
night_fraud=df[df['is_night_txn']==1]["is_fraud"].mean()*100
day_fraud=df[df['is_night_txn']==0]["is_fraud"].mean()*100
print(f"Faud rate - Night:{night_fraud:.2f}%    {day_fraud:.2f}%")

"""
Category 3 - Recipient Signals
Is this sender - recipient pair new or unusual?
"""
#Sort by time to establish"first time seen" chronologically
df=df.sort_values("created_at").reset_index(drop=True)

# track the first time each (sender,recipient) pair appears
pair_first_seen=(
    df.groupby(["sender_vpa","receiver_vpa"])["created_at"]
    .transform("min")       #finds the earliest date in the sender-receiver group
)

df["is_new_recipient"]=(df["created_at"]==pair_first_seen).astype(int)

# how many unique senders does the recipient normally receive from?
recipient_sender_count=(
    df.groupby("receiver_vpa")["sender_vpa"]
    .transform("nunique")           #group by operation to calculate the number of unique value
)
df["recipient_unique_senders"]=recipient_sender_count

# How frequently does this recipient appear overall?
recipient_freq=df.groupby("receiver_vpa")["receiver_vpa"].transform("count")
df["recipient_txn_frequency"]=recipient_freq

# days since the sender first ever transacted with this recipient
df["days_since_first_txn_to_vpa"]=(
    (df["created_at"]-pair_first_seen)
    .dt.total_seconds()/86400
).round(2)

#Flag:brand-new recipient +large amount(high risk combo)
amount_75pct=df["amount"].quantile(0.75)  # top 25% of transactions, idetifies amounts that are significantly higher than average behaviour
df["new_recipient_large_amount"]=(
    (df["is_new_recipient"]==1)&
    (df["amount"]>amount_75pct)
).astype(int)

print(f"New recipient txns      :{df['is_new_recipient'].sum():,}      ({df['is_new_recipient'].mean()*100:.1f}%)")

print(f"    New recipient+large amount: {df['new_recipient_large_amount'].sum():,}")

"""

Category 4 - Amount Anomaly
is this transaction unusual for this specific sender?
"""
print("\n Engineering amount anomaly features........")

# Per sender statistics 
sender_stats=(
    df.groupby("sender_vpa")["amount"]
    .agg(sender_mean="mean",sender_std="std",sender_median="median")
    .reset_index()
)
df=df.merge(sender_stats,on="sender_vpa",how="left")        # left join summary stacks to the original transaction list
df["sender_std"]=df["sender_std"].fillna(1)     # avoid divide by zero

# Z-score- standard score(maps to a standard normal distribution)

df["amount_zscore"]=(
    (df["amount"]-df["sender_mean"])/df["sender_std"]
).round(4)

# ratio:this transaction vs sender's historical average
df["amount_vs_avg_ratio"]=(df["amount"]/df["sender_mean"]).round(4)

# round amount flag:exactly divisble by 100 -> common fraud pattern
df["is_round_amount"]=(df["amount"]%100==0).astype(int)

# very high amount flag: above 95th percentile overall
p95=df["amount"].quantile(0.95)
df["is_high_amount"]=(df["amount"]>p95).astype(int)


# Check both high z score and new recipient
df["zscore_new_recip_combo"]=(
    (df["amount_zscore"]>2.5)&
    (df["is_new_recipient"]==1)
).astype(int)

df.drop(columns=["sender_mean","sender_std","sender_median"],inplace=True)

print(f"Round amount txns   :{df['is_round_amount'].sum():,}    ({df['is_round_amount'].mean()*100:.1f}%)")
print(f"high amount txns    :{df['is_high_amount'].sum():,}")
print(f"High zscore+new VPA:{df["zscore_new_recip_combo"].sum():,}%")

"""
Category 5 - Balance inconsistency feature
does the debit and credit add up?
"""
print("\n Engineering balance inconsistency features....")
# expected post balance=pre balance- amount(for sender)
df["expected _sender_balance"]=df["sender_balance_pre"]-df["amount"]
df["balance_discrepancy"]=(
    df["sender_balance_post"]-df["expected _sender_balance"]
).abs().round(2)

#flag the discrepancies > Rs.1(should be 0)
df["has_balance_error"]=(df["balance_discrepancy"]>1).astype(int)

# how much of the balance was drained in one shot?
# Avoid division by zero ffor  zero-balance accounts
df["balance_drop_ratio"]=np.where(
    df["sender_balance_pre"]>0,
    (df["amount"]/df["sender_balance_pre"]).clip(0,1).round(4),
    0
)

# recipient balance didn't increase when it should have
df["recipient_balance_unchanged"]=(
    (df["receiver_balance_post"]==df["receiver_balance_pre"])&
    (df["amount"]>0)
)

# near-zero balance after transaction(sender drained their account)
df["near_zero_balance_flag"]=(df["sender_balance_post"]<10).astype(int)

print(f"    Balance errors      : {df["has_balance_error"].sum():,}")
print(f"    near zero balance   : {df["near_zero_balance_flag"].sum():,}")
print(f"Recipient unchanged : {df["recipient_balance_unchanged"].sum():,}")

print("04")

"""
Category 6 - device and identity features
identify sudden changes in device or transaction type
"""
print("Engineering device and identity features....")

df= df.sort_values(["sender_vpa","created_at"]).reset_index(drop=True)


# most common devices per seender (their "normal" devices")
sender_common_device=(
       df.groupby("sender_vpa")["device_type"]
       .agg(lambda x :x.mode().iloc[0] if not x.mode().empty else "unknown")
       .reset_index()
       .rename(columns={"device_type":"usual_device"})
)# iloc[0] since dataset can technically have multiple modes, just grabs the first one

df=df.merge(sender_common_device, on="sender_vpa", how="left")
df["is_new_device"]=(df["device_type"]!=df["usual_device"]).astype(int)
df.drop(columns=["usual_device"],inplace=True)

# transaction type shift: sender's usual type vs current type
sender_common_type=(
    df.groupby("sender_vpa")["transaction_type"]
    .agg(lambda x :x.mode().iloc[0] if not x.mode().empty else "P2P")
    .reset_index()
    .rename(columns={"transaction_type":"usual_txn_type"}) 
)

df=df.merge(sender_common_type,on="sender_vpa",how="left")
df["txn_type_shift_flag"]=(
    df["transaction_type"]!=df["usual_txn_type"]
).astype(int)
df.drop(columns=["usual_txn_type"],inplace=True)

# Encode transaction type as numeric for the model
txn_type_map={"P2P":0,"P2M":1,"CASHOUT":2,"CASHIN":3,"DEBIT":4,"OTHER":5}
df["txn_type_encoded"]=df["transaction_type"].map(txn_type_map).fillna(5)

# Encode device type as numeric for the model
device_type_map={"Android":0,"iOS":1,"Web":2,"Feature Phone":3}
df["device_type_encoded"]=df["transaction_type"].map(device_type_map).fillna(-1)


# found both having new device + night transaction
df["new_device_night_combo"]=(
    (df["is_new_device"]==1)&(df["is_night_txn"]==1)
).astype(int)

print(f"New device txns     : {df["is_new_device"].sum():,} ({df["is_new_device"].mean()*100:.1f}%)")
print(f"transaction type shift  : {df["txn_type_shift_flag"].sum():,}")
print(f"New device+night:{df["new_device_night_combo"].sum():,}")


"""
Final composite risk score
a weighted sum of key binary flags
"""
print("\nBuilding composite risk score ...")

df["composite_risk_score"]=(
    df["is_velocity_burst"]          * 25 +
    df["is_new_recipient"]           * 20 +
    df["is_night_txn"]               * 15 +
    df["new_recipient_large_amount"] * 15 +
    df["has_balance_error"]          * 10 +
    df["zscore_new_recip_combo"]     * 10 +
    df["new_device_night_combo"]     * 5
).clip(0,100)


shap_data = {
    'Feature': [
        'Velocity burst',
        'New recipient',
        'Night transaction',
        'New recipient + large amount',
        'Balance error',
        'High z-score + new recipient',
        'New device + night'
    ],
    'Importance': [
        round(df["is_velocity_burst"].mean()          * 25, 4),
        round(df["is_new_recipient"].mean()           * 20, 4),
        round(df["is_night_txn"].mean()               * 15, 4),
        round(df["new_recipient_large_amount"].mean() * 15, 4),
        round(df["has_balance_error"].mean()          * 10, 4),
        round(df["zscore_new_recip_combo"].mean()     * 10, 4),
        round(df["new_device_night_combo"].mean()     *  5, 4),
    ]
}

shap_df = pd.DataFrame(shap_data)
shap_df = shap_df.sort_values("Importance", ascending=False).reset_index(drop=True)

shap_df.to_csv("shap_values.csv", index=False)
print("Saved successfully!")
print(shap_df.to_string(index=False))



# tier labels for the dashboard
def risk_tier(score):
    if score >=50: return "High" 
    if score >=25:return "Medium"
    return "Low"

df["risk_tier"]=df["composite_risk_score"].apply(risk_tier)

print(f"\n risk tier distribution:{df["risk_tier"].value_counts().to_string()}")
print(f"\n fraud rate by risk tier:")
print(
    df.groupby("risk_tier")["is_fraud"]
    .mean().mul(100).round(2)
    .to_string()
)
# Summary : all feature columns
feature_cols=[
    # transaction velocity
    "txn_count_last_15min","txn_count_last_1hr",
    "amount_sum_last_1hr","is_velocity_burst",
    # time
    "hour_of_day","days_of_week","is_wekend",
    "is_night_txn","is_late_evening","time_risk_score",
    #recipient
    "is_new_recipient","recipient_unique_senders",
    "recipient_txn_frequency","days_since_first_txn_to_vpa",
    "new_recipient_large_amount",
    #  amount
    "amount_zscore", "amount_vs_avg_ratio",
    "is_round_amount", "is_high_amount", "zscore_new_recip_combo",
    # Balance
    "balance_discrepancy", "has_balance_error",
    "balance_drop_ratio", "near_zero_balance_flag",
    "recipient_balance_unchanged",
    # Device & Identity
    "is_new_device", "txn_type_shift_flag",
    "txn_type_encoded", "device_type_encoded",
    "new_device_night_combo",
    # Composite
    "composite_risk_score", "risk_tier",
]

print(f"\n Total features engineered: {len(feature_cols)}")

#Save Output
output_file="transaction_features.csv"
df.to_csv(output_file, index=False)
print(f"\n Saved to '{output_file}' - shape: {df.shape}")


# quick correlation check : feature vs fraud label
print("\n Top 10  features by correleation with is_fraud:")
numeric_cols= [c for c in feature_cols if c not in ["risk_tier"]]
corr=(
    df[numeric_cols+["is_fraud"]]
    .corr()["is_fraud"]
    .drop("is_fraud")
    .abs()
    .sort_values(ascending=False)
    .head(10)
)

for feat,val in corr.items():
    bar="█" * int(val*40)
    print(f"{feat:<35}{val:.4f}{bar}")


print("Phase 2 complete!  →  Use transactions_features.csv in Phase 3")




