""" 
Project UPI Fraud Analyzer -  Data loading and Setup
Install pandas sqlalchemy pymysql in terminal by running
  -- pip install pandas sqlalchemy pymysql
  1. pip install pandas
  2. pip install sqlalchemy
  3. pip install pymysql
    
Before running the scripts
1. Save the folder as 'PS_20174392719_1491204439457_log.csv' in the same folder as this script
2. Create upi_fraud_db (database) and upi_transaction(table)
3. update DB_USER and DB_PASSWORD below with your credentials

 """

import pandas as pd
from sqlalchemy import create_engine, text
from datetime  import datetime,timedelta
from urllib.parse import quote_plus
import random

# Configure the sql
DB_USER = "root"
DB_PASSWORD = quote_plus("Your password")
DB_HOST="127.0.0.1"
DB_PORT="3306"
DB_NAME="upi_fraud_db"
PAYSIM_FILE="PS_20174392719_1491204439457_log.csv"

ROW_LIMIT=200_000 #Starting with 200,000 for faster testig during development

"""
Load PaySim CSV dataset
"""
print("STEP 1: loading Paysim dataset.....")

df=pd.read_csv(PAYSIM_FILE,nrows=ROW_LIMIT)
print(f"Rows loaded         : {len(df):,}")
print(f"Columns             : {list(df.columns)}")
print(f"Fraud rows          : {df['isFraud'].sum():,}    ({df['isFraud'].mean()*100:.2f}%)")
print(f"Non-Frauds rows     : {(df['isFraud']==0).sum()}")

"""
Map PaySim to UPI schema
"""
print("\n Step 2: Mapping PaySim columns with UPI schema\n")
# map PaySim transaction types to UPI eqiuvalents

TYPE_MAP ={
    "TRANSFER"  : "P2P",
    "CASH_OUT"  : "CASHOUT",
    "PAYMENT"   : "P2M",
    "CASH_IN"   : "CASHIN",
    "DEBIT"     : "DEBIT",

}
df["transaction_type"]=df["type"].map(TYPE_MAP).fillna("OTHER")

# Rename columns to match your SQL table
df=df.rename(columns={
    "nameOrig"    : "sender_vpa",
    "nameDest"    : "receiver_vpa",
    "oldbalanceOrg" : "sender_balance_pre",
    "newbalanceOrig"  : "sender_balance_post",
    "oldbalanceDest"    : "receiver_balance_pre",
    "newbalanceDest"  : "receiver_balance_post",
    "isFraud"    : "is_fraud",
})

#Create unique transaction id
df["transaction_id"]=["TXN"+str(i).zfill(9) for i in range(len(df))]

# Derive hour of day from 'step'(each step = 1 hour in PaySim)
df["hour_of_day"]=df["step"] %24

# simulate a created_at datetime(starting from jan 1 2023)
base_date=datetime(2023,1,1)
df["created_at"]=df["step"].apply(
    lambda s: base_date + timedelta(hours = int(s)))

""" 
Simulate device_type(This block is adding fake-simulated 
features to use later for analysis or modelling)
"""
random.seed(42)
device_types=["Android","iOS","Web","Feature Phone"]
device_weights=[0.55,0.30,0.10,0.05]    #Possibilty - Android (55%),iOS(30%) etc.
df["device_type"]=random.choices(device_types,weights=device_weights,k=len(df))
df["is_new_recipient"]=0

print("Column Mapping complete.")

#-------------------------------------------------
# Step 3: Clean and Validate
#-------------------------------------------------
before=len(df)
print("\n Step 3: Cleaning data....")
df=df.dropna(subset=["amount","sender_vpa","receiver_vpa"])
print(f"   Rows dropped (nulls) : {before-len(df):,}")

# remove zero amount transaction
df=df[df["amount"]>0]
print(f"  Rows after zero-amount filter : {len(df):,}")

# This blocks is limiting extreme transaction amounts so they don't distort your analysis
p999=df["amount"].quantile(0.999)
df["amount"]=df["amount"].clip(upper=p999)
print(f"  Amount capped at 99.9th pct   :₹{p999:,.2f}")
"""
`capped value that does not changes the actual value(used as testing )
df["amount_capped"] = df["amount"].clip(upper=p999)
print(f"  Amount capped at 99.9th pct   :₹{p999:,.2f}")
"""

# Trim to final columns matching your SQL Table
FINAL_COLS=[
    "transaction_id", "step", "transaction_type", "amount",
    "sender_vpa", "receiver_vpa",
    "sender_balance_pre", "sender_balance_post",
    "receiver_balance_pre", "receiver_balance_post",
    "is_fraud", "device_type", "is_new_recipient", "created_at"
]
df = df[FINAL_COLS]


print(f"\n  Final shape : {df.shape}")
print(f"  DTypes: \n{df.dtypes}")



"""
Step: Quick EDA(Explorarty data analysis) summary before loading
"""
print("\n Step 4: Quick summary before DB load.....")
print(f"\n  Transaction type breakdown:")
print(df["transaction_id"].value_counts().to_string())
print(f"\n  Fraud by transaction type:")
print(df.groupby("transaction_type")["is_fraud"].mean().mul(100).round(2).to_string())
print(f"\n  Amount Stats:")
print(df["amount"].describe().round(2).to_string())



"""
Step 5: Load into MySQL
"""
print("\n Step 5: Connecting to MySQL and loading data.....")

connection_string=(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

try:
    engine=create_engine(connection_string,echo=False)

    # Test connecction
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("     MySQL connection sucessful.")

    # Load in chunks to avoid memory issues
    total_chunks=(len(df)//10_000)+1
    print(f"Loading {len(df):,} rows in {total_chunks} chunks of {10_000:,}....")

    df.to_sql (
        name            =   "upi_transactions",
        con             =   engine,
        if_exists       =   "append",
        index           =   False,
        chunksize       =   10_000,
        method          =   "multi",
        )

    print("\n Data loaded succesfully!")
    # Verify row count in DB
    with engine.connect() as conn:
        result=conn.execute(text("SELECT COUNT(*) from upi_transactions"))
        db_count=result.scalar()
    print(f"Rows now in upi_transactions table: {db_count:,}")
except Exception as e:
    print(f"\n Error {e}")
    print("Check your DB_USER, DB_PASSWORD, and MysqL is running.")

"""
Step 6: Save clean CSV for phase 2
"""
output_file = "transactions_base.csv"
df.to_csv(output_file, index=False)
print(f"\nSTEP 6: Clean dataset saved to '{output_file}'")
print("        Use this file as input for Phase 2 ")
 
