import sys
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
GRAMS_GLOBAL = 31.1035
SOVEREIGN = 8
MAX_DAYS = 365

importDuty = 1.15
gst = 1.03
mintingCharges = 1.03
totalRetail = importDuty * gst * mintingCharges

Features = ["yesterdayGold", "yesterdayUSD", "yesterdayNifty",
            "goldChange1", "goldChange5", "USDChange1", "niftyChange1"]

def download(symbol):
    hist = yf.Ticker(symbol).history(period="3y", auto_adjust=True)
    
    if hist.empty:
        raise RuntimeError("Check internet connection")
    
    close = hist["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    
    return close 

def loadData():
    print("Connecting Yahoo Finance")
    raw = pd.DataFrame(
        {
            "goldUSD": download("GC=F"),
            "USDINR": download("INR=X"),
            "nifty": download("^NSEI"),
        }
    )

    raw = raw.sort_index().ffill().dropna()

    raw["goldRaw"] = (raw["goldUSD"] * raw["USDINR"] / GRAMS_GLOBAL) * SOVEREIGN
    raw["goldShowroom"] = raw["goldRaw"] * totalRetail

    return raw[["goldRaw", "goldShowroom", "USDINR", "nifty"]]

def buildFeatures(df):
    f = pd.DataFrame(index=df.index)
    f["yesterdayGold"] = df["goldShowroom"]
    f["yesterdayUSD"] = df["USDINR"]
    f["yesterdayNifty"] = df["nifty"]
    f["goldChange1"] = df["goldShowroom"].pct_change(1)
    f["goldChange5"] = df["goldShowroom"].pct_change(5)
    f["USDChange1"] = df["USDINR"].pct_change(1)
    f["niftyChange1"] = df["nifty"].pct_change(1)
    return f

def makeModel():
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    forest = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=SEED, n_jobs=-1)

    return VotingRegressor([("ridge", ridge), ("rf", forest)], weights=[0.6,0.4])

def forecastRecursive(model, data, steps):
    history = data.tail(10).reset_index(drop=True)
    price = history["goldShowroom"].iloc[-1]

    for _ in range(steps):
        x = buildFeatures(history).iloc[[-1]][Features]
        price = float(model.predict(x)[0])
        next_row = pd.DataFrame(
            {
                "goldShowroom" : [price],
                "USDINR" : [history["USDINR"].iloc[-1]],
                "nifty" : [history["nifty"].iloc[-1]],
            }
        )
        history = pd.concat([history, next_row], ignore_index=True)
    return price

def main():
    try:
        data = loadData()
    except Exception as err:
        print(f"Data loading failed! {err}")
        sys.exit(1)

    latestRaw = data['goldRaw'].iloc[-1]
    latestShowroom = data['goldShowroom'].iloc[-1]

    print(f"Current Raw Price : ₹ {latestRaw:.2f}")
    print(f"Current Showroom Price : ₹ {latestShowroom:.2f}")
    print("(Prices recorded in sovereign)")
    print()

    feats = buildFeatures(data)
    feats["target"] = data["goldShowroom"].shift(-1)
    feats = feats.dropna()
    X, y = feats[Features], feats["target"]

    split = int(len(feats) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = makeModel()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    naive_mae = mean_absolute_error(y_test, X_test["yesterdayGold"])

    print()
    print(f"Mean Absolute Error (MAE) : ₹ {mae:.2f}")
    print(f"R-squared Fit Metric      : ₹ {r2:.4f}")
    print(f"Baseline Naive MAE Score  : ₹ {naive_mae:.2f}")
    print()

    model.fit(X, y)

    print()

    while True:
        choice = input("\nEnter number of days to predict : ").strip()

        try:
            days = int(choice)
        except ValueError:
            print("Enter valid number!")
            continue

        if days == 0:
            print("Exiting\n")
            break
        if days < 0 or days > MAX_DAYS:
            print(f"Target must be between 1 and {MAX_DAYS}")
            continue

        steps = max(1, round(days * 5 / 7))
        predictedShowroom = forecastRecursive(model, data, steps)
        predictedRaw = predictedShowroom / totalRetail

        print()
        print(f" -> Estimated Raw Price :      ₹ {predictedRaw:.2f}")
        print(f" -> Estimated Showroom Price : ₹ {predictedShowroom:.2f}")

if __name__ == "__main__":
    main()