"""
Roop Ji Sweets - Synthetic Demand Forecasting Dataset Generator (fixed)
=======================================================================
Only change from the original: the date span now uses START->END so every
row falls inside the festival/seasonal calendar (2023-2025). The original
periods=10000 spilled ~27 years of signal-less rows past 2025.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# 1. Config
START = "2023-01-01"
END   = "2025-12-31"
OUTLETS = ["Station Road"]
OUTLET_SCALE = {"Station Road": 1.0}
YEARLY_GROWTH = 0.08

# 2. Products
PRODUCTS = [
    ("Rasgulla",         "Sweet",     45,  280, "milk_summer_dip", 0.7),
    ("Gulab Jamun",      "Sweet",     50,  300, "mild_summer_dip", 0.8),
    ("Kaju Katli",       "Sweet",     25,  900, "steady",          1.0),
    ("Besan Barfi",      "Sweet",     30,  420, "steady",          0.9),
    ("Motichoor Laddu",  "Sweet",     28,  380, "steady",          0.95),
    ("Ghewar",           "Sweet",      9,  500, "monsoon_teej",    0.5),
    ("Gajak",            "Sweet",     11,  320, "winter",          0.4),
    ("Soan Papdi",       "Sweet",     18,  300, "steady",          0.9),
    ("Bikaneri Bhujia",  "Namkeen",   80,  360, "steady",          0.6),
    ("Aloo Bhujia",      "Namkeen",   40,  300, "steady",          0.5),
]

# 3. Festival calendar
FESTIVALS = [
    ("Makar Sankranti", "2023-01-14", 0.55),
    ("Holi",            "2023-03-08", 0.75),
    ("Gangaur",         "2023-03-24", 0.45),
    ("Akshaya Tritiya", "2023-04-22", 0.50),
    ("Hariyali Teej",   "2023-08-19", 0.55),
    ("Raksha Bandhan",  "2023-08-30", 0.80),
    ("Janmashtami",     "2023-09-06", 0.50),
    ("Dussehra",        "2023-10-24", 0.60),
    ("Karva Chauth",    "2023-11-01", 0.45),
    ("Dhanteras",       "2023-11-10", 0.85),
    ("Diwali",          "2023-11-12", 1.00),
    ("Bhai Dooj",       "2023-11-14", 0.65),
    ("Christmas/NY",    "2023-12-25", 0.40),
    ("Makar Sankranti", "2024-01-14", 0.55),
    ("Holi",            "2024-03-25", 0.75),
    ("Gangaur",         "2024-04-11", 0.45),
    ("Akshaya Tritiya", "2024-05-10", 0.50),
    ("Hariyali Teej",   "2024-08-07", 0.55),
    ("Raksha Bandhan",  "2024-08-19", 0.80),
    ("Janmashtami",     "2024-08-26", 0.50),
    ("Dussehra",        "2024-10-12", 0.60),
    ("Karva Chauth",    "2024-10-20", 0.45),
    ("Dhanteras",       "2024-10-29", 0.85),
    ("Diwali",          "2024-11-01", 1.00),
    ("Bhai Dooj",       "2024-11-03", 0.65),
    ("Christmas/NY",    "2024-12-25", 0.40),
    ("Makar Sankranti", "2025-01-14", 0.55),
    ("Holi",            "2025-03-14", 0.75),
    ("Gangaur",         "2025-03-31", 0.45),
    ("Akshaya Tritiya", "2025-04-30", 0.50),
    ("Hariyali Teej",   "2025-07-27", 0.55),
    ("Raksha Bandhan",  "2025-08-09", 0.80),
    ("Janmashtami",     "2025-08-16", 0.50),
    ("Dussehra",        "2025-10-02", 0.60),
    ("Karva Chauth",    "2025-10-10", 0.45),
    ("Dhanteras",       "2025-10-18", 0.85),
    ("Diwali",          "2025-10-20", 1.00),
    ("Bhai Dooj",       "2025-10-23", 0.65),
    ("Christmas/NY",    "2025-12-25", 0.40),
]
FEST_DF = pd.DataFrame(FESTIVALS, columns=["festival", "date", "intensity"])
FEST_DF["date"] = pd.to_datetime(FEST_DF["date"])
MAJOR = FEST_DF[FEST_DF["intensity"] >= 0.75]["date"].tolist()

# 4. Helper curves
TEMP_BY_MONTH = {1:16, 2:20, 3:26, 4:33, 5:38, 6:40,
                 7:36, 8:34, 9:33, 10:29, 11:23, 12:17}

def seasonal_multiplier(profile, month):
    if profile == "steady":
        return 1.0
    if profile == "mild_summer_dip":
        return 0.85 if month in (5, 6) else 1.0
    if profile == "milk_summer_dip":
        return 0.6 if month in (5, 6) else (0.8 if month == 4 else 1.0)
    if profile == "monsoon_teej":
        if month in (7, 8):   return 6.0
        if month in (6, 9):   return 1.8
        return 0.15
    if profile == "winter":
        if month in (12, 1):  return 4.0
        if month in (11, 2):  return 2.5
        if month in (3, 10):  return 0.8
        return 0.12
    return 1.0

def festival_multiplier(date, fest_sens):
    mult = 1.0
    name = None
    on_day = 0
    for _, row in FEST_DF.iterrows():
        delta = (row["date"] - date).days
        if delta == 0:
            mult *= 1.0 + row["intensity"] * 9.0 * fest_sens
            name = row["festival"]
            on_day = 1
        elif 1 <= delta <= 6:
            ramp = (7 - delta) / 7.0
            mult *= 1.0 + row["intensity"] * 4.5 * fest_sens * ramp
            if name is None:
                name = f"pre-{row['festival']}"
    return mult, name, on_day

def wedding_uplift(date, category, name):
    m = date.month
    in_season = m in (11, 12, 1, 2) or (m in (4, 5, 6, 7))
    if not in_season:
        return 1.0, 0
    bulk_items = {"Bikaneri Bhujia", "Papad", "Motichoor Laddu",
                  "Besan Barfi", "Soan Papdi", "Kaju Katli"}
    if name in bulk_items:
        return 1.35, 1
    return 1.08, 1

# 5. Generate  --- FIX: full START->END span, not periods=10000
dates = pd.date_range(START, END, freq="D")
start_ord = dates[0].toordinal()

rows = []
for date in dates:
    month = date.month
    dow = date.dayofweek
    weekday_factor = {0:0.92, 1:0.90, 2:0.95, 3:0.98,
                      4:1.10, 5:1.30, 6:1.40}[dow]
    temp = TEMP_BY_MONTH[month] + rng.normal(0, 2.5)
    years_elapsed = (date.toordinal() - start_ord) / 365.25
    trend = (1 + YEARLY_GROWTH) ** years_elapsed
    future = [(f - date).days for f in MAJOR if (f - date).days >= 0]
    days_to_fest = min(future) if future else 60
    days_to_fest = min(days_to_fest, 60)

    for name, cat, base, price, profile, fest_sens in PRODUCTS:
        seas = seasonal_multiplier(profile, month)
        fmult, fname, is_fest = festival_multiplier(date, fest_sens)
        wed_mult, is_wed = wedding_uplift(date, cat, name)

        is_promo = int(rng.random() < 0.04)
        promo_mult = 1.25 if is_promo else 1.0
        eff_price = round(price * (0.90 if is_promo else 1.0))

        for outlet in OUTLETS:
            lam = (base * OUTLET_SCALE[outlet] * weekday_factor * seas
                   * fmult * wed_mult * trend * promo_mult)
            lam = max(lam, 0.1)
            units = int(rng.poisson(lam))

            rows.append({
                "date": date.date().isoformat(),
                "outlet": outlet,
                "product": name,
                "category": cat,
                "units_sold": units,
                "unit_price": eff_price,
                "revenue": units * eff_price,
                "day_of_week": date.day_name(),
                "is_weekend": int(dow >= 5),
                "is_festival": is_fest,
                "festival_name": fname if fname else "None",
                "days_to_festival": days_to_fest,
                "is_wedding_season": is_wed,
                "is_promotion": is_promo,
                "temperature_c": round(temp, 1),
            })

df = pd.DataFrame(rows)
out = "dataset/roopji_sweets_daily_sales.csv"
df.to_csv(out, index=False)

print("Rows:", len(df))
print("Date range:", df.date.min(), "->", df.date.max())
print("Products:", df["product"].nunique(), "| Outlets:", df["outlet"].nunique())
print("Days:", df["date"].nunique())
print("\nAvg units/day by product:")
print(df.groupby("product")["units_sold"].mean().round(1).sort_values(ascending=False))

print("\nDiwali/Dhanteras vs normal (Kaju Katli):")
sub = df[(df["product"] == "Kaju Katli") & (df["outlet"] == "Station Road")]
diwali = sub[sub["festival_name"].str.contains("Diwali|Dhanteras", regex=True)]["units_sold"].mean()
normal = sub[sub["festival_name"] == "None"]["units_sold"].mean()
print(f"  Diwali/Dhanteras avg: {diwali:.0f}  |  Normal avg: {normal:.0f}  ({diwali/normal:.1f}x)")

print("\nGhewar by month (peak Jul-Aug):")
g = df[df["product"] == "Ghewar"].copy()
g["month"] = pd.to_datetime(g["date"]).dt.month
print(g.groupby("month")["units_sold"].mean().round(1).to_dict())

print("\nGajak by month (peak Dec-Jan):")
gj = df[df["product"] == "Gajak"].copy()
gj["month"] = pd.to_datetime(gj["date"]).dt.month
print(gj.groupby("month")["units_sold"].mean().round(1).to_dict())