"""
Smartphone dataset generator.

Produces a realistic 1,247-record dataset with statistically grounded
distributions — price ranges, spec correlations, and brand positioning
reflect real 2023-2025 smartphone market conditions.

Usage:
    python data/generate_dataset.py
    python data/generate_dataset.py --records 2000 --output data/phones_large.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Brand configuration ───────────────────────────────────────────────────────
BRAND_CONFIG = {
    "Apple": {
        "prestige": 10, "market_share": 0.18,
        "price_range": (69999, 159999),
        "models": ["iPhone 13", "iPhone 14", "iPhone 14 Pro", "iPhone 14 Pro Max",
                   "iPhone 15", "iPhone 15 Plus", "iPhone 15 Pro", "iPhone 15 Pro Max",
                   "iPhone SE (2022)", "iPhone 16", "iPhone 16 Pro", "iPhone 16 Pro Max"],
        "chipsets": ["A15 Bionic", "A16 Bionic", "A17 Pro", "A18", "A18 Pro"],
        "ram_range": (4, 8), "storage_range": (128, 1024),
        "rating_mean": 4.5, "rating_std": 0.2,
        "battery_range": (3279, 4685), "charge_range": (20, 30),
    },
    "Samsung": {
        "prestige": 9, "market_share": 0.22,
        "price_range": (12999, 149999),
        "models": ["Galaxy A14", "Galaxy A34", "Galaxy A54", "Galaxy A73",
                   "Galaxy S23", "Galaxy S23+", "Galaxy S23 Ultra",
                   "Galaxy S24", "Galaxy S24+", "Galaxy S24 Ultra",
                   "Galaxy Z Flip5", "Galaxy Z Fold5", "Galaxy M34", "Galaxy M54"],
        "chipsets": ["Exynos 1330", "Snapdragon 888", "Snapdragon 8 Gen 1",
                     "Snapdragon 8 Gen 2", "Snapdragon 8 Gen 3", "Exynos 2400"],
        "ram_range": (4, 12), "storage_range": (64, 512),
        "rating_mean": 4.2, "rating_std": 0.3,
        "battery_range": (4000, 5000), "charge_range": (25, 65),
    },
    "OnePlus": {
        "prestige": 7, "market_share": 0.06,
        "price_range": (15999, 89999),
        "models": ["OnePlus Nord CE3 Lite", "OnePlus Nord CE3", "OnePlus Nord 3",
                   "OnePlus 11", "OnePlus 11R", "OnePlus 12", "OnePlus 12R",
                   "OnePlus Open", "OnePlus Nord 4"],
        "chipsets": ["Snapdragon 782G", "Snapdragon 8 Gen 1", "Snapdragon 8 Gen 2",
                     "Snapdragon 8 Gen 3", "Dimensity 9300"],
        "ram_range": (6, 16), "storage_range": (128, 512),
        "rating_mean": 4.3, "rating_std": 0.25,
        "battery_range": (4500, 5400), "charge_range": (67, 100),
    },
    "Google": {
        "prestige": 8, "market_share": 0.04,
        "price_range": (37999, 109999),
        "models": ["Pixel 7a", "Pixel 7", "Pixel 7 Pro", "Pixel 8",
                   "Pixel 8 Pro", "Pixel 8a", "Pixel 9", "Pixel 9 Pro",
                   "Pixel 9 Pro XL", "Pixel Fold"],
        "chipsets": ["Google Tensor G2", "Google Tensor G3", "Google Tensor G4"],
        "ram_range": (8, 16), "storage_range": (128, 512),
        "rating_mean": 4.4, "rating_std": 0.2,
        "battery_range": (4492, 5050), "charge_range": (18, 30),
    },
    "Xiaomi": {
        "prestige": 6, "market_share": 0.12,
        "price_range": (8999, 89999),
        "models": ["Redmi Note 12", "Redmi Note 13", "Redmi Note 13 Pro",
                   "Mi 13", "Mi 13 Pro", "Mi 14", "Mi 14 Pro",
                   "Xiaomi 13T", "Xiaomi 14T Pro", "POCO F5", "POCO X5 Pro"],
        "chipsets": ["Snapdragon 7s Gen 2", "Snapdragon 8 Gen 1", "Snapdragon 8 Gen 2",
                     "Snapdragon 8 Gen 3", "Dimensity 7200", "Dimensity 9200"],
        "ram_range": (6, 16), "storage_range": (128, 512),
        "rating_mean": 4.1, "rating_std": 0.35,
        "battery_range": (4500, 5000), "charge_range": (33, 120),
    },
    "Realme": {
        "prestige": 5, "market_share": 0.08,
        "price_range": (8999, 44999),
        "models": ["Realme 11", "Realme 11 Pro", "Realme 11 Pro+",
                   "Realme 12", "Realme 12 Pro", "Realme GT 5 Pro",
                   "Narzo 60", "Narzo 70 Pro"],
        "chipsets": ["Snapdragon 6 Gen 1", "Snapdragon 7s Gen 2", "Dimensity 7050",
                     "Dimensity 8200", "Snapdragon 8 Gen 2"],
        "ram_range": (6, 12), "storage_range": (128, 256),
        "rating_mean": 4.0, "rating_std": 0.35,
        "battery_range": (4500, 5000), "charge_range": (33, 67),
    },
    "Vivo": {
        "prestige": 5, "market_share": 0.07,
        "price_range": (12999, 89999),
        "models": ["Vivo V27", "Vivo V29", "Vivo V29 Pro", "Vivo V30",
                   "Vivo V30 Pro", "Vivo X100", "Vivo X100 Pro",
                   "iQOO 11", "iQOO 12", "iQOO Neo 9 Pro"],
        "chipsets": ["Snapdragon 7 Gen 1", "Snapdragon 8 Gen 2", "Snapdragon 8 Gen 3",
                     "Dimensity 9200", "Dimensity 9300"],
        "ram_range": (8, 16), "storage_range": (128, 512),
        "rating_mean": 4.1, "rating_std": 0.3,
        "battery_range": (4500, 5000), "charge_range": (44, 120),
    },
    "Nothing": {
        "prestige": 7, "market_share": 0.02,
        "price_range": (19999, 54999),
        "models": ["Nothing Phone (1)", "Nothing Phone (2)", "Nothing Phone (2a)",
                   "Nothing Phone (2a) Plus"],
        "chipsets": ["Snapdragon 778G+", "Snapdragon 8+ Gen 1", "Dimensity 7200 Pro"],
        "ram_range": (8, 12), "storage_range": (128, 256),
        "rating_mean": 4.3, "rating_std": 0.2,
        "battery_range": (4500, 4700), "charge_range": (45, 50),
    },
}

# ── Price tier thresholds (INR) ───────────────────────────────────────────────
PRICE_TIERS = [
    (0,       15000,  "Budget"),
    (15000,   30000,  "Mid-range"),
    (30000,   60000,  "Premium"),
    (60000,   100000, "Ultra-premium"),
    (100000,  999999, "Flagship"),
]

def assign_price_tier(price: float) -> str:
    for low, high, label in PRICE_TIERS:
        if low <= price < high:
            return label
    return "Flagship"

def compute_value_score(row: pd.Series) -> float:
    """
    Composite value score (0-10) combining:
    - Spec density relative to price
    - User satisfaction (rating weighted by review volume)
    - Price retention
    """
    # Spec score (normalised sum of key specs)
    spec_raw = (
        row["ram_gb"] * 0.15 +
        np.log1p(row["storage_gb"]) * 1.2 +
        row["main_cam_mp"] * 0.05 +
        row["battery_mah"] / 500 * 0.3 +
        row["refresh_rate_hz"] / 30 * 0.4 +
        (1 if row["has_5g"] else 0) * 0.8 +
        (1 if row["nfc"] else 0) * 0.3
    )
    spec_score = min(spec_raw / 25, 1.0) * 4  # max 4 points

    # Satisfaction score
    review_confidence = min(np.log1p(row["review_count"]) / np.log1p(50000), 1.0)
    satisfaction = (row["user_rating"] / 5.0) * review_confidence * 3  # max 3 points

    # Value for money (inverse price relative to tier mean)
    price_norm = row["launch_price_inr"] / 100000
    value_for_money = max(0, (1 - price_norm * 0.5)) * 3  # max 3 points

    return round(min(spec_score + satisfaction + value_for_money, 10.0), 2)


def generate_dataset(n_records: int = 1247, seed: int = 42) -> pd.DataFrame:
    """Generate the full smartphone dataset."""
    np.random.seed(seed)
    records = []

    # Distribute records proportionally by market share
    brand_names = list(BRAND_CONFIG.keys())
    shares = np.array([BRAND_CONFIG[b]["market_share"] for b in brand_names])
    shares = shares / shares.sum()
    brand_counts = np.round(shares * n_records).astype(int)
    brand_counts[-1] = n_records - brand_counts[:-1].sum()  # fix rounding

    for brand, count in zip(brand_names, brand_counts):
        cfg = BRAND_CONFIG[brand]
        
        for _ in range(count):
            model = np.random.choice(cfg["models"])
            chipset = np.random.choice(cfg["chipsets"])
            release_year = np.random.choice([2022, 2023, 2024, 2025], p=[0.15, 0.35, 0.35, 0.15])
            release_month = np.random.randint(1, 13)
            
            # Specs — correlated with price tier
            launch_price = np.random.uniform(*cfg["price_range"])
            price_tier = assign_price_tier(launch_price)
            
            # RAM/storage scale with price
            price_pct = (launch_price - cfg["price_range"][0]) / (cfg["price_range"][1] - cfg["price_range"][0] + 1)
            ram_gb = int(np.clip(
                cfg["ram_range"][0] + price_pct * (cfg["ram_range"][1] - cfg["ram_range"][0]) + np.random.normal(0, 1),
                cfg["ram_range"][0], cfg["ram_range"][1]
            ))
            # RAM must be power of 2
            ram_gb = max(4, 2 ** round(np.log2(ram_gb)))

            storage_gb = int(np.random.choice(
                [64, 128, 256, 512, 1024],
                p=np.array([0.05, 0.40, 0.35, 0.15, 0.05]) if brand != "Apple" else [0.0, 0.40, 0.35, 0.20, 0.05]
            ))

            # Camera: flagship = more MPs
            main_cam_mp = int(np.random.choice(
                [12, 48, 50, 64, 108, 200],
                p=[0.15, 0.25, 0.30, 0.15, 0.10, 0.05] if price_tier != "Budget" else [0.30, 0.40, 0.20, 0.07, 0.03, 0.0]
            ))
            has_ultrawide = price_tier not in ["Budget"]
            ultrawide_mp = int(np.random.choice([8, 12, 13, 50]) if has_ultrawide else 0)
            has_telephoto = price_tier in ["Premium", "Ultra-premium", "Flagship"]
            telephoto_mp = int(np.random.choice([5, 10, 12, 50]) if has_telephoto else 0)
            front_cam_mp = int(np.random.choice([8, 12, 16, 32], p=[0.15, 0.35, 0.30, 0.20]))

            # Display
            screen_size = round(np.random.uniform(6.1, 6.9) if price_tier != "Flagship" else np.random.uniform(6.5, 7.2), 1)
            refresh_rate = int(np.random.choice(
                [60, 90, 120, 144, 165],
                p=[0.15, 0.20, 0.45, 0.12, 0.08] if price_tier not in ["Budget"] else [0.50, 0.35, 0.15, 0.0, 0.0]
            ))
            display_type = np.random.choice(
                ["AMOLED", "Super AMOLED", "OLED", "IPS LCD"],
                p=[0.40, 0.25, 0.25, 0.10] if price_tier not in ["Budget"] else [0.10, 0.10, 0.05, 0.75]
            )
            resolution = np.random.choice(["HD+", "FHD+", "QHD+", "2K"],
                p=[0.15, 0.55, 0.20, 0.10] if price_tier not in ["Budget"] else [0.40, 0.55, 0.05, 0.0])

            # Battery & charging
            battery_mah = int(np.random.uniform(*cfg["battery_range"]))
            fast_charge_w = int(np.random.uniform(*cfg["charge_range"]))
            wireless_charge = brand in ["Apple", "Samsung", "Google"] and price_tier not in ["Budget", "Mid-range"]

            # Connectivity
            has_5g = launch_price > 15000 or np.random.random() < 0.15
            nfc = launch_price > 12000 or brand in ["Apple", "Google", "Samsung"]
            wifi6 = launch_price > 25000
            usb_type = "USB-C" if brand != "Apple" else "Lightning" if release_year < 2023 else "USB-C"

            # Market data
            days_since_launch = max(0, int(np.random.normal(400, 180)))
            # Price drops over time — faster for Android, slower for Apple
            depreciation_rate = 0.0005 if brand == "Apple" else np.random.uniform(0.0007, 0.0015)
            current_price = max(launch_price * 0.4, launch_price * np.exp(-depreciation_rate * days_since_launch))
            price_drop_pct = round((1 - current_price / launch_price) * 100, 1)

            # Ratings — correlated with brand prestige and slightly with price
            base_rating = cfg["rating_mean"] + price_pct * 0.15
            user_rating = round(np.clip(np.random.normal(base_rating, cfg["rating_std"]), 2.5, 5.0), 1)
            review_count = int(np.clip(
                np.random.lognormal(mean=np.log(5000 * cfg["market_share"]), sigma=1.2),
                50, 500000
            ))

            record = {
                "brand": brand,
                "model": model,
                "chipset": chipset,
                "release_year": release_year,
                "release_month": release_month,
                "ram_gb": ram_gb,
                "storage_gb": storage_gb,
                "main_cam_mp": main_cam_mp,
                "ultrawide_mp": ultrawide_mp,
                "telephoto_mp": telephoto_mp,
                "front_cam_mp": front_cam_mp,
                "screen_size_in": screen_size,
                "resolution": resolution,
                "refresh_rate_hz": refresh_rate,
                "display_type": display_type,
                "battery_mah": battery_mah,
                "fast_charge_w": fast_charge_w,
                "wireless_charge": wireless_charge,
                "has_5g": has_5g,
                "nfc": nfc,
                "wifi6": wifi6,
                "usb_type": usb_type,
                "launch_price_inr": round(launch_price),
                "current_price_inr": round(current_price),
                "price_drop_pct": price_drop_pct,
                "days_since_launch": days_since_launch,
                "user_rating": user_rating,
                "review_count": review_count,
                "price_tier": price_tier,
                "brand_prestige": cfg["prestige"],
            }
            records.append(record)

    df = pd.DataFrame(records)

    # Compute value score
    df["value_score"] = df.apply(compute_value_score, axis=1)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.index.name = "phone_id"

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate smartphone dataset")
    parser.add_argument("--records", type=int, default=1247, help="Number of records")
    parser.add_argument("--output", type=str, default="data/phones_dataset.csv", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {args.records} smartphone records...")
    df = generate_dataset(args.records, args.seed)
    df.to_csv(args.output)
    print(f"Saved {len(df)} records to {args.output}")
    print(f"\nBrand distribution:\n{df.groupby('brand')['model'].count().sort_values(ascending=False)}")
    print(f"\nPrice tier distribution:\n{df['price_tier'].value_counts()}")
    print(f"\nValue score stats:\n{df['value_score'].describe().round(2)}")
