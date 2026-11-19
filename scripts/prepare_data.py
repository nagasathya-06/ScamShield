"""
Dataset Preparation and Preprocessing Script.
Merges UCI SMS benchmark dataset with custom Indian scam/banking corpus,
creates stratified train/val/test splits, and generates the adversarial obfuscation test suite.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import (
    SMS_RAW_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    OBFUSCATED_TEST_PATH,
    FULL_DATASET_PATH,
    RANDOM_STATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)
from data.custom_indian_corpus import (
    INDIAN_SCAMS,
    INDIAN_GENUINE,
    generate_obfuscated_dataset,
    obfuscate_text
)


def build_and_prepare_dataset():
    print("=" * 60)
    print("Step 1: Loading UCI SMS Spam Benchmark Dataset...")
    if not SMS_RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {SMS_RAW_PATH}")

    df_uci = pd.read_csv(SMS_RAW_PATH, sep='\t', header=None, names=['raw_label', 'message_text'])
    # Map raw_label: 'ham' -> 0 (Safe), 'spam' -> 1 (Scam)
    df_uci['label'] = df_uci['raw_label'].map({'ham': 0, 'spam': 1})
    df_uci['language_style'] = 'English'
    df_uci['is_obfuscated'] = 0
    df_uci = df_uci[['message_text', 'label', 'language_style', 'is_obfuscated']]
    print(f"Loaded {len(df_uci)} messages from UCI dataset (Safe: {sum(df_uci['label'] == 0)}, Scam: {sum(df_uci['label'] == 1)})")

    # Step 2: Incorporate Custom Indian Scam and Genuine Banking Data
    print("=" * 60)
    print("Step 2: Incorporating Custom Indian UPI, Banking, and Hinglish Messages...")
    custom_records = []

    # Add custom Indian scams (label 1)
    for scam in INDIAN_SCAMS:
        custom_records.append({
            "message_text": scam["message_text"],
            "label": 1,
            "language_style": scam["language_style"],
            "is_obfuscated": 0
        })

    # Add custom Indian genuine banking & service alerts (label 0)
    for gen in INDIAN_GENUINE:
        custom_records.append({
            "message_text": gen["message_text"],
            "label": 0,
            "language_style": gen["language_style"],
            "is_obfuscated": 0
        })

    # Augment realistic permutations of Indian bank notifications to expand representation
    bank_names = ["HDFC Bank", "SBI", "ICICI Bank", "Axis Bank", "Kotak Bank", "Punjab National Bank", "Canara Bank", "Union Bank"]
    upi_apps = ["PhonePe", "Google Pay", "Paytm", "BHIM UPI", "Cred"]
    merchants = ["SWIGGY", "ZOMATO", "AMAZON", "FLIPKART", "UBER", "BLINKIT", "ZEPTO"]

    np.random.seed(RANDOM_STATE)
    # Generate realistic safe bank notifications
    for _ in range(250):
        bank = np.random.choice(bank_names)
        amt = np.random.randint(50, 25000)
        acc = np.random.randint(1000, 9999)
        ref = np.random.randint(10000000, 99999999, dtype=np.int64)
        bal = np.random.randint(5000, 150000)
        merchant = np.random.choice(merchants)
        day = np.random.randint(1, 28)
        
        if np.random.random() < 0.6:
            text = f"Dear Customer, INR {amt:,.2f} debited from A/C **{acc} on {day}-Nov-24 to {merchant}. UPI Ref {ref}. Avail Bal: INR {bal:,.2f}. If not done by you, call 1800-425-0018 - {bank}"
        elif np.random.random() < 0.85:
            otp = np.random.randint(100000, 999999)
            text = f"{bank} Alert: OTP for txn of INR {amt:,.2f} at {merchant} on Card ending {acc} is {otp}. Valid for 10 mins. Do NOT share OTP with anyone."
        else:
            text = f"Your A/C **{acc} is credited with INR {amt:,.2f} on {day}-Nov-24 by UPI/P2A/{ref}. Clear Bal: INR {bal:,.2f}. - {bank}"
            
        custom_records.append({
            "message_text": text,
            "label": 0,
            "language_style": "English",
            "is_obfuscated": 0
        })

    # Generate realistic scam permutations (UPI cashback, KYC suspension, Electricity, Job)
    for _ in range(250):
        bank = np.random.choice(bank_names)
        app = np.random.choice(upi_apps)
        amt = np.random.choice([2500, 3000, 5000, 7500, 9999, 25000, 500000])
        phone = f"9{np.random.randint(100000000, 999999999, dtype=np.int64)}"
        shortener = np.random.choice(["bit.ly", "tinyurl.com", "cutt.ly", "rb.gy", "wa.me"])
        scam_type = np.random.choice(["kyc", "cashback", "electricity", "job", "lottery"])

        if scam_type == "kyc":
            if np.random.random() < 0.5:
                text = f"Dear {bank} customer, your account will be suspended today due to pending KYC update. Click http://{shortener}/{bank.lower().replace(' ', '')}-kyc-portal to verify immediately."
                lang = "English"
            else:
                text = f"Aapka {bank} account aaj block ho gaya hai. Turant apna PAN and Aadhaar update karein is link par: http://{shortener}/update-kyc"
                lang = "Hinglish"
        elif scam_type == "cashback":
            if np.random.random() < 0.5:
                text = f"Congratulations! You won Rs {amt:,} cashback on {app}. Click {shortener}/{app.lower().replace(' ', '')}-reward to accept and enter your UPI PIN."
                lang = "English"
            else:
                text = f"Aapko {app} par Rs {amt:,} ka cashback reward mila hai. Paise lene ke liye UPI PIN enter karein: {shortener}/claim-cashback"
                lang = "Hinglish"
        elif scam_type == "electricity":
            text = f"Dear consumer, your electricity power will be disconnected tonight at 9:30 PM due to unpaid bill. Immediately contact power officer at {phone}."
            lang = "English"
        elif scam_type == "job":
            text = f"Earn Rs 3,000 to Rs 8,000 daily part time work from home. Simple YouTube video rating job. No investment. Contact WhatsApp: {shortener}/{phone}"
            lang = "English"
        else:
            text = f"You are selected for KBC Lucky Draw prize of Rs 25,00,000! Call manager at {phone} to claim your prize check. Ref: KBC-2024."
            lang = "English"

        custom_records.append({
            "message_text": text,
            "label": 1,
            "language_style": lang,
            "is_obfuscated": 0
        })

    df_custom = pd.DataFrame(custom_records)
    print(f"Created {len(df_custom)} custom Indian messages (Safe: {sum(df_custom['label'] == 0)}, Scam: {sum(df_custom['label'] == 1)})")

    # Merge Full Dataset
    df_combined = pd.concat([df_uci, df_custom], ignore_index=True)
    df_combined.drop_duplicates(subset=['message_text'], inplace=True)
    df_combined.reset_index(drop=True, inplace=True)

    print("=" * 60)
    print(f"Combined Dataset Total: {len(df_combined)} samples")
    print(df_combined['label'].value_counts().rename({0: 'Safe (0)', 1: 'Scam (1)'}))

    # Step 3: Stratified Splitting (Train: 70%, Val: 15%, Test: 15%)
    print("=" * 60)
    print("Step 3: Creating Stratified Train / Validation / Test Splits...")
    train_df, temp_df = train_test_split(
        df_combined,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df_combined['label'],
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=RANDOM_STATE
    )

    print(f"Train Set: {len(train_df)} ({sum(train_df['label'] == 0)} safe, {sum(train_df['label'] == 1)} scam)")
    print(f"Validation Set: {len(val_df)} ({sum(val_df['label'] == 0)} safe, {sum(val_df['label'] == 1)} scam)")
    print(f"Test Set: {len(test_df)} ({sum(test_df['label'] == 0)} safe, {sum(test_df['label'] == 1)} scam)")

    # Step 4: Generating Held-Out Obfuscated Test Set (Adversarial Robustness Testing)
    print("=" * 60)
    print("Step 4: Generating Adversarial Obfuscated Test Suite...")
    # Extract scam messages from test set and custom pool to create obfuscated counterparts
    test_scams = test_df[test_df['label'] == 1]['message_text'].tolist()
    # Augment with Indian scams
    test_scams.extend([s['message_text'] for s in INDIAN_SCAMS])

    obfuscated_records = generate_obfuscated_dataset(test_scams)
    df_obfuscated = pd.DataFrame(obfuscated_records)
    print(f"Generated {len(df_obfuscated)} obfuscated adversarial test messages")

    # Step 5: Save Processed Files
    FULL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    df_obfuscated.to_csv(OBFUSCATED_TEST_PATH, index=False)
    df_combined.to_csv(FULL_DATASET_PATH, index=False)

    print("=" * 60)
    print("All processed datasets saved successfully:")
    print(f" - Train: {TRAIN_DATA_PATH}")
    print(f" - Validation: {VAL_DATA_PATH}")
    print(f" - Test: {TEST_DATA_PATH}")
    print(f" - Obfuscated Test: {OBFUSCATED_TEST_PATH}")
    print(f" - Full Dataset: {FULL_DATASET_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    build_and_prepare_dataset()
