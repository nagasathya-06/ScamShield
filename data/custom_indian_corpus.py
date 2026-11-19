"""
Custom Indian Scam, Banking, and Obfuscated Corpus Generator.
Creates authentic Indian UPI/banking scams, genuine bank notifications,
Hinglish examples, and adversarial obfuscated test messages.
"""

import random
import re
from typing import List, Dict, Tuple
from src.config import LEET_MAP

# 1. Custom Indian Scam Messages (Label 1)
INDIAN_SCAMS: List[Dict[str, str]] = [
    # UPI & Cashback Scams
    {
        "message_text": "Dear user, you have won Rs 3,500 cashback on PhonePe for your recent recharge! Click here to receive money in your account: bit.ly/phonepe-reward-claim. Enter UPI PIN to approve.",
        "language_style": "English",
        "category": "upi_cashback"
    },
    {
        "message_text": "Google Pay reward alert! Rs 2,000 has been credited to your scratch card. Click wa.me/919876543210 to claim immediately before expiry.",
        "language_style": "English",
        "category": "upi_cashback"
    },
    {
        "message_text": "Aapke Paytm wallet me Rs 5,000 cashback pending hai. Abhi link par click karein aur UPI PIN daalkar paise seedhe bank a/c me paayein: bit.ly/paytm-cashback-now",
        "language_style": "Hinglish",
        "category": "upi_cashback"
    },
    {
        "message_text": "Dear customer, Rs 4,999 refund approved from Amazon Pay. Please click bit.ly/amz-refund-upi and authorize payment via GooglePay or PhonePe PIN.",
        "language_style": "English",
        "category": "upi_cashback"
    },
    {
        "message_text": "Important: Aapko PhonePe par 1,999 rupaye ka bonus mila hai. Turant accept karne ke liye apna UPI pin verify karein: cutt.ly/phonepe-bonus",
        "language_style": "Hinglish",
        "category": "upi_cashback"
    },
    
    # KYC Suspension & Bank Account Blocking
    {
        "message_text": "Dear SBI Customer, your YONO NetBanking account has been blocked today due to pending PAN/KYC update. Kindly verify at http://bit.ly/sbi-yono-kyc-verify within 24 hours to resume services.",
        "language_style": "English",
        "category": "kyc_suspension"
    },
    {
        "message_text": "HDFC Bank Alert: Your saving a/c **4921 has been suspended today because of outdated KYC documents. Update immediately at bit.ly/hdfc-kyc-portal to avoid debit freeze.",
        "language_style": "English",
        "category": "kyc_suspension"
    },
    {
        "message_text": "Dear customer, your ICICI bank account will be closed within 12 hours due to unverified Aadhaar card. Visit tinyurl.com/icici-aadhaar-link to complete e-KYC now.",
        "language_style": "English",
        "category": "kyc_suspension"
    },
    {
        "message_text": "Punjab National Bank Warning: Your ATM debit card has been deactivated. Please call bank branch manager at 9845123456 or update at bit.ly/pnb-card-reactivate",
        "language_style": "English",
        "category": "kyc_suspension"
    },
    {
        "message_text": "Aapka SBI account aaj raat block kar diya jayega kyonki KYC verification baki hai. Turant is link par jayein aur apna details update karein: rb.gy/sbi-pan-kyc",
        "language_style": "Hinglish",
        "category": "kyc_suspension"
    },
    {
        "message_text": "Axis Bank alert: Your account is put on temporary hold. Submit OTP and PAN card details at is.gd/axis-online-verify to prevent permanent suspension.",
        "language_style": "English",
        "category": "kyc_suspension"
    },
    {
        "message_text": "Canara Bank Alert: Dear customer, your NetBanking access is restricted. Verify your login password and MPIN at bit.ly/canara-netbanking-fix",
        "language_style": "English",
        "category": "kyc_suspension"
    },
    {
        "message_text": "Dhyan de: Aapka bank account me KYC documents pending hone ki wajah se len-den band ho gaya hai. Chalu karne ke liye abhi call karein 8899776655.",
        "language_style": "Hinglish",
        "category": "kyc_suspension"
    },

    # Electricity Bill & Utility Disconnection Scams
    {
        "message_text": "Dear Consumer, your electricity power supply will be disconnected tonight at 9:30 PM from the main power office because your previous month bill was not updated. Immediately contact our Electricity Officer Mr. Verma at 9876543210.",
        "language_style": "English",
        "category": "electricity_scam"
    },
    {
        "message_text": "BESCOM Alert: Your electric meter will be cut off today at 8:00 PM due to unpaid electricity bill Rs 1,450. Call department officer on 8765432190 to avoid disconnection.",
        "language_style": "English",
        "category": "electricity_scam"
    },
    {
        "message_text": "Priy Upbhokta, aapka bijli connection aaj raat 9 baje kat diya jayega kyonki pichhle mahine ka bill update nahi hua hai. Turant bijli adhikaari se sampark karein: 9123456789.",
        "language_style": "Hinglish",
        "category": "electricity_scam"
    },
    {
        "message_text": "MSEDCL Notice: Urgent bill clearance needed. Your power line is scheduled for suspension in 2 hours. Pay online at bit.ly/power-bill-pay or dial 9012345678.",
        "language_style": "English",
        "category": "electricity_scam"
    },

    # Part-time Job, Work From Home & Telegram Scams
    {
        "message_text": "Part Time Job Offer! Earn Rs 3,000 to Rs 8,000 daily by simply rating hotels and liking videos on YouTube. No investment needed. WhatsApp your CV to 919988776655.",
        "language_style": "English",
        "category": "job_scam"
    },
    {
        "message_text": "Ghar baithe kamayein Rs 2,000 se Rs 5,000 rozana! Sirf hotel reviews dene hain. Daily payout via UPI. Telegram join karein: t.me/daily_earn_india",
        "language_style": "Hinglish",
        "category": "job_scam"
    },
    {
        "message_text": "Work from home vacancy in Amazon Logistics! Salary Rs 35,000/month + bonus. Limited vacancies. Register with your Aadhaar and bank details at bit.ly/amazon-hiring-now",
        "language_style": "English",
        "category": "job_scam"
    },
    {
        "message_text": "Aapke liye sunhera mauka! Part time data entry work karke har hafte Rs 15,000 kamayein. Sampark karein WhatsApp number 9811223344.",
        "language_style": "Hinglish",
        "category": "job_scam"
    },

    # Lottery, Prize & KBC Lucky Draw
    {
        "message_text": "Congratulations! You have been selected as the 1st prize winner of Rs 25,00,000 in KBC Jio All India Lucky Draw 2024. Contact Head Manager Rana Pratap on 9123487650 to claim your lottery check.",
        "language_style": "English",
        "category": "lottery_scam"
    },
    {
        "message_text": "Badhai ho! Aapka mobile number Kaun Banega Crorepati lucky draw me Rs 50 Lakh jeet chuka hai. Lottery file charge Rs 1,500 jama karke prize claim karein: call 9345678901.",
        "language_style": "Hinglish",
        "category": "lottery_scam"
    },
    {
        "message_text": "TATA Motors Mega Anniversary Lucky Draw: Your mobile number has won a brand new Tata Nexon or cash equivalent Rs 12.5 Lakhs. Click bit.ly/tata-lucky-draw to claim.",
        "language_style": "English",
        "category": "lottery_scam"
    },

    # Income Tax Refund & Loan Scams
    {
        "message_text": "Income Tax Department: An approved tax refund of Rs 18,450 has been sanctioned for your PAN. Please confirm your bank account and enter your netbanking password at bit.ly/incometax-refund-gov",
        "language_style": "English",
        "category": "refund_loan"
    },
    {
        "message_text": "Aapka Rs 5,00,000 ka personal loan pre-approved ho gaya hai bina kisi income proof ke. Zero interest for 6 months. Claim karein abhi: bit.ly/easy-loan-approved",
        "language_style": "Hinglish",
        "category": "refund_loan"
    },
    {
        "message_text": "Instant pre-approved Mudra Loan of Rs 10,00,000 granted by Govt of India. Disbursal in 10 minutes. Click bit.ly/pm-mudra-loan-online to apply.",
        "language_style": "English",
        "category": "refund_loan"
    },

    # SIM Card Block & Delivery Phishing
    {
        "message_text": "Dear Jio Customer, your 5G SIM will be permanently blocked within 24 hours due to non-verification of Aadhaar. Immediately dial our verification officer at 9765432109.",
        "language_style": "English",
        "category": "sim_block"
    },
    {
        "message_text": "Airtel 5G Alert: Aapka SIM card 12 ghante me band ho jayega. Turant e-KYC update karein is link par: cutt.ly/airtel-sim-kyc",
        "language_style": "Hinglish",
        "category": "sim_block"
    },
    {
        "message_text": "India Post Alert: Your parcel could not be delivered due to incomplete street address. Pay re-delivery fee of Rs 25 at bit.ly/indiapost-redelivery or item will be returned.",
        "language_style": "English",
        "category": "delivery_phishing"
    },
    {
        "message_text": "BlueDart Courier: Package #BD-89211 on hold. Address confirmation and customs duty clearance pending. Complete here: bit.ly/bluedart-track-verify",
        "language_style": "English",
        "category": "delivery_phishing"
    }
]

# 2. Genuine Indian Bank & Service Messages (Label 0 - Safe)
INDIAN_GENUINE: List[Dict[str, str]] = [
    # Genuine Bank Transaction Alerts
    {
        "message_text": "Dear Customer, INR 1,450.00 debited from A/C **4589 on 18-Nov-24 to SWIGGY UPI Ref 4321987654. Avail Bal INR 34,210.50. Call 18002586161 if not done by you - HDFC Bank",
        "language_style": "English",
        "category": "bank_alert"
    },
    {
        "message_text": "Dear SBI User, your A/C ending with 3921 has been credited by Rs 55,000.00 on 01-Nov-24 by NEFT salary from INFOSYS LTD. Total Balance: Rs 62,100.00. - State Bank of India",
        "language_style": "English",
        "category": "bank_alert"
    },
    {
        "message_text": "ICICI Bank: OTP for online transaction of INR 2,199.00 on AMAZON PAY with Card ending 8021 is 849201. Valid for 10 mins. Do NOT share OTP with anyone including bank staff.",
        "language_style": "English",
        "category": "bank_otp"
    },
    {
        "message_text": "Axis Bank Alert: Rs 2,000.00 withdrawn from ATM at MG Road Bangalore on 15-Nov-24. Available balance: Rs 18,920.00. Report fraud to 18604195555.",
        "language_style": "English",
        "category": "bank_alert"
    },
    {
        "message_text": "Dear Customer, your Kotak NetBanking password has been successfully changed on 14-Nov-2024. If this was not you, SMS BLOCK to 5676788 immediately.",
        "language_style": "English",
        "category": "bank_alert"
    },
    {
        "message_text": "Canara Bank: INR 350.00 debited towards monthly SMS charges for account **1092. Current balance is INR 8,420.00.",
        "language_style": "English",
        "category": "bank_alert"
    },
    {
        "message_text": "Punjab National Bank: Dear Customer, your account **7812 is credited with Rs 500.00 on 10-Nov-24 by UPI/P2A/432190. Clear Balance: Rs 14,200.00.",
        "language_style": "English",
        "category": "bank_alert"
    },
    {
        "message_text": "Bank of Baroda: Your monthly e-statement for account ending in 4432 has been sent to your registered email ID. For queries, visit bankofbaroda.in.",
        "language_style": "English",
        "category": "bank_alert"
    },
    {
        "message_text": "SBI Alert: 582103 is your OTP to add beneficiary on SBI YONO app. Do not disclose it to anyone. SBI never calls asking for OTP or passwords.",
        "language_style": "English",
        "category": "bank_otp"
    },
    {
        "message_text": "Your PhonePe verification code is 492019. Valid for 5 minutes. Use this code to log in to PhonePe. Never share your OTP with anyone.",
        "language_style": "English",
        "category": "bank_otp"
    },

    # Service & Delivery Notifications
    {
        "message_text": "Your Swiggy delivery partner Ramesh is on the way with your order #92841. Share PIN 3921 with the partner upon delivery. Track in app.",
        "language_style": "English",
        "category": "service_notification"
    },
    {
        "message_text": "Zomato: Your order from Meghana Foods has been accepted and is being prepared! Estimated delivery in 35 mins.",
        "language_style": "English",
        "category": "service_notification"
    },
    {
        "message_text": "Indian Railways: PNR 4521098736 Confirmed. Coach B3 Berth 42 (Side Lower) Train 12658 SBC CHENNAI EXP on 20-Nov-2024. Chart Prepared. Happy Journey.",
        "language_style": "English",
        "category": "service_notification"
    },
    {
        "message_text": "Amazon India: Your order 402-9842192-38102 containing Wireless Mouse will be delivered today by 8 PM by agent Suresh. Keep OTP 1928 ready.",
        "language_style": "English",
        "category": "service_notification"
    },
    {
        "message_text": "Airtel Thanks: Your daily data quota of 1.5GB is 80% consumed. Recharge with 1GB data pack for Rs 19 on Airtel Thanks App.",
        "language_style": "English",
        "category": "service_notification"
    },
    {
        "message_text": "Uber: Your driver Mohammed (KA01AB1234) is arriving in 3 mins. Share OTP 7421 to start your trip.",
        "language_style": "English",
        "category": "service_notification"
    },

    # Everyday Personal Messages (English & Hinglish)
    {
        "message_text": "Are you coming to college today? ML lab is scheduled at 2 PM in ground floor lab.",
        "language_style": "English",
        "category": "personal_chat"
    },
    {
        "message_text": "Bhai assignment submit kar diya kya? Deadline kal dopahar 12 baje tak hai.",
        "language_style": "Hinglish",
        "category": "personal_chat"
    },
    {
        "message_text": "Please bring the machine learning textbook and lab manual today, sir will check it.",
        "language_style": "English",
        "category": "personal_chat"
    },
    {
        "message_text": "Haan main 10 minute me pahunch raha hu metro station, wait karna wahi par.",
        "language_style": "Hinglish",
        "category": "personal_chat"
    },
    {
        "message_text": "Can you send the synopsis PDF? Evaluator is asking for the printout during internal review.",
        "language_style": "English",
        "category": "personal_chat"
    },
    {
        "message_text": "Kal sham ko football khelne chalenge ground pe, sabko inform kar dena.",
        "language_style": "Hinglish",
        "category": "personal_chat"
    },
    {
        "message_text": "Did you check the exam timetable? First paper is on 4th December.",
        "language_style": "English",
        "category": "personal_chat"
    },
    {
        "message_text": "Chai peene chalte hain canteen me, break ho gaya hai.",
        "language_style": "Hinglish",
        "category": "personal_chat"
    }
]

def obfuscate_text(text: str, intensity: float = 0.5) -> str:
    """
    Simulates adversarial obfuscation used by real-world scammers:
    - Leetspeak character substitutions (e.g., 'o' -> '0', 'e' -> '3', 'a' -> '@', 'i' -> '1')
    - Inserting dots in keywords (e.g., 'KYC' -> 'K.Y.C', 'SBI' -> 'S.B.I', 'OTP' -> 'O.T.P')
    - Spacing variations (e.g., 'blocked' -> 'b l o c k e d' or 'bl ocked')
    - Deliberate character duplications or replacements
    """
    # Key target terms to specifically dot or spaced-out
    dot_targets = ["KYC", "SBI", "HDFC", "OTP", "PIN", "UPI", "URGENT", "BESCOM", "KBC", "PAN"]
    for target in dot_targets:
        pattern = re.compile(rf"\b{target}\b", re.IGNORECASE)
        dotted = ".".join(list(target))
        if random.random() < 0.7:
            text = pattern.sub(dotted, text)

    words = text.split()
    obfuscated_words = []
    
    for word in words:
        # Don't touch URLs or pure numbers
        if "http" in word or "bit.ly" in word or "wa.me" in word or word.isdigit():
            obfuscated_words.append(word)
            continue
            
        lower_word = word.lower()
        
        # High impact spam keywords get spaced or substituted
        if lower_word in ["blocked", "urgent", "reward", "winner", "cashback", "refund", "verify"]:
            if random.random() < 0.6:
                # Insert spacing or leet
                chars = []
                for c in word:
                    c_lower = c.lower()
                    if c_lower in LEET_MAP and random.random() < 0.8:
                        replacement = random.choice(LEET_MAP[c_lower])
                        chars.append(replacement)
                    else:
                        chars.append(c)
                obfuscated_words.append("".join(chars))
                continue
                
        # General leet conversion based on intensity
        new_chars = []
        for char in word:
            char_lower = char.lower()
            if char_lower in LEET_MAP and random.random() < intensity:
                new_chars.append(random.choice(LEET_MAP[char_lower]))
            else:
                new_chars.append(char)
        obfuscated_words.append("".join(new_chars))
        
    return " ".join(obfuscated_words)


def generate_obfuscated_dataset(scam_messages: List[str]) -> List[Dict[str, any]]:
    """
    Generates a dedicated test set of obfuscated scam messages for robustness evaluation.
    """
    obfuscated_records = []
    for msg in scam_messages:
        obfuscated_text = obfuscate_text(msg, intensity=0.65)
        # Ensure it actually differs
        if obfuscated_text == msg:
            obfuscated_text = msg.replace("a", "@").replace("e", "3").replace("o", "0")
        
        obfuscated_records.append({
            "message_text": obfuscated_text,
            "original_text": msg,
            "label": 1,
            "language_style": "Obfuscated_Mix",
            "is_obfuscated": 1
        })
    return obfuscated_records
