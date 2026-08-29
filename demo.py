"""
Bhairava — Fraud Detection System
demo.py

Interactive live demonstration of the Bhairava Fraud Prevention Engine.
Simulates incoming merchant transactions, performs real-time ML risk scoring,
and executes automated defense policies (Allow / Step-Up 3DS / Auto-Decline).
"""

import time
from app.schemas.transaction import TransactionPayload
from app.services.auto_responder import auto_responder_service


def print_banner():
    print("\n" + "=" * 80)
    print("      BHAIRAVA AI -> REAL-TIME FRAUD DETECTOR & AUTO-RESPONDER DEMO")
    print("      Razorpay AI Buildathon 2026 | Track 2: AI Risk Manager")
    print("=" * 80 + "\n")


def format_action_badge(action: str) -> str:
    if action == "ALLOW":
        return "[ALLOW -> 0% Friction (Instant Cleared)]"
    elif action == "CHALLENGE_3DS":
        return "[CHALLENGE 3DS -> Step-Up OTP Verification]"
    else:
        return "[AUTO-DECLINE -> High-Risk Fraud Blocked]"


def run_demo():
    print_banner()

    test_scenarios = [
        {
            "title": "Scenario 1: Verified Repeat Customer Purchase",
            "description": "Customer with rich transaction history purchasing grocery/apparel during normal hours.",
            "payload": TransactionPayload(
                transaction_id="TXN_INR_88301",
                amount=750.0,
                merchant_id="merch_swiggy_01",
                customer_id="cust_rahul_99",
                card1=13926,
                card2=321.0,
                card3=150.0,
                addr1=299.0,
                P_emaildomain="gmail.com",
                R_emaildomain="gmail.com",
                has_identity_data=1,
                extra_features={
                    "amount_vs_card_mean": 0.95,
                    "card1_txn_count": 48,
                    "card_addr_txn_count": 35,
                    "card_full_txn_count": 30,
                    "rapid_card_activity": 0,
                    "card_time_since_prev": 172800.0,
                    "V258": 1.0,
                    "V257": 1.0,
                    "V70": 0.0,
                    "V201": 1.0,
                },
            ),
        },
        {
            "title": "Scenario 2: Suspicious Amount & Location Change (Gray Zone)",
            "description": "Established card attempting an unusually high transaction from a new regional address.",
            "payload": TransactionPayload(
                transaction_id="TXN_INR_88302",
                amount=8500.0,
                merchant_id="merch_croma_05",
                customer_id="cust_ananya_42",
                card1=15498,
                card2=490.0,
                card3=150.0,
                addr1=441.0,
                P_emaildomain="yahoo.com",
                R_emaildomain="yahoo.com",
                has_identity_data=1,
                extra_features={
                    "amount_vs_card_mean": 3.2,
                    "card1_txn_count": 12,
                    "card_addr_txn_count": 0,
                    "is_new_card_addr": 1,
                    "rapid_card_activity": 0,
                    "card_time_since_prev": 43200.0,
                },
            ),
        },
        {
            "title": "Scenario 3: Bot Attack / Stolen Card Velocity Burst",
            "description": "High-velocity automated testing with disposable email and missing telemetry.",
            "payload": TransactionPayload(
                transaction_id="TXN_INR_88303",
                amount=99000.0,
                merchant_id="merch_apple_store_02",
                customer_id="cust_anon_999",
                card1=99999,
                card2=500.0,
                card3=150.0,
                addr1=100.0,
                P_emaildomain="temp-inbox-proxy.xyz",
                R_emaildomain="unlinked-target.ru",
                has_identity_data=0,
                extra_features={
                    "amount_vs_card_mean": 18.5,
                    "amount_zscore": 5.2,
                    "card1_txn_count": 0,
                    "card_addr_txn_count": 0,
                    "rapid_card_activity": 1,
                    "card_time_since_prev": 15.0,
                    "email_domain_freq": 1,
                    "is_rare_email_domain": 1,
                    "is_new_card_addr": 1,
                    "V258": 6.0,
                    "V257": 5.0,
                },
            ),
        },
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"[{i}/3] {scenario['title']}")
        print(f"Context:     {scenario['description']}")
        print(f"Transaction: ID={scenario['payload'].transaction_id} | Amount=INR {scenario['payload'].amount:,.2f}")
        
        start_t = time.perf_counter()
        decision = auto_responder_service.evaluate_transaction(scenario["payload"])
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        print("\n--- BHAIRAVA DECISION ENGINE OUTPUT ---")
        print(f"Action:             {format_action_badge(decision.action.value)}")
        print(f"Fraud Risk Score:   {decision.risk_score * 100:.2f}% (Tier: {decision.risk_tier.value})")
        print(f"Model Confidence:   {decision.confidence * 100:.1f}%")
        print(f"Inference Latency:  {elapsed_ms:.2f} ms")
        print(f"OTP Required:       {'YES (Step-Up Challenge)' if decision.requires_otp_challenge else 'NO'}")
        print(f"Audit Trail ID:     {decision.audit_id}")
        
        print("Risk Reason Codes:")
        for reason in decision.reasons:
            print(f"  * {reason}")

        if decision.merchant_notification.enabled:
            print(f"\nMerchant Notification [{decision.merchant_notification.severity.value}]:")
            print(f"  Title:   {decision.merchant_notification.title}")
            print(f"  Message: {decision.merchant_notification.message}")
            print(f"  Action:  {decision.merchant_notification.action_required}")

        print("-" * 80 + "\n")
        time.sleep(0.3)

    print("=" * 80)
    print("Demo simulation completed successfully.")
    print("REST API Docs available at: http://localhost:8000/docs")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_demo()
