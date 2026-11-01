import os
import pandas as pd
from datetime import datetime, timedelta

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from datetime import datetime, timedelta
from tools.tools import lookup_order, create_ticket
# ... rest of the file stays unchanged
from tools.tools import lookup_order, create_ticket

TODAY_STR = "2026-10-20"
TODAY = datetime.strptime(TODAY_STR, "%Y-%m-%d")

class SupportToolsManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.holidays = set()
        self._load_reference_data()

    def _load_reference_data(self):
        holidays_path = os.path.join(self.data_dir, "holidays.csv")
        if os.path.exists(holidays_path):
            df_hol = pd.read_csv(holidays_path, dtype=str)
            self.holidays = set(df_hol["date"].dropna().str.strip().tolist())

        self.orders_df = pd.read_csv(os.path.join(self.data_dir, "orders.csv"), dtype=str)
        self.customers_df = pd.read_csv(os.path.join(self.data_dir, "customers.csv"), dtype=str)

    def calculate_business_days_ahead(self, start_date: datetime, days: int) -> datetime:
        """Counts business days Mon-Sat, skipping dates in holidays.csv"""
        current = start_date
        added = 0
        while added < days:
            current += timedelta(days=1)
            # Mon=0 ... Sat=5, Sun=6
            if current.weekday() < 6 and current.strftime("%Y-%m-%d") not in self.holidays:
                added += 1
        return current

    def verify_caller(self, order_id: str, verification_val: str) -> bool:
        """
        Verifies customer identity:
        Matches delivery pincode OR last 4 digits of phone number.
        """
        order_row = self.orders_df[self.orders_df["order_id"] == order_id]
        if order_row.empty:
            return False

        order = order_row.iloc[0]
        delivery_pincode = str(order.get("delivery_pincode", "")).strip()

        clean_input = str(verification_val).strip()
        if clean_input == delivery_pincode:
            return True

        cust_id = str(order.get("customer_id", "")).strip()
        cust_row = self.customers_df[self.customers_df["customer_id"] == cust_id]
        if not cust_row.empty:
            phone = str(cust_row.iloc[0].get("phone", "")).strip()
            if len(phone) >= 4 and clean_input == phone[-4:]:
                return True

        return False

    def get_order_details(self, order_id: str):
        return lookup_order(order_id)

    def trigger_ticket_creation(self, summary: str):
        return create_ticket(summary)

if __name__ == "__main__":
    manager = SupportToolsManager()
    print("Testing verification and tools layer...")

    test_order_id = manager.orders_df.iloc[0]["order_id"]
    test_pincode = manager.orders_df.iloc[0]["delivery_pincode"]

    is_valid = manager.verify_caller(test_order_id, test_pincode)
    print(f"Order: {test_order_id} | Verified with pincode {test_pincode}: {is_valid}")

    order_data = manager.get_order_details(test_order_id)
    print(f"Lookup Order result status: {order_data.get('status', 'OK')}")