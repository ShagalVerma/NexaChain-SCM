"""
generate_data.py
Generates synthetic SCM dataset for NexaChain Electronics (P3 - Forecasting Accuracy)
3 years of monthly demand data: Jan 2022 - Dec 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

np.random.seed(42)
os.makedirs('data', exist_ok=True)

# ─────────────────────────────────────────────
# 1. SUPPLIERS
# ─────────────────────────────────────────────
suppliers = pd.DataFrame({
    'supplier_id':          ['S001', 'S002', 'S003', 'S004', 'S005'],
    'supplier_name':        ['TechParts Asia', 'GlobalChip Co.', 'ElectroSupply Ltd',
                             'Pacific Components', 'CoreTech India'],
    'country':              ['China', 'Taiwan', 'South Korea', 'Japan', 'India'],
    'reliability_score':    [4.2, 4.7, 4.0, 4.5, 3.8],
    'avg_lead_time_days':   [25, 20, 30, 22, 15],
    'on_time_delivery_pct': [85, 92, 78, 90, 88],
    'quality_rating':       [4.1, 4.6, 3.9, 4.4, 3.7],
})
suppliers.to_csv('data/suppliers.csv', index=False)

# ─────────────────────────────────────────────
# 2. PRODUCTS
# ─────────────────────────────────────────────
products = pd.DataFrame({
    'product_id':    ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008'],
    'product_name':  ['Smartphone A', 'Laptop Pro', 'Wireless Earbuds', 'Tablet X',
                      'Smart TV 4K', 'Gaming Console', 'Smartwatch', 'Power Bank'],
    'category':      ['Mobile', 'Computing', 'Audio', 'Computing',
                      'Display', 'Gaming', 'Wearable', 'Accessories'],
    'unit_price':    [18000, 65000, 3500, 28000, 45000, 35000, 8000, 1500],
    'supplier_id':   ['S001', 'S002', 'S001', 'S002', 'S003', 'S004', 'S001', 'S005'],
    'reorder_point': [500, 200, 800, 300, 150, 100, 400, 1000],
    'safety_stock':  [250, 100, 400, 150, 75, 50, 200, 500],
})
products.to_csv('data/products.csv', index=False)

# ─────────────────────────────────────────────
# 3. DEMAND HISTORY (Jan 2022 – Dec 2024)
# ─────────────────────────────────────────────
months  = pd.date_range(start='2022-01-01', end='2024-12-01', freq='MS')
regions = ['North', 'South', 'West']

base_demand = {
    'P001': 1200, 'P002': 400,  'P003': 1800, 'P004': 500,
    'P005': 250,  'P006': 180,  'P007': 700,  'P008': 2000,
}
growth_rate = {
    'P001': 0.005, 'P002': 0.003, 'P003': 0.012, 'P004': 0.004,
    'P005': 0.002, 'P006': 0.006, 'P007': 0.015, 'P008': 0.008,
}
region_split  = {'North': 0.35, 'South': 0.40, 'West': 0.25}

# Indian festival seasonality multipliers
festival_multiplier = {
    1: 1.25, 2: 1.10, 3: 1.00, 4: 0.95, 5: 0.90, 6: 0.95,
    7: 1.10, 8: 1.05, 9: 1.15, 10: 1.45, 11: 1.35, 12: 1.20,
}

records = []
for _, prod in products.iterrows():
    pid = prod['product_id']
    for i, month in enumerate(months):
        base     = base_demand[pid]
        trend    = (1 + growth_rate[pid]) ** i
        seasonal = festival_multiplier[month.month]

        for region in regions:
            regional_base = base * trend * seasonal * region_split[region]
            noise         = np.random.normal(1.0, 0.08)
            demand        = max(10, int(regional_base * noise))

            # Simulate manual forecast error — higher variance, slight bias
            # Real-world manual forecasts typically have 20–35% MAPE
            manual_error = np.random.normal(1.05, 0.28)  # overestimate bias + high noise
            manual_forecast = max(5, int(demand * manual_error))

            records.append({
                'record_id':       f"D{len(records)+1:05d}",
                'product_id':      pid,
                'product_name':    prod['product_name'],
                'category':        prod['category'],
                'year_month':      month.strftime('%Y-%m'),
                'date':            month,
                'region':          region,
                'actual_demand':   demand,
                'manual_forecast': manual_forecast,
                'unit_price':      prod['unit_price'],
                'revenue':         demand * prod['unit_price'],
            })

demand_df = pd.DataFrame(records)
demand_df.to_csv('data/demand_history.csv', index=False)

# ─────────────────────────────────────────────
# 4. INVENTORY
# ─────────────────────────────────────────────
inv_records = []
for _, prod in products.iterrows():
    pid           = prod['product_id']
    opening_stock = prod['reorder_point'] * 3

    for month in months:
        month_str    = month.strftime('%Y-%m')
        total_demand = demand_df[
            (demand_df['product_id'] == pid) &
            (demand_df['year_month'] == month_str)
        ]['actual_demand'].sum()

        units_received = 0
        if opening_stock < prod['reorder_point'] * 1.5:
            units_received = int(prod['reorder_point'] * 2.5 * np.random.uniform(0.9, 1.1))

        available     = opening_stock + units_received
        units_sold    = min(total_demand, available)
        closing_stock = max(0, available - units_sold)
        stockout      = 1 if total_demand > available else 0
        stockout_qty  = max(0, total_demand - available)

        inv_records.append({
            'product_id':       pid,
            'product_name':     prod['product_name'],
            'year_month':       month_str,
            'opening_stock':    opening_stock,
            'units_received':   units_received,
            'total_demand':     total_demand,
            'units_sold':       units_sold,
            'closing_stock':    closing_stock,
            'stockout_flag':    stockout,
            'stockout_qty':     stockout_qty,
            'reorder_triggered': 1 if units_received > 0 else 0,
            'holding_cost':     int(closing_stock * prod['unit_price'] * 0.002),
        })

        opening_stock = closing_stock

inv_df = pd.DataFrame(inv_records)
inv_df.to_csv('data/inventory.csv', index=False)

print(" Data generated successfully!")
print(f"   Suppliers  : {len(suppliers)} records")
print(f"   Products   : {len(products)} records")
print(f"   Demand hist: {len(demand_df)} records  ({len(months)} months × {len(regions)} regions × 8 products)")
print(f"   Inventory  : {len(inv_df)} records")
