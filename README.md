# NexaChain SCM Dashboard — P3 Forecasting Accuracy
## UE23CS342BA1 · Jan–May 2026

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset
python generate_data.py

# 3. Train ML models & generate forecasts
python train_model.py

# 4. Launch dashboard
streamlit run app.py
```

### Project Structure
```
scm_p3_project/
├── data/
│   ├── demand_history.csv   ← 864 demand records (36 months × 3 regions × 8 products)
│   ├── forecasts.csv        ← Historical predictions + 6-month future forecast
│   ├── inventory.csv        ← 288 inventory records with stockout flags
│   ├── products.csv         ← 8 electronics products
│   ├── suppliers.csv        ← 5 suppliers
│   └── model_metrics.csv    ← Per-product MAE, RMSE, MAPE, accuracy
├── generate_data.py         ← Synthetic dataset generator
├── train_model.py           ← Random Forest model training
├── app.py                   ← Streamlit dashboard (5 pages)
├── requirements.txt
└── README.md
```

### Dashboard Pages
| Page | Description |
|------|-------------|
| Executive Overview | KPI cards, revenue trends, product performance |
| Demand Forecast | Actual vs Predicted + 6-month outlook with confidence bands |
| Product Analysis | Region breakdown, seasonal heatmap, revenue contribution |
| Inventory Health | Stockout alerts, reorder signals, fill rate tracking |
| ML Model Insights | Accuracy metrics, feature importance, model explanation |

### ML Model
- **Algorithm**: Random Forest Regressor (200 trees, max depth 8)
- **Features**: Lag values (1,2,3,6,12m), rolling stats, trend, festival flags, cyclical encoding
- **Result**: 91.2% average accuracy vs 85.0% manual → **+6.2% improvement**
