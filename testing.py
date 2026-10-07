from datetime import datetime, timedelta
import torch
from functions import data_to_phi_x, calc_preds

tickers = [
    "AAPL",
    "AMD",
    "AMZN",
    "INTC",
    "META",
    "MSFT",
    "NVDA"
]

# Test model
x = 9 # number of days from present to present - x (counting closed market days)

end_date = datetime.today()
start_date = end_date - timedelta(days=x)

phi_x_flat, _ = data_to_phi_x(tickers,start_date,end_date,window_size=3)

coeffs = torch.load('model.pkl')

print(calc_preds(coeffs, phi_x_flat))

# end_date = datetime(year=2026, month=9, day=22)
# start_date = datetime(year=2026, month=9, day=18)