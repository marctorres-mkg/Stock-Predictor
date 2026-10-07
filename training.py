# DEEP LEARNING MODEL
from datetime import datetime
import torch
from functions import data_to_phi_x, split, train_model, acc

start_date = datetime(year=2015, month=12, day=31)
end_date = datetime(year=2026, month=9, day=27)

tickers = [
    "AAPL",
    "AMD",
    "AMZN",
    "INTC",
    "META",
    "MSFT",
    "NVDA"
]

phi_x, y = data_to_phi_x(tickers,start_date,end_date,window_size=3)

trn_phi_x, trn_y, val_phi_x, val_y = split(phi_x,y,perc=0.8)

lr = 0.01
coeffs = train_model(trn_phi_x,trn_y,epochs=100,hiddens=[64,64],lr=lr)

print(f'Accuracy of the model = {acc(val_phi_x,val_y,coeffs)}. Learning rate = {lr}')

torch.save(coeffs, "model.pkl")
