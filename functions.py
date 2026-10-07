import yfinance as yf
import torch, numpy as np, pandas as pd
from torch import tensor
import torch.nn.functional as F

# Prepare dataset
def data_to_phi_x(tickers,start_date,end_date,window_size=5):
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d", # Hyperparameter. Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo]
        group_by="ticker",
        auto_adjust=True,
        progress=False
    )

    df = df.astype({(name, 'Volume'): 'float64' for name in tickers})

    features = ['Open','High','Low','Close','Volume']

    for name in tickers:
        for feature in features:
            if feature != 'Volume':
                norm = np.log(df[name][feature] / df[name][feature].shift(1))
            else:
                norm = np.log(df[name][feature] + 1)
            df.loc[:, (name,feature)] = norm

    rename_dict = {
        'Open': 'Open_LogRet',
        'High': 'High_LogRet',
        'Low': 'Low_LogRet',
        'Close': 'Close_LogRet',
        'Volume': 'NormVolume'
    }
    
    df.columns = pd.MultiIndex.from_tuples([
        (name, rename_dict.get(feature, feature))
        for name, feature in df.columns
    ])
        
    df.dropna(inplace=True)
    array = np.array([df[names].values for names in tickers],dtype=float) 
    x = torch.tensor(array, dtype=torch.float32) # (name,time,features)

    x_windows = x.unfold( # (name,samples,features,windows)
        dimension=1,
        size=window_size,
        step=1
    ).permute(1, 0, 3, 2) # (samples,names,windows,features)

    if len(x_windows) > 1:
        phi_x = x_windows[:-1]
    else:
        phi_x = x_windows
    
    # prev_cost = x[:,window_size-1:-1,3]
    # curr_cost = x[:,window_size:,3]
    # y = (curr_cost > prev_cost).int().T # (sample_output,names)
    if window_size > 1:
        returns = x[:,window_size:,3].T
        y = (returns > 0).int()
    else:
        y = None

    # Normalize input data if necessary
    vals,indices = phi_x[:,:,:,4].max(dim=0) # We only normalize volume
    phi_x[:,:,:,4] = phi_x[:,:,:,4] / vals

    phi_x_flat = phi_x.flatten(start_dim=1)

    return phi_x_flat, y

# Split the data into two different sets: training/validation
def split(phi_x_flat,y,perc=0.7):
    phi_x_flat_dim = phi_x_flat.shape[0]
    trn_phi_x,val_phi_x = phi_x_flat[:int(perc*phi_x_flat_dim)],phi_x_flat[int(perc*phi_x_flat_dim):]
    trn_y,val_y = y[:int(perc*phi_x_flat_dim)],y[int(perc*phi_x_flat_dim):]
    return trn_phi_x, trn_y, val_phi_x, val_y
    
# Define Neural Network
def init_coeffs(n_coeff,hiddens):
    # We create two hidden layers (this means 175xsize_layer1 - size_layer1xsize_layer2 - size_layer2x7)
    sizes = [n_coeff] + hiddens + [7]
    n = len(sizes)
    layers = [(torch.rand(sizes[i], sizes[i+1])-0.3)/sizes[i+1]*4 for i in range(n-1)] # Creating random coefficients
    consts = [(torch.rand(1)[0]-0.5)*0.1 for i in range(n-1)] # torch.rand(.,1) Turns coeff into column vector
    for l in layers+consts: l.requires_grad_()
    return layers,consts

# Predictions and Loss
def calc_preds(coeffs, phi_x):
    layers,consts = coeffs
    n = len(layers)
    res = phi_x
    for i,l in enumerate(layers):
        res = res@l + consts[i] # @ Matricial product
        if i!=n-1: res = F.relu(res) # Rectified Linear Unit activation (max(0,res))
    return torch.sigmoid(res)

def calc_loss(w, phi_x, y): 
    return F.binary_cross_entropy_with_logits(calc_preds(w, phi_x), y.float())

# Updating algorithm
def update_coeffs(coeffs, lr):
    layers,consts = coeffs
    for layer in layers+consts:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()

def one_epoch(trn_phi_x,trn_y,w, lr):
    loss = calc_loss(w, trn_phi_x, trn_y)
    loss.backward()
    with torch.no_grad(): update_coeffs(w,lr)
    # print(f"{loss:.3f}", end="; ")

# Main training loop
def train_model(trn_phi_x,trn_y,epochs=10, hiddens=[64,64], lr=0.01):
    torch.manual_seed(442)
    n_coeff = trn_phi_x.shape[1] # 175
    w = init_coeffs(n_coeff,hiddens)
    for i in range(epochs): 
        one_epoch(trn_phi_x,trn_y,w, lr=lr)
    return w

# Accuracy
def acc(val_phi_x, val_y, coeffs): 
    return (val_y.bool()==(calc_preds(coeffs, val_phi_x)>0.5)).float().mean()