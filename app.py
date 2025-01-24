import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import pandas as pd
import os

# ------------------------------
# 1. Global Settings
# ------------------------------
INPUT_WINDOW = 40            # number of past days to feed the model
ADD_ROLLING_VOL_FEATURE = True
VOL_WINDOW = 40
N_FACTORS = 2
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT_PROB = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

WEIGHTS_PATH = "factor_model_weights.pth"  # must exist (pre-trained)
DEFAULT_TICKERS = ["NVDA", "AAPL", "MSFT", "AMZN", "2222.SR", "META", "TSLA", "TSM", "AVGO"]

# ------------------------------
# 2. FactorCovModel Definition
# ------------------------------
class FactorCovModel(nn.Module):
    """
    Same architecture as your training script. 
    This model outputs a covariance matrix for n_assets assets.
    """
    def __init__(self, input_size, hidden_size, num_layers, n_assets, n_factors, dropout_prob):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        # out_dim = loadings (n_assets*n_factors) + factor_log_vars (n_factors) + idio_log_vars (n_assets)
        self.out_dim = (n_assets * n_factors) + n_factors + n_assets
        self.fc = nn.Linear(hidden_size, self.out_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len=INPUT_WINDOW, input_size)
        returns: (batch_size, n_assets, n_assets) --> predicted covariance matrix
        """
        batch_size = x.size(0)

        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = lstm_out[:, -1, :]   # final time step
        last_out = self.dropout(last_out)
        raw_out = self.fc(last_out)     # shape: (batch_size, out_dim)

        # Decompose raw_out into loadings, factor_log_var, idio_log_var
        idx = 0
        loadings_size   = self.n_assets * self.n_factors
        factor_var_size = self.n_factors
        idio_var_size   = self.n_assets

        loadings_flat  = raw_out[:, idx : idx + loadings_size]
        idx += loadings_size
        factor_log_var = raw_out[:, idx : idx + factor_var_size]
        idx += factor_var_size
        idio_log_var   = raw_out[:, idx : idx + idio_var_size]
        idx += idio_var_size

        # Reshape and exponentiate
        loadings = loadings_flat.view(batch_size, self.n_assets, self.n_factors)
        factor_vars = torch.exp(factor_log_var)
        idio_vars   = torch.exp(idio_log_var)

        # Build covariance
        Sigma_batch = []
        for b in range(batch_size):
            Lambda = loadings[b]            # (n_assets, n_factors)
            F_diag = torch.diag(factor_vars[b])  # (n_factors, n_factors)
            factor_cov = Lambda @ F_diag @ Lambda.T
            idio_cov = torch.diag(idio_vars[b])
            Sigma_b = factor_cov + idio_cov
            Sigma_batch.append(Sigma_b)

        Sigma_pred = torch.stack(Sigma_batch, dim=0)  # shape: (batch_size, n_assets, n_assets)
        return Sigma_pred

# ------------------------------
# 3. Utility Functions
# ------------------------------
def load_pretrained_model(weights_path, input_size, n_assets, n_factors, hidden_size, num_layers, dropout_prob):
    """
    Instantiate the FactorCovModel and load saved weights.
    """
    model = FactorCovModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        n_assets=n_assets,
        n_factors=n_factors,
        dropout_prob=dropout_prob
    ).to(DEVICE)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Could not find {weights_path} in current directory.")
    
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model

def prepare_inference_input(df_returns, input_window, add_rolling_vol, vol_window, n_assets):
    """
    Build the final (1-batch) input window from the most recent data for inference.
    - Optionally includes rolling volatility.
    """
    if add_rolling_vol:
        df_vol = df_returns.rolling(vol_window).std().fillna(0.0)
        vol_cols = [f"{col}_vol" for col in df_vol.columns]
        df_vol.columns = vol_cols
        df_features = pd.concat([df_returns, df_vol], axis=1)
        input_size = n_assets * 2
    else:
        df_features = df_returns.copy()
        input_size = n_assets

    # We need at least 'input_window' days
    if len(df_features) < input_window:
        st.warning(f"Not enough rows in df_features to extract a {input_window}-day window.")
        return None, input_size
    
    # Extract the last 'input_window' rows as a single window
    window_data = df_features.iloc[-input_window:].values  # shape: (input_window, input_size)
    X_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)  # (1, input_window, input_size)
    return X_tensor, input_size

# ------------------------------
# 4. Streamlit App
# ------------------------------
def main():
    st.title("Pre-trained FactorCovModel Inference App")

    # Let user enter Tickers or use defaults
    default_tickers_str = ",".join(DEFAULT_TICKERS)
    user_tickers_str = st.text_input("Enter comma-separated tickers:", default_tickers_str)
    tickers = [t.strip() for t in user_tickers_str.split(",") if t.strip()]
    st.write(f"**Using tickers**: {tickers}")

    # Download data for the last few years
    end_date = None
    start_date = "2020-01-01"
    st.write(f"Downloading data from {start_date} to {end_date} ...")
    df_data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"].dropna()
    df_returns = df_data.pct_change().dropna()
    st.write("Data shape (daily returns):", df_returns.shape)

    # Number of assets
    n_assets = len(tickers)

    # Instantiate / Load Model
    st.subheader("Load Pre-trained Model")
    try:
        # We'll build a dummy input_size just to create the model. We'll get the real input_size
        # after we know if we're using rolling vol or not (which is set in the code).
        # But let's do minimal or we can just do the maximum (n_assets*2). It's fine as long as it matches at inference.
        dummy_input_size = n_assets * 2 if ADD_ROLLING_VOL_FEATURE else n_assets

        model = load_pretrained_model(
            weights_path=WEIGHTS_PATH,
            input_size=dummy_input_size,
            n_assets=n_assets,
            n_factors=N_FACTORS,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout_prob=DROPOUT_PROB
        )
        st.success("Model weights loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Prepare the single inference window
    st.subheader("Prepare Inference Window")
    X_infer, real_input_size = prepare_inference_input(
        df_returns = df_returns, 
        input_window=INPUT_WINDOW,
        add_rolling_vol=ADD_ROLLING_VOL_FEATURE,
        vol_window=VOL_WINDOW,
        n_assets=n_assets
    )

    if X_infer is None:
        st.stop()  # stops execution here if not enough data

    # If your pre-trained model definitely used (n_assets*2) as input_size (rolling vol),
    # make sure real_input_size matches the model's input size.
    # Otherwise, if there's a mismatch, you'll get an error. 
    # (In practice, you'd do more checks or have the same config as your training.)
    
    # Prediction button
    if st.button("Predict Covariance"):
        with torch.no_grad():
            X_infer = X_infer.to(DEVICE)
            Sigma_pred = model(X_infer)  # shape: (1, n_assets, n_assets)
        
        # Convert to numpy
        Sigma_np = Sigma_pred[0].cpu().numpy()
        st.write("**Predicted Covariance Matrix**:")
        st.write(Sigma_np)

        # Compute correlation matrix
        diag_std = np.sqrt(np.diag(Sigma_np))
        outer_std = np.outer(diag_std, diag_std)
        corr_matrix = Sigma_np / outer_std
        st.write("**Predicted Correlation Matrix**:")
        st.write(corr_matrix)

        st.success("Inference complete!")

if __name__ == "__main__":
    main()