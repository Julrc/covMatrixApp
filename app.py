import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# 1. Global Settings
# ------------------------------
INPUT_WINDOW = 40
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
    def __init__(self, input_size, hidden_size, num_layers, n_assets, n_factors, dropout_prob):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors

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
        x: (batch_size, seq_len=INPUT_WINDOW, input_size)
        returns: (batch_size, n_assets, n_assets) covariance
        """
        batch_size = x.size(0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # final time step
        last_out = self.dropout(last_out)
        raw_out = self.fc(last_out)    # shape: (batch_size, out_dim)

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

        Sigma_batch = []
        for b in range(batch_size):
            Lambda = loadings[b]                    # (n_assets, n_factors)
            F_diag = torch.diag(factor_vars[b])     # (n_factors, n_factors)
            factor_cov = Lambda @ F_diag @ Lambda.T
            idio_cov = torch.diag(idio_vars[b])
            Sigma_b = factor_cov + idio_cov
            Sigma_batch.append(Sigma_b)

        Sigma_pred = torch.stack(Sigma_batch, dim=0)  # (batch_size, n_assets, n_assets)
        return Sigma_pred

# ------------------------------
# 3. Utility Functions
# ------------------------------
def load_pretrained_model(weights_path, input_size, n_assets, n_factors, hidden_size, num_layers, dropout_prob):
    model = FactorCovModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        n_assets=n_assets,
        n_factors=n_factors,
        dropout_prob=dropout_prob
    ).to(DEVICE)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Could not find {weights_path}.")
    
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model

def plot_heatmap(matrix, labels, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f", cmap='viridis')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(plt)
    plt.close()

def build_feature_window(df_returns, add_rolling_vol, vol_window):
    """
    Given a df of daily returns, optionally compute rolling vol and
    return a new DataFrame with either:
      - columns = returns only
      - columns = returns + rolling_vol
    """
    if add_rolling_vol:
        df_vol = df_returns.rolling(vol_window).std().fillna(0.0)
        vol_cols = [f"{col}_vol" for col in df_vol.columns]
        df_vol.columns = vol_cols
        df_features = pd.concat([df_returns, df_vol], axis=1)
    else:
        df_features = df_returns.copy()

    return df_features

def df_to_tensor(df_window):
    """
    Convert the last INPUT_WINDOW rows of a DataFrame into a single input tensor: (1, INPUT_WINDOW, input_size)
    """
    window_data = df_window.values  # shape: (INPUT_WINDOW, input_size)
    X_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
    return X_tensor

def predict_multiple_days(
    model, df_window, tickers, num_days, 
    input_window, vol_window, add_rolling_vol
):
    """
    Iteratively predict covariance matrices for the next `num_days`.
    - df_window: a DataFrame of shape (INPUT_WINDOW, num_features) with the *most recent* window.
    - Each iteration:
        1. Build input tensor from df_window
        2. Predict covariance matrix (Sigma)
        3. Generate synthetic returns from N(0, Sigma)
        4. Append the synthetic returns to df_window (drop oldest row)
        5. Recompute rolling volatility if needed
        6. Store the predicted Sigma
    
    Returns: list of (Sigma_np, corr_matrix_np)
    """
    results = []
    n_assets = len(tickers)
    model.eval()

    for day in range(1, num_days + 1):
        # 1) Build input tensor
        X_infer = df_to_tensor(df_window).to(DEVICE)

        with torch.no_grad():
            Sigma_pred = model(X_infer)  # shape: (1, n_assets, n_assets)
        Sigma_np = Sigma_pred[0].cpu().numpy()

        # 2) Compute correlation from Sigma
        diag_std = np.sqrt(np.diag(Sigma_np))
        # Guard against zero or negative diagonal
        diag_std[diag_std <= 1e-12] = 1e-12
        outer_std = np.outer(diag_std, diag_std)
        corr_matrix = Sigma_np / outer_std

        # Store result
        results.append((Sigma_np, corr_matrix))

        # 3) Generate synthetic returns from N(0, Sigma_pred)
        synthetic_returns = np.random.multivariate_normal(
            mean=np.zeros(n_assets), cov=Sigma_np
        )
        synthetic_returns = pd.Series(synthetic_returns, index=tickers)

        # 4) Update df_window: drop oldest row, append new row
        df_returns_part = df_window[tickers].copy().iloc[1:]  # drop oldest
        new_idx = df_returns_part.index[-1] + 1
        df_returns_part.loc[new_idx] = synthetic_returns.values
        df_returns_part = df_returns_part.sort_index()

        # 5) Recompute rolling vol if needed
        if add_rolling_vol:
            # We have the last input_window-1 real rows plus 1 new synthetic row
            # Compute rolling vol over VOL_WINDOW, but we only have input_window rows in memory.
            # If input_window == vol_window, that’s easy; or you can store more history externally.
            df_vol_part = df_returns_part.rolling(vol_window).std().fillna(0.0)
            vol_cols = [f"{col}_vol" for col in tickers]
            df_vol_part.columns = vol_cols

            df_features_part = pd.concat([df_returns_part, df_vol_part], axis=1)
        else:
            df_features_part = df_returns_part

        # Now df_features_part should have exactly INPUT_WINDOW rows again
        if len(df_features_part) > input_window:
            df_features_part = df_features_part.iloc[-input_window:]

        df_window = df_features_part  # update for next iteration

    return results

# ------------------------------
# 4. Streamlit App
# ------------------------------
def main():
    st.title("Correlation Prediction App")

    # 1) Collect user tickers
    default_tickers_str = ",".join(DEFAULT_TICKERS)
    user_tickers_str = st.text_input("Enter comma-separated tickers:", default_tickers_str)
    tickers = [t.strip().upper() for t in user_tickers_str.split(",") if t.strip()]
    st.write(f"**Requested tickers**: {tickers}")

    # 2) Download data, handle missing tickers
    end_date = None
    start_date = "2020-01-01"
    st.write(f"Downloading data from {start_date} to {end_date} ...")

    try:
        # Download
        df_data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
        
        # If df_data is a Series for single ticker, make it a DataFrame
        if isinstance(df_data, pd.Series):
            df_data = df_data.to_frame()

        df_data = df_data.dropna(how="all", axis=1)
        valid_tickers = df_data.columns.tolist()  # these tickers actually have data

        # Check for missing
        missing_tickers = set(tickers) - set(valid_tickers)
        if missing_tickers:
            st.warning(f"Some tickers had no data and were removed: {missing_tickers}")

        # Final ticker list is those that remain
        tickers = [t for t in tickers if t in valid_tickers]

        if len(tickers) < 2:
            st.error("Fewer than 2 tickers remain. Cannot proceed.")
            st.stop()

        df_data = df_data[tickers].dropna()

        # Returns
        df_returns = df_data.pct_change().dropna()
        st.write("Data shape (daily returns):", df_returns.shape)

    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

    # 3) Prepare to load the model
    n_assets = len(tickers)

    # If your pre-trained model is strictly for a certain number of assets:
    PRETRAINED_ASSETS = 9  # e.g. your original training with 9 tickers
    if n_assets != PRETRAINED_ASSETS:
        st.error(f"This pre-trained model expects {PRETRAINED_ASSETS} assets, but you provided {n_assets}.")
        st.stop()

    dummy_input_size = n_assets * 2 if ADD_ROLLING_VOL_FEATURE else n_assets
    try:
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
        st.stop()

    # 4) Build the last INPUT_WINDOW features for single-day prediction
    df_features_full = build_feature_window(df_returns, ADD_ROLLING_VOL_FEATURE, VOL_WINDOW)
    if len(df_features_full) < INPUT_WINDOW:
        st.error(f"Not enough data to build a {INPUT_WINDOW}-day window.")
        st.stop()

    df_window = df_features_full.iloc[-INPUT_WINDOW:].copy()
    X_infer = df_to_tensor(df_window)  # shape (1, INPUT_WINDOW, input_size)
    if X_infer.shape[2] != dummy_input_size:
        st.error("Feature dimension mismatch. Check rolling-vol logic and input_size.")
        st.stop()

    # 5) Single-day prediction
    st.subheader("Predict Next Day Covariance")
    if st.button("Predict Covariance (Single Day)"):
        with torch.no_grad():
            Sigma_pred = model(X_infer.to(DEVICE))  # (1, n_assets, n_assets)
        Sigma_np = Sigma_pred[0].cpu().numpy()

        # Correlation
        diag_std = np.sqrt(np.diag(Sigma_np))
        diag_std[diag_std <= 1e-12] = 1e-12
        outer_std = np.outer(diag_std, diag_std)
        corr_matrix = Sigma_np / outer_std

        st.write("**Predicted Correlation Matrix**:")
        plot_heatmap(corr_matrix, tickers, "Predicted Correlation")

        st.success("Single-day inference complete!")

    # 6) Multi-day forecast
    st.subheader("Predict Multiple Days")
    num_days = st.slider("Select number of days to predict", min_value=1, max_value=30, value=5)

    if st.button(f"Predict Covariance for Next {num_days} Days"):
        results = predict_multiple_days(
            model=model,
            df_window=df_window,          # last 40 days
            tickers=tickers,
            num_days=num_days,
            input_window=INPUT_WINDOW,
            vol_window=VOL_WINDOW,
            add_rolling_vol=ADD_ROLLING_VOL_FEATURE
        )

        for day_idx, (Sigma_np, corr_matrix) in enumerate(results, start=1):

            st.write("**Correlation Matrix**")
            plot_heatmap(corr_matrix, tickers, f"Day {day_idx} Correlation")

        st.success(f"Multi-day inference ({num_days} days) complete!")

if __name__ == "__main__":
    main()
