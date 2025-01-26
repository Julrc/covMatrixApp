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
VOL_WINDOW = 10
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
        df_vol = df_returns.rolling(window=vol_window, min_periods=1).std().fillna(0.0)
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

def min_var_portfolio(cov_matrix):
    """
    Solve for ht eminimum variance portfolio subject to sum of weights = 1.
    No short-sale constraints here (weights can be negative)

    min w^T Sigma w, subject to sum(w) = 1

    cov_matrix: shape (n_assets, n_assets)
    returns: (n_assets,) weight vector
    """

    n_assets = cov_matrix.shape[0]
    # Covariance might be signular or ill-conditioned, so add a small ridge
    cov_matrix += np.eye(n_assets) * 1e-6

    # We can solve with the formula for unconstrained min-var with sum(w)=1:
    # w* (Sigma^-1 * u) / (u^T Sigma^-1 u), where u is a vector of ones
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(n_assets)
    numerator = inv_cov @ ones
    denom = ones @ numerator
    w = numerator / denom
    return w

def walk_forward_compare(model, df_features, df_returns, tickers, input_window):
    """
    Walk-forward:
    - For each day i from [input_window ..end-1], use [i-input_window ..i-1] as window
    to estimate covariances:
    1. Rollingcov (Direct from np.cov)
    2. Factor Cov (our LSTM model)
    - Then build min-var portfolio for each approach
    -Evaluate PnL on day i
    Returns a DataFrame with daily PnL
    """
    results = []
    all_dates = df_features.index
    n_assets = len(tickers)

    for i in range(input_window, len(df_features) - 1):
        # We'll predict for day i, then realize the returns on day i + 1
        window_start = i - input_window
        window_end = i

        # 1) Build rolling cov
        # For rolling-sample cov, just use the raw returns from df_returns
        # (The last input_window days)
        rolling_window = df_returns.iloc[window_start:window_end].values
        roll_cov = np.cov(rolling_window.T, ddof=1) # Shape (n_assets, n_assets)

        # 2) Build Factor Cov
        factor_window = df_features.iloc[window_start:window_end]
        X_infer = df_to_tensor(factor_window).to(DEVICE)

        with torch.no_grad():
            Sigma_pred = model(X_infer) # (1, n_assets, n_assets)
        Sigma_np = Sigma_pred[0].cpu().numpy()

        # 3) Compute min-var portfolios

        w_roll = min_var_portfolio(roll_cov)
        w_factor = min_var_portfolio(Sigma_np)
        
        # 4) Next day's realized return
        next_day_idx = i # Day i+1 in terms of zero-based index inside df_returns
        next_day_rets = df_returns.iloc[next_day_idx].values # shape (n_assets,)

        pnl_roll = np.dot(w_roll, next_day_rets)
        pnl_factor = np.dot(w_factor, next_day_rets)

        results.append({
            "date": all_dates[next_day_idx],
            "pnl_rolling": pnl_roll,
            "pnl_factor": pnl_factor
        })

    return pd.DataFrame(results)

def plot_cumulative_returns(df_pnl):
    """
    df_pnl has columns ["data", "pnl_rolling", "pnl_factor"]
    Plot the cumulative sum of each strategy
    """

    df_pnl = df_pnl.copy()
    df_pnl["cum_roll"] = df_pnl["pnl_rolling"].cumsum()
    df_pnl["cum_factor"] = df_pnl["pnl_factor"].cumsum()

    plt.figure(figsize=(10,6))
    plt.plot(df_pnl["date"], df_pnl["cum_roll"], label="Rolling Cov")
    plt.plot(df_pnl["date"], df_pnl["cum_factor"], label="Factor Cov")
    plt.title("Cumulative Returns Comparison")
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)
    plt.close()


def walk_forward_covariance(model, df_features, tickers):
    """
    For each day i from input_window to the endo f df_features
    Use the last input_window rows up to day i-1
    Predict covariacne for day i
    Compare or store it
    """
    results = []
    all_dates = df_features.index
    n_assets = len(tickers)

    # Start from day=INPUT_WINDOW, so we have enough past data
    for i in range(INPUT_WINDOW, len(df_features)):
        # The window of hte previous Input_window days
        window_end_idx = i
        window_start_idx = i - INPUT_WINDOW
        df_window = df_features.iloc[window_start_idx:window_end_idx]

        # Build input tensor
        X_infer = df_to_tensor(df_window).to(DEVICE)

        with torch.no_grad():
            Sigma_pred = model(X_infer)
        Sigma_np = Sigma_pred[0].cpu().numpy()

        # Turn into correlation matrix
        diag_std = np.sqrt(np.diag(Sigma_np))
        diag_std[diag_std <=1e-12] = 1e-12
        outer_std = np.outer(diag_std, diag_std)
        corr_matrix = Sigma_np / outer_std

        # Store results
        pred_date = all_dates[i] # This is the current day 
        results.append({
            "date": pred_date,
            "Sigma":Sigma_np,
            "Corr":corr_matrix
        })

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
    start_date = "2000-01-01"
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

    st.subheader("Run Walk-Forward Covariance")

    if st.button("Compute Covariances:"):
        # We'll do a simple walk-forward from day INPUT_WINDOW
        results = walk_forward_covariance(model, df_features_full, tickers)

        st.write(f"Computed walk-forward covariance for {len(results)} days.")

        if results:
            last_day_result = results[-1]
            st.write(f"Latest Predicted covariance** (Date: {last_day_result['date']})")
            # Plot correlation
            corr_matrix = last_day_result["Corr"]
            plot_heatmap(corr_matrix,tickers,f"Predicted Correlation on {last_day_result['date']}")

    
    st.subheader("Run Backtest and Compare Factor Cov vs. Rolling Cov")

    if st.button("Run Backtest"):
        df_pnl = walk_forward_compare(
            model=model,
            df_features=df_features_full,
            df_returns=df_returns,
            tickers=tickers,
            input_window=INPUT_WINDOW
        )

        st.write("Backtest results (first 5 rows):")
        st.write(df_pnl.head())

        plot_cumulative_returns(df_pnl)

        final_roll = df_pnl["pnl_rolling"].sum()
        final_factor= df_pnl["pnl_factor"].sum()

        st.write("**Total PNL - Rolling Cov**", round(final_roll, 4))
        st.write("**Total PNL - Factor Cov**", round(final_factor, 4))

        # Possibly compute sharpe, etc.
        st.write("**Rolling Cov Sharpe**:",
            round(df_pnl["pnl_rolling"].mean() / df_pnl["pnl_rolling"].std(), 3))
        st.write("**Factor Cov SHarpe:**",
            round(df_pnl["pnl_factor"].mean() / df_pnl["pnl_factor"].std(), 3))


if __name__ == "__main__":
    main()
