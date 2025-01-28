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
VOL_WINDOW = 20
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

def model_predict(model, df_window):
    X_infer = df_to_tensor(df_window).to(DEVICE)
    with torch.no_grad():
        Sigma_pred = model(X_infer)
    return Sigma_pred[0].cpu().numpy()

def calculate_equal_weighted_returns(df_returns):
    n_assets = df_returns.shape[1]
    equal_weights = np.ones(n_assets) / n_assets # Equal weights for all of the assets
    equal_weighted_returns = df_returns.dot(equal_weights)
    return equal_weighted_returns

def walk_forward_compare(model, df_features, df_returns, tickers, input_window, benchmark):
    """
    Walk-forward backtest on the test set, comparing:
    - LSTM Factor Covariance Model
    - Selected Benchmark (Rolling Covariance or Equal-Weighted Portfolio)
    """
    results = []
    all_dates = df_features.index
    n_assets = len(tickers)

    # Calculate equal-weighted portfolio returns (if needed)
    if benchmark == "equal_weight":
        equal_weighted_returns = calculate_equal_weighted_returns(df_returns)

    # Start from the first day of the test set
    start_idx = input_window  # Ensure we have enough data for the first window

    for i in range(start_idx, len(df_features) - 1):
        window_start = i - input_window
        window_end = i

        # Rolling covariance (if needed)
        if benchmark == "rolling_cov":
            rolling_window = df_returns.iloc[window_start:window_end].values
            roll_cov = np.cov(rolling_window.T, ddof=1)
            w_roll = min_var_portfolio(roll_cov)
            pnl_roll = np.dot(w_roll, df_returns.iloc[i].values)
            results.append({
                "date": all_dates[i],
                "pnl_factor": np.dot(min_var_portfolio(model_predict(model, df_features.iloc[window_start:window_end])), df_returns.iloc[i].values),
                "pnl_benchmark": pnl_roll
            })

        # Equal-weighted portfolio (if needed)
        elif benchmark == "equal_weight":
            pnl_equal_weight = equal_weighted_returns.iloc[i]
            results.append({
                "date": all_dates[i],
                "pnl_factor": np.dot(min_var_portfolio(model_predict(model, df_features.iloc[window_start:window_end])), df_returns.iloc[i].values),
                "pnl_benchmark": pnl_equal_weight
            })

    return pd.DataFrame(results)

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

def split_train_test(df, test_size=0.15):
    n_test = int(len(df) * test_size)
    df_train = df.iloc[:-n_test]
    df_test = df.iloc[-n_test:]
    
    return df_train, df_test

def calculate_sharpe_ratio(pnl_series, annualization=252):
    daily_returns = pnl_series.dropna()
    if len(daily_returns) < 2:
        return 0.0 # Not enough data
    
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    if std_return ==0:
        return 0.0 # No division by 0
    
    daily_sharpe = mean_return / std_return
    annualized_sharpe = daily_sharpe * np.sqrt(annualization)
    return annualized_sharpe


def plot_and_display(df_pnl, benchmark):

    # Calculate cumulative
    df_pnl["cum_factor"] = (1 + df_pnl["pnl_factor"]).cumprod() - 1
    df_pnl["cum_benchmark"] = (1 + df_pnl["pnl_benchmark"]).cumprod() - 1

    plt.figure(figsize=(10, 6))
    plt.plot(df_pnl["date"], df_pnl["cum_factor"], label="Factor Covariance (LSTM Model)")
    if benchmark == "rolling_cov":
        plt.plot(df_pnl["date"], df_pnl["cum_benchmark"], label="Rolling COv")
    elif benchmark == "equal_weight":
        plt.plot(df_pnl["date"], df_pnl["cum_benchmark"], label="Equal-Weighted")
    plt.title("Cumulative Returns Comparison")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)
    plt.close()

    sharpe_factor = calculate_sharpe_ratio(df_pnl["pnl_factor"])
    sharpe_benchmark = calculate_sharpe_ratio(df_pnl["pnl_benchmark"])

    st.write("**Total PNL - LSTM model", round(df_pnl["cum_factor"].iloc[-1], 4))
    if benchmark == "rolling_cov":
        st.write("**Total PNL - Rolling cov", round(df_pnl["cum_benchmark"].iloc[-1],4))
    elif benchmark == "equal_weight":
        st.write("**Total PNL - Equal-Weighted", round(df_pnl["cum_benchmark"].iloc[-1], 4))
    st.write("**LSTM model Sharpe ratio: **", round(sharpe_factor, 3))
    st.write(f"**{benchmark.replace('_', ' ').title()} Sharpe ratio:**", round(sharpe_benchmark, 3))


# ------------------------------
# 4. Streamlit App
# ------------------------------

# Ensures these keys exist so we don't get KeyError in tabs.
if "model" not in st.session_state:
    st.session_state["model"] = None
if "df_features_train" not in st.session_state:
    st.session_state["df_features_train"] = None
if "df_features_test" not in st.session_state:
    st.session_state["df_features_test"] = None
if "df_test" not in st.session_state:
    st.session_state["df_test"] = None
if "tickers" not in st.session_state:
    st.session_state["tickers"] = []


def main():
    # ------------------ Custom CSS ------------------
    st.markdown(
        """
        <style>
        .header {
            color: #4B8BBE;
            font-size: 2.5em;
            text-align: center;
            padding: 10px;
        }
        .subheader {
            color: #2E86AB;
            font-size: 1.8em;
            padding: 5px 0;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            height: 3em;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # ------------------ Main Header ------------------
    st.markdown('<div class="header">LSTM Model For Portfolio Optimization</div>', unsafe_allow_html=True)
    
    # ------------------ Sidebar for Settings ------------------
    st.sidebar.header("Settings")
    
    # 1) Collect user tickers
    default_tickers_str = ",".join(DEFAULT_TICKERS)
    user_tickers_str = st.sidebar.text_input(
        "Enter comma-separated tickers:", 
        default_tickers_str,
        help="Provide stock ticker symbols separated by commas, e.g. AAPL, MSFT, GOOGL"
    )
    tickers = [t.strip().upper() for t in user_tickers_str.split(",") if t.strip()]
    st.sidebar.markdown(f"**Requested tickers**: {', '.join(tickers)}")
    
    # 2) Date selection
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2000-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    
    st.sidebar.markdown("---")
    
    # Button to load pre-trained model
    load_model_btn = st.sidebar.button("Load Pre-trained Model")
    
    # ------------------ Tabs ------------------
    tabs = st.tabs(["Data & Model", "Covariance Computation", "Benchmark Comparisons"])
    
    # ============== TAB 1: Data & Model ==============
    with tabs[0]:
        st.subheader("Data Download and Model Loading")
        
        # Download the data right away
        st.write(f"Downloading data from **{start_date}** to **{end_date}** ...")
        try:
            df_data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
            
            # Handle single ticker case
            if isinstance(df_data, pd.Series):
                df_data = df_data.to_frame()
                
            # Drop empty columns
            df_data = df_data.dropna(how="all", axis=1)
            valid_tickers = df_data.columns.tolist()
            
            # Warn if some tickers have no data
            missing_tickers = set(tickers) - set(valid_tickers)
            if missing_tickers:
                st.warning(f"Some tickers had no data and were removed: {', '.join(missing_tickers)}")
            
            # Filter tickers to only those with data
            tickers = [t for t in tickers if t in valid_tickers]
            if len(tickers) < 2:
                st.error("Fewer than 2 tickers remain. Cannot proceed.")
                st.stop()
            
            df_data = df_data[tickers].dropna()
            df_returns = df_data.pct_change().dropna()
            st.write("**Data shape (daily returns):**", df_returns.shape)
            
            # Split train/test
            df_train, df_test = split_train_test(df_returns, test_size=0.15)
            
            # Build feature windows
            df_features_train = build_feature_window(df_train, ADD_ROLLING_VOL_FEATURE, VOL_WINDOW)
            df_features_test = build_feature_window(df_test, ADD_ROLLING_VOL_FEATURE, VOL_WINDOW)
            
            # Store in session_state so other tabs can access
            st.session_state["tickers"] = tickers
            st.session_state["df_test"] = df_test
            st.session_state["df_features_train"] = df_features_train
            st.session_state["df_features_test"] = df_features_test
            
            st.success("Data downloaded and processed successfully!")
            
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            st.stop()
        
        # ------------------ Model Loading Section ------------------
        st.markdown("### Model Loading")
        if load_model_btn:
            n_assets = len(tickers)
            PRETRAINED_ASSETS = 9
            if n_assets != PRETRAINED_ASSETS:
                st.error(f"This pre-trained model expects {PRETRAINED_ASSETS} assets, but you provided {n_assets}.")
                st.stop()
            
            # Build the correct input dimension
            dummy_input_size = n_assets * 2 if ADD_ROLLING_VOL_FEATURE else n_assets
            
            try:
                # Load the model and store in session state
                model = load_pretrained_model(
                    weights_path=WEIGHTS_PATH,
                    input_size=dummy_input_size,
                    n_assets=n_assets,
                    n_factors=N_FACTORS,
                    hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS,
                    dropout_prob=DROPOUT_PROB
                )
                st.session_state["model"] = model
                st.success("Model weights loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()
        else:
            st.info("Click the sidebar button to load the pre-trained model.")
        
        # Show data preview
        st.markdown("### Data Preview")
        st.dataframe(df_data.head())
    
    # ============== TAB 2: Covariance Computation ==============
    with tabs[1]:
        st.subheader("Run Walk-Forward Covariance")
        
        # Check if model is loaded
        if st.session_state["model"] is None:
            st.warning("Please load the model first from the 'Data & Model' tab.")
            st.stop()
        
        # Retrieve from session_state
        model = st.session_state["model"]
        df_features_train = st.session_state["df_features_train"]
        tickers = st.session_state["tickers"]
        
        # Compute Covariance
        if st.button("Compute Covariances"):
            # We'll do a simple walk-forward from day INPUT_WINDOW
            results = walk_forward_covariance(model, df_features_train, tickers)

            st.write(f"Computed walk-forward covariance for {len(results)} days.")

            if results:
                last_day_result = results[-1]
                st.write(f"Latest Predicted covariance** (Date: {last_day_result['date']})")
                # Plot correlation
                corr_matrix = last_day_result["Corr"]
                plot_heatmap(corr_matrix,tickers,f"Predicted Correlation on {last_day_result['date']}")


                        
    # ============== TAB 3: Benchmark Comparisons ==============
    with tabs[2]:
        st.subheader("Choose Benchmark")
        
        # Check if model is loaded
        if st.session_state["model"] is None:
            st.warning("Please load the model first from the 'Data & Model' tab.")
            st.stop()
        
        # Retrieve from session_state
        model = st.session_state["model"]
        df_features_test = st.session_state["df_features_test"]
        df_test = st.session_state["df_test"]
        tickers = st.session_state["tickers"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Rolling Covariance"):
                with st.spinner('Running Rolling Covariance benchmark...'):
                    try:
                        df_pnl = walk_forward_compare(
                            model=model,
                            df_features=df_features_test,
                            df_returns=df_test,
                            tickers=tickers,
                            input_window=INPUT_WINDOW,
                            benchmark="rolling_cov"
                        )
                        plot_and_display(df_pnl, benchmark="Rolling Covariance")
                    except Exception as e:
                        st.error(f"Error in Rolling Covariance benchmark: {e}")
        
        with col2:
            if st.button("Equal-Weighted Portfolio"):
                with st.spinner('Running Equal-Weighted Portfolio benchmark...'):
                    try:
                        df_pnl = walk_forward_compare(
                            model=model,
                            df_features=df_features_test,
                            df_returns=df_test,
                            tickers=tickers,
                            input_window=INPUT_WINDOW,
                            benchmark="equal_weight"
                        )
                        plot_and_display(df_pnl, benchmark="Equal-Weighted Portfolio")
                    except Exception as e:
                        st.error(f"Error in Equal-Weighted Portfolio benchmark: {e}")
    
    # ------------------ Footer ------------------
    st.markdown("""
    ---
    **About:** This app utilizes an LSTM model to optimize portfolio allocations based on 
    covariance matrices. Input your desired tickers, load the pre-trained model, compute 
    predicted covariances, and compare your model's performance against standard benchmarks.
    """)

# Standard Streamlit entry point
if __name__ == "__main__":
    main()