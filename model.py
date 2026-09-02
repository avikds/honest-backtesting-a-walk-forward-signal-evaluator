"""
Honest Backtesting: A Walk-Forward Signal Evaluator

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - to_log_returns
import numpy as np

def to_log_returns(prices):
    prices = np.asarray(prices, dtype=float)
    return np.diff(np.log(prices))

# Step 2 - rolling_zscore
def rolling_zscore(x, window):
    x = np.asarray(x, dtype=float)
    
    z = np.full(len(x), np.nan, dtype=float)
    
    for t in range(window - 1, len(x)):
        w = x[t - window + 1:t + 1]
        mean = np.mean(w)
        std = np.std(w, ddof=0)
        
        if std == 0:
            z[t] = 0.0
        else:
            z[t] = (x[t] - mean) / std
    
    return z

# Step 3 - momentum_feature
def momentum_feature(prices, lookback):
    prices = np.asarray(prices, dtype=float)
    
    momentum = np.full(len(prices), np.nan, dtype=float)
    momentum[lookback:] = np.log(prices[lookback:] / prices[:-lookback])
    
    return momentum

# Step 4 - has_lookahead
def has_lookahead(feature_fn, x):
    x = np.asarray(x, dtype=float)

    original = np.asarray(feature_fn(x)).copy()

    perturbed_x = x.copy()
    perturbed_x[-1] += 1000.0
    perturbed = np.asarray(feature_fn(perturbed_x))

    return not np.allclose(
        original[:-1],
        perturbed[:-1],
        equal_nan=True
    )

# Step 5 - audit_features
def audit_features(feature_fns, x):
    return {
        name: has_lookahead(fn, x)
        for name, fn in feature_fns.items()
    }

# Step 6 - purged_walk_forward_splits
def purged_walk_forward_splits(n_samples, n_splits, embargo):
    fold_size = n_samples // (n_splits + 1)
    splits = []

    for k in range(1, n_splits + 1):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_splits else n_samples

        train_end = max(0, test_start - embargo)

        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)

        splits.append((train_idx, test_idx))

    return splits

# Step 7 - fit_ols
def fit_ols(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_design = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

    return coef

# Step 8 - predict_ols
def predict_ols(X, coef):
    X = np.asarray(X, dtype=float)
    coef = np.asarray(coef, dtype=float)

    return coef[0] + X @ coef[1:]

# Step 9 - r2_score
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1.0 - ss_res / ss_tot

# Step 10 - information_coefficient
def information_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)

    denominator = np.sqrt(
        np.sum(y_true_centered ** 2) *
        np.sum(y_pred_centered ** 2)
    )

    if denominator == 0:
        return 0.0

    return np.sum(y_true_centered * y_pred_centered) / denominator

# Step 11 - sharpe_ratio
def sharpe_ratio(returns, periods_per_year):
    returns = np.asarray(returns, dtype=float)

    if returns.size < 2:
        return 0.0

    std = np.std(returns, ddof=1)

    if std == 0:
        return 0.0

    return np.mean(returns) / std * np.sqrt(periods_per_year)

# Step 12 - walk_forward_predict
def walk_forward_predict(X, y, splits):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    preds = np.full(len(y), np.nan, dtype=float)

    for train_idx, test_idx in splits:
        if len(train_idx) == 0:
            continue

        coef = fit_ols(X[train_idx], y[train_idx])
        preds[test_idx] = predict_ols(X[test_idx], coef)

    return preds

# Step 13 - positions_from_predictions
def positions_from_predictions(preds, threshold):
    preds = np.asarray(preds, dtype=float)

    positions = np.zeros(len(preds), dtype=float)
    valid = ~np.isnan(preds) & (np.abs(preds) >= threshold)

    positions[valid] = np.sign(preds[valid])

    return positions

# Step 14 - strategy_returns
def strategy_returns(positions, forward_returns, cost_bps):
    positions = np.asarray(positions, dtype=float)
    forward_returns = np.asarray(forward_returns, dtype=float)

    gross = positions * forward_returns

    prev_positions = np.concatenate(([0.0], positions[:-1]))
    turnover = np.abs(positions - prev_positions)

    costs = turnover * cost_bps / 10000.0

    return gross - costs

# Step 15 - permutation_pvalue
def permutation_pvalue(positions, forward_returns, n_permutations, seed):
    positions = np.asarray(positions, dtype=float)
    forward_returns = np.asarray(forward_returns, dtype=float)

    observed = np.mean(positions * forward_returns)

    rng = np.random.default_rng(seed)
    count = 0

    for _ in range(n_permutations):
        permuted_returns = rng.permutation(forward_returns)
        statistic = np.mean(positions * permuted_returns)

        if statistic >= observed:
            count += 1

    return (count + 1) / (n_permutations + 1)

# Step 16 - deflate_pvalue
def deflate_pvalue(p_value, n_trials):
    return float(min(1.0, p_value * n_trials))

# Step 17 - backtest_report
def backtest_report(net_returns, oos_pred, y_true, p_value, n_trials, periods_per_year):
    net_returns = np.asarray(net_returns, dtype=float)
    oos_pred = np.asarray(oos_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    mask = ~np.isnan(oos_pred)

    ic = information_coefficient(y_true[mask], oos_pred[mask])
    oos_r2 = r2_score(y_true[mask], oos_pred[mask])

    deflated_p = deflate_pvalue(p_value, n_trials)

    return {
        "n_periods": int(len(net_returns)),
        "sharpe": round(sharpe_ratio(net_returns, periods_per_year), 4),
        "ic": round(ic, 4),
        "oos_r2": round(oos_r2, 4),
        "p_value": round(float(p_value), 4),
        "deflated_p_value": round(deflated_p, 4),
        "verdict": "keep" if deflated_p < 0.05 else "reject",
    }

