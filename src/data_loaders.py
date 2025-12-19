# src/data_loaders.py
import os
import warnings
import yaml
import pyreadr
import pandas as pd
import numpy as np


def _get_base_dir():
    """
    Returns the directory of this script
    """
    return os.path.dirname(os.path.abspath(__file__))

def return_default_config():
    """
    Returns the default configuration dictionary
    """
    return {
        "paths": {
            "data_dir": "./data",
            "results_dir": "./results",
            "target_tfr_csv": "./data/target_tfrs.csv",
            "midpoints_csv": "./data/midpoints.csv",
            "mortality_improvements_csv": "./data/mortality_improvements.csv",  # <-- NEW default
        },
        "diagnostics": {
            "print_target_csv": True,
            "mortality_improvements_debug": False,  # optional printing
        },
        "projections": {
            "start_year": 2018, "end_year": 2070, "step_years": 5,
            "death_choices": ["EEVV", "censo_2018", "midpoint"],
            "last_observed_year_by_death": {"EEVV": 2023, "censo_2018": 2018, "midpoint": 2018},
            "period_years": 5, "flows_latest_year": 2021,
        },
        "fertility": {
            "default_tfr_target": 1.5, "convergence_years": 50,
            "smoother": {"kind": "exp", "converge_frac": 0.99, "logistic": {"mid_frac": 0.5, "steepness": None}},
        },
        "midpoints": {"default_eevv_weight": 0.5},
        "age_bins": {
            "expected_bins": ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"],
            "order":         ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"],
        },
        "mortality": {
            "use_ma": True, "ma_window": 5,
            "improvement_total": 0.10,
            "convergence_years": 50,
            "smoother": {
                "kind": "exp",
                "converge_frac": 0.99,
                "logistic": {"mid_frac": 0.5, "steepness": None},
            },
        },
        "runs": {
            "mode": "no_draws",
            "no_draws_tasks": [
                {"sample_type": "mid",  "distribution": None, "label": "mid_omissions"},
                {"sample_type": "low",  "distribution": None, "label": "low_omissions"},
                {"sample_type": "high", "distribution": None, "label": "high_omissions"},
            ],
            "draws": {"num_draws": 1000, "dist_types": ["uniform","pert","beta","normal"], "label_pattern": "{dist}_draw_{i}"},
        },
        "unabridging": {"enabled": True},
        "filenames": {"asfr": "asfr.csv", "lt_M": "lt_M_t.csv", "lt_F": "lt_F_t.csv", "lt_T": "lt_T_t.csv"},
    }

def _resolve(ROOT_DIR, p):
    """
    Resolve path p relative to ROOT_DIR if not absolute.
    """
    return os.path.abspath(os.path.join(ROOT_DIR, p))

def _deep_merge(dst, src):
    """
    Recursively merge src into dst
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _validate_config(cfg: dict) -> None:
    """
    Sanity-check key config values; warn on implausible, error on impossible.
    """
    # Fertility
    fert = cfg.get("fertility", {})
    tfr = fert.get("default_tfr_target", None)
    if tfr is not None:
        if tfr <= 0:
            raise ValueError("fertility.default_tfr_target must be > 0.")
        if tfr < 0.5 or tfr > 5.0:
            warnings.warn(f"fertility.default_tfr_target={tfr} looks implausible; check input.")
    conv_years_f = fert.get("convergence_years", None)
    if conv_years_f is not None and conv_years_f <= 0:
        raise ValueError("fertility.convergence_years must be > 0.")
    cfrac = (fert.get("smoother", {}) or {}).get("converge_frac", None)
    if cfrac is not None and not (0 < cfrac < 1):
        raise ValueError("fertility.smoother.converge_frac must be in (0,1).")
    mid_frac = ((fert.get("smoother", {}) or {}).get("logistic", {}) or {}).get("mid_frac", None)
    if mid_frac is not None and not (0 < mid_frac < 1):
        warnings.warn(f"fertility.smoother.logistic.mid_frac={mid_frac} should be in (0,1); check input.")

    # Mortality
    mort = cfg.get("mortality", {})
    imp = mort.get("improvement_total", None)
    if imp is not None:
        if not (0 <= imp < 1):
            raise ValueError("mortality.improvement_total must be in [0,1).")
        if imp > 0.6:
            warnings.warn(f"mortality.improvement_total={imp} is unusually high; verify intent.")
    conv_years_m = mort.get("convergence_years", None)
    if conv_years_m is not None and conv_years_m <= 0:
        raise ValueError("mortality.convergence_years must be > 0.")
    ma_win = mort.get("ma_window", None)
    if ma_win is not None and ma_win < 1:
        raise ValueError("mortality.ma_window must be >= 1.")
    cfrac_m = (mort.get("smoother", {}) or {}).get("converge_frac", None)
    if cfrac_m is not None and not (0 < cfrac_m < 1):
        raise ValueError("mortality.smoother.converge_frac must be in (0,1).")
    mid_frac_m = ((mort.get("smoother", {}) or {}).get("logistic", {}) or {}).get("mid_frac", None)
    if mid_frac_m is not None and not (0 < mid_frac_m < 1):
        warnings.warn(f"mortality.smoother.logistic.mid_frac={mid_frac_m} should be in (0,1); check input.")

    # Midpoints
    mid = cfg.get("midpoints", {})
    w = mid.get("default_eevv_weight", None)
    if w is not None:
        if not (0 <= w <= 1):
            raise ValueError("midpoints.default_eevv_weight must be between 0 and 1.")
        if w < 0.1 or w > 0.9:
            warnings.warn(f"midpoints.default_eevv_weight={w} is extreme; confirm the blend.")

def _load_config(ROOT_DIR: str, path: str):
    """
    Load YAML config if present; otherwise use defaults for both config and paths.
    Returns (cfg, PATHS)
    """
    cfg = return_default_config()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        _deep_merge(cfg, user)
    else:
        print(f"[config] No config file at {path}; using built-in defaults.")

    _validate_config(cfg)

    PATHS = {
        "data_dir": _resolve(ROOT_DIR, cfg["paths"]["data_dir"]),
        "results_dir": _resolve(ROOT_DIR, cfg["paths"]["results_dir"]),
        "target_tfr_csv": _resolve(ROOT_DIR, cfg["paths"]["target_tfr_csv"]),
        "midpoints_csv": _resolve(ROOT_DIR, cfg["paths"]["midpoints_csv"]),
        "mortality_improvements_csv": _resolve(ROOT_DIR, cfg["paths"]["mortality_improvements_csv"]),  # <-- HERE
    }
    return cfg, PATHS

# ----------------------------- existing helpers ------------------------------

def allocate_and_drop_missing_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Redistributes rows with missing age data proportionally to rows with known ages,
    then removes the missing age rows.
    """
    df = df.copy()
    grouping = ['DPTO_NOMBRE', 'SEXO', 'FUENTE', 'ANO', 'VARIABLE']
    for keys, grp in df.groupby(grouping):
        idx = grp.index
        miss = grp['EDAD'].isna()
        obs  = ~miss
        M = grp.loc[miss, 'VALOR'].sum()
        S = grp.loc[obs,  'VALOR'].sum()
        if M > 0 and S > 0:
            weights = grp.loc[obs, 'VALOR'] / S
            df.loc[idx[obs], 'VALOR_withmissing'] += M * weights
    df = df[df['EDAD'].notna()].copy()
    return df

def get_lifetables_ex(DPTO):
    """
    Get life tables for a specific department (DPTO) from various distribution directories.
    """
    base_dir = _get_base_dir()
    distributions = ["beta", "normal", "pert", "uniform"]
    series_list = []
    for dist in distributions:
        dist_dir = os.path.join(base_dir, "..", "results", "lifetables", DPTO, "draw", dist)
        for file_name in os.listdir(dist_dir):
            file_path = os.path.join(dist_dir, file_name)
            df = pd.read_csv(file_path)
            if 'ex' not in df.columns:
                # print(df)
                raise Warning(f"'ex' column not found in {file_path}")
            else:
                series = df["ex"].rename(f"{dist}_{file_name}")
            series_list.append(series)
        ex_df_all = pd.concat(series_list, axis=1)
    return ex_df_all

def get_fertility():
    """
    Get fertility data from various distribution directories.
    """
    base_dir = _get_base_dir()
    distributions = ["beta", "normal", "pert", "uniform"]
    df_list = []
    tfr_dict = {}
    for dist in distributions:
        dist_dir = os.path.join(base_dir, "..", "results", "asfr", "total_nacional", "draw", dist)
        if not os.path.isdir(dist_dir):
            tfr_dict[dist] = []
            continue
        asfr_series = []
        tfr_values = []
        for file_name in os.listdir(dist_dir):
            file_path = os.path.join(dist_dir, file_name)
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                df = pd.read_csv(file_path)
                if 'asfr' in df.columns and not df.empty:
                    series = df['asfr'].rename(f"{dist}_{file_name}")
                    asfr_series.append(series)
                    tfr_values.append(5 * series.sum())
                else:
                    raise Warning(f"'asfr' column missing or empty in {file_path}")
        if asfr_series:
            df_dist = pd.concat(asfr_series, axis=1)
            df_list.append(df_dist)
        tfr_dict[dist] = tfr_values
    stacked_df_all = pd.concat(df_list, axis=1) if df_list else pd.DataFrame()
    tfr_df = pd.DataFrame({dist: pd.Series(vals) for dist, vals in tfr_dict.items()})
    return stacked_df_all, tfr_df

def read_rds_file(file_path: str) -> pd.DataFrame:
    """
    Reads an RDS file and returns its contents as a pandas DataFrame.
    """
    try:
        result = pyreadr.read_r(file_path)
        return result[None]
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_path}: {e}")

def load_all_data(data_dir) -> dict:
    """
    Loads all data files from the specified directory.
    """
    data_files = {'conteos': os.path.join(data_dir, 'conteos.rds')}
    for name, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    return {name: read_rds_file(path) for name, path in data_files.items()}

def correct_valor_for_omission(
    df: pd.DataFrame,
    sample_type: str,
    distribution: str = None,
    valor_col: str = 'VALOR',
    omision_col: str = 'OMISION'
) -> pd.Series:
    """
    Corrects census/survey values for undercounting (omission) using deterministic or 
    stochastic adjustment factors based on omission quality levels.
    
    Each omission level represents a range of undercount percentages:
        Level 1: 0.4% - 10.4% undercount
        Level 2: 10.5% - 20.3% undercount  
        Level 3: 20.4% - 30.1% undercount
        Level 4: 30.2% - 40.0% undercount
        Level 5: 40.1% - 49.9% undercount
    
    Args:
        df (pd.DataFrame): Input DataFrame with values to correct and omission levels.
        sample_type (str): Sampling strategy when distribution is None:
            - 'low': Conservative (uses lower bound of omission range)
            - 'mid': Best estimate (uses midpoint of omission range)
            - 'high': Liberal (uses upper bound of omission range)
        distribution (str, optional): Statistical distribution for stochastic correction:
            - None: Deterministic correction based on sample_type
            - 'uniform': Equal probability across range (use when no prior info)
            - 'pert': Modified beta favoring midpoint (use for expert judgment)
            - 'beta': Symmetric beta(2,2) (use for smooth central tendency)
            - 'normal': Truncated normal (use for natural measurement error)
        valor_col (str): Column name containing values to correct. Default: 'VALOR'
        omision_col (str): Column name containing omission levels (1-5). Default: 'OMISION'

    Returns:
        pd.Series: Corrected values calculated as: original_value × (1 + omission_rate)
        
    Raises:
        ValueError: If sample_type or distribution is invalid
        KeyError: If valor_col or omision_col not found in df
        
    Example:
        >>> df = pd.DataFrame({'VALOR': [1000, 2000], 'OMISION': [1, 3]})
        >>> corrected = correct_valor_for_omission(df, 'mid')
        >>> # Returns approximately [1054, 2505] using midpoint adjustments
    """
    # Validate inputs
    if valor_col not in df.columns:
        raise KeyError(f"Column '{valor_col}' not found in DataFrame")
    if omision_col not in df.columns:
        raise KeyError(f"Column '{omision_col}' not found in DataFrame")
    
    # Omission ranges based on census quality assessment
    omission_ranges = {
        1: (0.004, 0.104),
        2: (0.105, 0.203),
        3: (0.204, 0.301),
        4: (0.302, 0.400),
        5: (0.401, 0.499),
    }
    
    midpoints = {i: (low + high) / 2 for i, (low, high) in omission_ranges.items()}

    V   = df[valor_col]
    eps = pd.Series(0.0, index=df.index)

    # Only correct rows with valid omission levels
    valid = df[omision_col].notna()
    if not valid.any():
        # No omissions to correct, return original values
        return V
    
    levels = df.loc[valid, omision_col].astype(int)
    
    # Validate omission levels
    invalid_levels = levels[~levels.isin(omission_ranges.keys())]
    if len(invalid_levels) > 0:
        raise ValueError(f"Invalid omission levels found: {invalid_levels.unique()}. Must be 1-5.")

    # Deterministic correction (no distribution)
    if distribution is None:
        if sample_type == 'low':
            eps_vals = levels.map(lambda i: omission_ranges[i][0])
        elif sample_type == 'mid':
            eps_vals = levels.map(lambda i: midpoints[i])
        elif sample_type == 'high':
            eps_vals = levels.map(lambda i: omission_ranges[i][1])
        else:
            raise ValueError(f"sample_type must be 'low', 'mid', or 'high', got: '{sample_type}'")
        eps.loc[valid] = eps_vals
        
    # Stochastic correction (with distribution)
    else:
        dist = distribution.lower()
        a = levels.map(lambda i: omission_ranges[i][0]).to_numpy()
        b = levels.map(lambda i: omission_ranges[i][1]).to_numpy()
        m = levels.map(lambda i: midpoints[i]).to_numpy()
        k = len(a)

        if dist == 'uniform':

            draws = np.random.uniform(a, b, size=k)
            
        elif dist == 'pert':
            alpha = 1 + 4 * ((m - a) / (b - a))
            beta_param = 1 + 4 * ((b - m) / (b - a))
            draws = np.random.beta(alpha, beta_param, size=k) * (b - a) + a
            
        elif dist == 'beta':
            draws = np.random.beta(2, 2, size=k) * (b - a) + a
            
        elif dist == 'normal':
            sigma = (b - a) / 6
            draws = np.random.normal(loc=m, scale=sigma, size=k)

            mask_bad = (draws < a) | (draws > b)
            while mask_bad.any():
                draws[mask_bad] = np.random.normal(loc=m[mask_bad], scale=sigma[mask_bad])
                mask_bad = (draws < a) | (draws > b)
        else:
            raise ValueError(
                f"distribution must be 'uniform', 'pert', 'beta', or 'normal', got: '{distribution}'"
            )
            
        eps.loc[valid] = draws

    return V * (1.0 + eps)
