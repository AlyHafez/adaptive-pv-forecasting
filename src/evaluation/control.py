import cvxpy as cp # type: ignore[import]
import numpy as np# type: ignore[import]
import pandas as pd # type: ignore[import]
import logging
from src.config import file_config, residuals_config, control_config # type: ignore[import]
from src.data.data_acquisition import get_pv_system # type: ignore[import]
from scipy.stats import norm # type: ignore[import]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def mpc(max_charge_rate:float, max_discharge_rate:float, max_import_rate:float,
         max_export_rate:float, H:int, back_off:np.ndarray)->dict:
    forecast_median = cp.Parameter(shape=H, name="forecast_median", nonneg=True)
    forecast_lower = cp.Parameter(shape=H, name="forecast_lower", nonneg=True)
    forecast_upper = cp.Parameter(shape=H, name="forecast_upper", nonneg=True)
    soc_value = cp.Parameter(name="soc_value", nonneg=True)
    import_energy_price = cp.Parameter(shape = H, name="import_price", nonneg=True)
    export_energy_price = cp.Parameter(shape = H, name="export_price", nonneg=True)

    # Shared action variables (here-and-now decisions)

    charge = cp.Variable(H, nonneg=True)
    discharge = cp.Variable(H, nonneg=True)
    is_charging  = cp.Variable(H,   boolean=True)


    # Scenario-specific recourse variables
    import_q10 = cp.Variable(H, nonneg=True, name="import_q10")
    import_q50 = cp.Variable(H, nonneg=True, name="import_q50")
    import_q90 = cp.Variable(H, nonneg=True, name="import_q90")
    export_q10 = cp.Variable(H, nonneg=True, name="export_q10")
    export_q50 = cp.Variable(H, nonneg=True, name="export_q50")
    export_q90 = cp.Variable(H, nonneg=True, name="export_q90")

    is_importing_q10 = cp.Variable(H, boolean=True)
    is_importing_q50 = cp.Variable(H, boolean=True)
    is_importing_q90 = cp.Variable(H, boolean=True)
    soc_q10       = cp.Variable(H+1, nonneg=True)
    soc_q50       = cp.Variable(H+1, nonneg=True)
    soc_q90       = cp.Variable(H+1, nonneg=True)


 

    constraints = []
    constraints += [
        # initial SOC
        soc_q10[0] == soc_value,
        soc_q50[0] == soc_value,
        soc_q90[0] == soc_value,
        
        # SOC dynamics
        soc_q10[1:] == soc_q10[:-1] + (charge - discharge) / control_config.battery_capacity,
        soc_q50[1:] == soc_q50[:-1] + (charge - discharge) / control_config.battery_capacity,
        soc_q90[1:] == soc_q90[:-1] + (charge - discharge) / control_config.battery_capacity,
        
        # SOC limits over horizon
        soc_q10[1:] >= control_config.min_soc + back_off,  # don't go too low
        soc_q50[1:] >= control_config.min_soc + back_off,  # don't go too low
        soc_q90[1:] >= control_config.min_soc + back_off,  # don't go too low
        soc_q10[1:] <= control_config.max_soc - back_off,  # don't go too high
        soc_q50[1:] <= control_config.max_soc - back_off,  # don't go too high
        soc_q90[1:] <= control_config.max_soc - back_off,  # don't go too high

        forecast_lower  + import_q10 + discharge == export_q10 + charge,
        forecast_median + import_q50 + discharge == export_q50 + charge,
        forecast_upper  + import_q90 + discharge == export_q90 + charge,

        # mutex - mode shared
        charge    <= max_charge_rate    * is_charging,
  
        discharge <= max_discharge_rate * (1 - is_charging),

        import_q10    <= max_import_rate    * is_importing_q10,
        import_q50    <= max_import_rate    * is_importing_q50,
        import_q90    <= max_import_rate    * is_importing_q90,
        export_q10    <= max_export_rate    * (1 - is_importing_q10),
        export_q50    <= max_export_rate    * (1 - is_importing_q50),
        export_q90    <= max_export_rate    * (1 - is_importing_q90),
    ]
    
    objective = cp.Maximize(
            control_config.q10_weight * (
                cp.sum(cp.multiply(export_q10, export_energy_price))
                - cp.sum(cp.multiply(import_q10, import_energy_price))
            )
            + control_config.q50_weight * (
                cp.sum(cp.multiply(export_q50, export_energy_price))
                - cp.sum(cp.multiply(import_q50, import_energy_price))
            )
            + control_config.q90_weight * (
                cp.sum(cp.multiply(export_q90, export_energy_price))
                - cp.sum(cp.multiply(import_q90, import_energy_price))
            )
        )

    problem = cp.Problem(objective, constraints)
    return {
        "problem":          problem,
        "forecast_median":  forecast_median,
        "forecast_lower":   forecast_lower,
        "forecast_upper":   forecast_upper,
        "soc_init":         soc_value,
        "import_price":     import_energy_price,
        "export_price":     export_energy_price,
        "charge":           charge,    # execute Q50
        "discharge":        discharge,
        "import_energy":    import_q50,
        "export_energy":    export_q50,
        "soc":              soc_q50,
    }

def run_mpc(mpc:dict, t:int, results: pd.DataFrame, import_prices: np.ndarray, export_prices: np.ndarray, current_soc: float, H: int, forecast_type: str = "probabilistic", is_residual: bool = True, point_forecast:str = "tft", kwp: float = 1.0)->dict:
    if forecast_type == "probabilistic":
        if is_residual:
            median = results["corrected_median"].values[t:t+H]
            lower  = results["corrected_lower"].values[t:t+H]
            upper  = results["corrected_upper"].values[t:t+H]
            
        else:
            median = results["tft_pred"].values[t:t+H]
            lower  = results["tft_lower"].values[t:t+H]
            upper  = results["tft_upper"].values[t:t+H]
    else:
        if is_residual:
            median = results["corrected_median"].values[t:t+H]
        else:
            if point_forecast == "tft":
                median = results["tft_pred"].values[t:t+H]
            elif point_forecast == "arima":
                median = results["arima_pred"].values[t:t+H]
            elif point_forecast == "persistence":
                if t == 0:
                    raise ValueError("Persistence forecast not available at t=0")
                median = results["actual"].values[t-1:t-1+H]
        
            else:
                raise ValueError(f"Unknown point_forecast type: {point_forecast}")
        # point forecasts have no quantiles
        lower  = median.copy()
        upper  = median.copy()

    # fix quantile crossing for all cases
    lower  = np.minimum(lower, median)
    upper  = np.maximum(upper, median)

    # assign
    mpc["forecast_median"].value = median *kwp
    mpc["forecast_lower"].value  = lower * kwp
    mpc["forecast_upper"].value  = upper * kwp
    mpc["soc_init"].value = current_soc
    mpc["import_price"].value = import_prices[t:t+H]
    mpc["export_price"].value = export_prices[t:t+H]

    mpc["problem"].solve(solver=cp.GUROBI, verbose=False, reoptimize=True)

    if mpc["problem"].status == cp.OPTIMAL:
        return {
            "t":                t,
            "charge":          mpc["charge"].value[0],
            "discharge":       mpc["discharge"].value[0],
            "import_energy":   mpc["import_energy"].value[0],
            "export_energy":   mpc["export_energy"].value[0],
            "soc_next":        mpc["soc"].value[1],
            "planned_revenue": mpc["problem"].value,
            "forecast_type":   forecast_type,
        }

    else:
        logging.warning(f"t={t} status: {mpc['problem'].status}")
        raise ValueError(f"MPC optimization failed at time {t} with status {mpc['problem'].status}")

def compute_back_off(results: pd.DataFrame, kwp: float, H: int, 
                     forecast_type: str, is_residual: bool, 
                     point_forecast: str) -> np.ndarray:
    
    # get correct forecast column
    if forecast_type == "probabilistic":
        if is_residual:
            pred_col = "corrected_median"
        else:
            pred_col = "tft_pred"
    else:
        if point_forecast == "tft":
            pred_col = "tft_pred"
        elif point_forecast == "arima":
            pred_col = "arima_pred"
        elif point_forecast == "persistence":
            pred_col = None  # handle separately
    
    if pred_col is None:
        # persistence - use previous actual as forecast
        forecast_errors = (results["actual"] - results["actual"].shift(1)) * kwp
    else:
        forecast_errors = (results["actual"] - results[pred_col]) * kwp
    
    Qw = forecast_errors.dropna().var()
    z  = norm.ppf(1 - control_config.beta)
    
    back_off = np.array([
        z * np.sqrt(i * Qw) / control_config.battery_capacity
        for i in range(1, H+1)
    ])
    
    max_back_off = (control_config.max_soc - control_config.min_soc) / 4
    return np.clip(back_off, 0, max_back_off)
def simulate_control(results: pd.DataFrame, initial_soc: float, H: int, forecast_type: str = "deterministic", is_residual: bool = True, point_forecast:str = "tft", kwp: float = 1.0)->pd.DataFrame:
    import_prices, export_prices = get_synthetic_prices(len(results))
    
    back_off = compute_back_off(results, kwp, H, forecast_type, is_residual, point_forecast)
    
    logging.info(f"back_off range: {back_off.min():.4f} to {back_off.max():.4f}")
    
    mpc_controller = mpc(
        max_charge_rate=control_config.max_charge_rate,
        max_discharge_rate=control_config.max_discharge_rate,
        max_import_rate=control_config.max_import_rate,
        max_export_rate=control_config.max_export_rate,
        H=H,
        back_off=back_off  # pass in
    )
    current_soc = initial_soc
    control_actions = []
    start_t = 1 if point_forecast == "persistence" else 0

    for t in range(start_t, len(results)-H):
        action = run_mpc(mpc_controller, t, results, import_prices, export_prices, current_soc, H, forecast_type, is_residual, point_forecast, kwp)
        
        actual_gen = results["actual"].values[t] *kwp
        actual_net = actual_gen - action["charge"] + action["discharge"]
        actual_export = max(actual_net, 0)
        actual_import = max(-actual_net, 0)
        actual_revenue = (
            actual_export * export_prices[t]
            - actual_import * import_prices[t]
        )
        action["actual_revenue"] = actual_revenue
        action["actual_export"]  = actual_export
        action["actual_import"]  = actual_import
        action["actual_pv"]      = actual_gen
        
        control_actions.append(action)
        current_soc = action["soc_next"]
    
    return pd.DataFrame(control_actions)

def get_synthetic_prices(n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    # typical UK daily pattern, repeated
    daily_import = np.array([
        0.10, 0.10, 0.10, 0.10,  # 00-04 cheap overnight
        0.12, 0.15, 0.20, 0.28,  # 04-08 morning ramp
        0.30, 0.28, 0.25, 0.22,  # 08-12 morning peak
        0.20, 0.18, 0.18, 0.20,  # 12-16 midday
        0.25, 0.32, 0.35, 0.30,  # 16-20 evening peak
        0.22, 0.18, 0.14, 0.10,  # 20-24 evening drop
    ])
    
    daily_export = np.full(24, 0.15)  # flat 15p SEG
    
    # repeat for n_steps
    n_days = n_steps // 24 + 1
    import_prices = np.tile(daily_import, n_days)[:n_steps]
    export_prices = np.tile(daily_export, n_days)[:n_steps]
    
    return import_prices, export_prices

if __name__ == "__main__":
    _, _, _, kwp = get_pv_system()
    results = pd.read_csv(f"{file_config.results_dir}/results/predictions_rolling_finetuned_ensemble.csv")
    
    scenarios = {
        "persistence":  dict(forecast_type="point",         is_residual=False, point_forecast="persistence"),
        "tft_point":    dict(forecast_type="point",         is_residual=False, point_forecast="tft"),
        "tft_prob":     dict(forecast_type="probabilistic", is_residual=False, point_forecast="tft"),
        "tft_residual": dict(forecast_type="probabilistic", is_residual=True,  point_forecast="tft"),
    }
    
    for name, kwargs in scenarios.items():
        logging.info(f"Running scenario: {name}")
        df = simulate_control(results, initial_soc=0.5, H=24, kwp=kwp, **kwargs)
        logging.info(df[["charge","discharge","actual_export","actual_import","actual_revenue"]].sum())
        logging.info(f"{name}: planned=£{df['planned_revenue'].sum():.4f} actual=£{df['actual_revenue'].sum():.4f}")
        df.to_csv(f"{file_config.results_dir}/control_{name}.csv", index=False)