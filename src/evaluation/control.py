import cvxpy as cp # type: ignore[import]
import numpy as np# type: ignore[import]
import pandas as pd # type: ignore[import]
import logging
import sys
from src.config import file_config, residuals_config, control_config # type: ignore[import]
from src.data.data_acquisition import get_pv_system # type: ignore[import]
from scipy.stats import norm # type: ignore[import]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def mpc(max_charge_rate:float, max_discharge_rate:float, max_import_rate:float,
         max_export_rate:float, H:int, back_off:np.ndarray, battery_capacity:float)->dict:
    """Define the MPC optimization problem for PV self-consumption with battery storage.
    Args:
        max_charge_rate (float): Maximum charging power of the battery (kW).
        max_discharge_rate (float): Maximum discharging power of the battery (kW).
        max_import_rate (float): Maximum power that can be imported from the grid (kW
        max_export_rate (float): Maximum power that can be exported to the grid (kW).
        H (int): The control horizon length (number of time steps to optimize over).
        back_off (np.ndarray): An array of back-off values to adjust SOC constraints based on forecast uncertainty.
        battery_capacity (float): The total energy capacity of the battery (kWh).
        returns:
        dict: A dictionary containing the MPC problem and relevant variables for optimization."""
    forecast_median = cp.Parameter(shape=H, name="forecast_median", nonneg=True)
    forecast_lower = cp.Parameter(shape=H, name="forecast_lower", nonneg=True)
    forecast_upper = cp.Parameter(shape=H, name="forecast_upper", nonneg=True)
    soc_value = cp.Parameter(name="soc_value", nonneg=True)
    import_energy_price = cp.Parameter(shape = H, name="import_price", nonneg=True)
    export_energy_price = cp.Parameter(shape = H, name="export_price", nonneg=True)
    load = cp.Parameter(shape=H, name="load", nonneg=True)




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
    charge_q10 = cp.Variable(H, nonneg=True, name="charge_q10")
    charge_q50 = cp.Variable(H, nonneg=True, name="charge_q50")
    charge_q90 = cp.Variable(H, nonneg=True, name="charge_q90")
    discharge_q10 = cp.Variable(H, nonneg=True, name="discharge_q10")
    discharge_q50 = cp.Variable(H, nonneg=True, name="discharge_q50")
    discharge_q90 = cp.Variable(H, nonneg=True, name="discharge_q90")

   
    is_charging_q10 = cp.Variable(H, boolean=True)
    is_charging_q50 = cp.Variable(H, boolean=True)
    is_charging_q90 = cp.Variable(H, boolean=True)
    soc_q10       = cp.Variable(H+1, nonneg=True)
    soc_q50       = cp.Variable(H+1, nonneg=True)
    soc_q90       = cp.Variable(H+1, nonneg=True)
    is_charging_shared = cp.Variable(H, boolean=True)
    is_importing_shared = cp.Variable(H, boolean=True)

 

    constraints = []
    constraints += [
        is_charging_q10[0]  == is_charging_shared[0],
        is_charging_q50[0]  == is_charging_shared[0],
        is_charging_q90[0]  == is_charging_shared[0],
        is_importing_q10[0] == is_importing_shared[0],
        is_importing_q50[0] == is_importing_shared[0],
        is_importing_q90[0] == is_importing_shared[0],
        charge_q10[0]    == charge_q50[0],
        charge_q50[0]    == charge_q90[0],
        discharge_q10[0] == discharge_q50[0],
        discharge_q50[0] == discharge_q90[0],
        # initial SOC

        soc_q10[0] == soc_value,
        soc_q50[0] == soc_value,
        soc_q90[0] == soc_value,
        
        # SOC dynamics
        soc_q10[1:] == soc_q10[:-1] + (charge_q10 - discharge_q10) / battery_capacity,
        soc_q50[1:] == soc_q50[:-1] + (charge_q50 - discharge_q50) / battery_capacity,
        soc_q90[1:] == soc_q90[:-1] + (charge_q90 - discharge_q90) / battery_capacity,
        
        # SOC limits over horizon
        soc_q10[1:] >= control_config.min_soc + back_off,  # don't go too low
        soc_q50[1:] >= control_config.min_soc + back_off,  # don't go too low
        soc_q90[1:] >= control_config.min_soc + back_off,  # don't go too low
        soc_q10[1:] <= control_config.max_soc - back_off,  # don't go too high
        soc_q50[1:] <= control_config.max_soc - back_off,  # don't go too high
        soc_q90[1:] <= control_config.max_soc - back_off,  # don't go too high

        forecast_lower  + import_q10 + discharge_q10 == export_q10 + charge_q10 + load,
        forecast_median + import_q50 + discharge_q50 == export_q50 + charge_q50 + load,
        forecast_upper  + import_q90 + discharge_q90 == export_q90 + charge_q90 + load,

        # mutex - mode shared
        charge_q10    <= max_charge_rate    * is_charging_q10,
        charge_q50    <= max_charge_rate    * is_charging_q50,
        charge_q90    <= max_charge_rate    * is_charging_q90,

        discharge_q10 <= max_discharge_rate * (1 - is_charging_q10),
        discharge_q50 <= max_discharge_rate * (1 - is_charging_q50),
        discharge_q90 <= max_discharge_rate * (1 - is_charging_q90),

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
            - control_config.cycle_penalty * (
            control_config.q10_weight * cp.sum(charge_q10)
            + control_config.q50_weight * cp.sum(charge_q50)
            + control_config.q90_weight * cp.sum(charge_q90)
        )
        - control_config.import_penalty * (
            control_config.q10_weight * cp.sum(import_q10)
            + control_config.q50_weight * cp.sum(import_q50)
            + control_config.q90_weight * cp.sum(import_q90)
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
        "charge":           charge_q50,    # execute Q50
        "discharge":        discharge_q50,
        "load":             load,
        "import_energy":    import_q50,
        "export_energy":    export_q50,
        "soc":              soc_q50,
    }

def run_mpc(mpc:dict, t:int, results: pd.DataFrame, import_prices: np.ndarray, export_prices: np.ndarray,load: np.ndarray, current_soc: float, H: int, forecast_type: str = "probabilistic", is_residual: bool = True, point_forecast:str = "tft", kwp: float = 1.0)->dict:
    """Run the MPC optimization for a single time step.
    args:
        mpc (dict): The MPC problem and parameters.
        t (int): The current time step index.
        results (pd.DataFrame): A DataFrame containing the true values and forecast predictions.
        import_prices (np.ndarray): An array of import prices for the horizon.
        export_prices (np.ndarray): An array of export prices for the horizon.
        load (np.ndarray): An array of load values for the horizon.
        current_soc (float): The current state of charge of the battery.
        H (int): The control horizon length.
        forecast_type (str): The type of forecast used ("deterministic" or "probabilistic")
        is_residual (bool): Whether the forecast is a residual-corrected forecast or not.
        point_forecast (str): If using deterministic forecasts, which point forecast to use ("tft", "arima", "persistence")
        kwp (float): The kW rating of the PV system, used to scale the forecasts and errors to actual energy units.
    returns:     dict: A dictionary containing the optimized control actions and relevant information.
    """
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
    mpc["load"].value = load[t:t+H]
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

def compute_back_off_rolling(results: pd.DataFrame, t: int,
                              kwp: float, H: int,
                              forecast_type: str,
                              is_residual: bool,
                              point_forecast: str, battery_capacity: float) -> np.ndarray:
    """
    Rolling 30-day back-off using only past data.
    No lookahead bias — only uses errors up to time t.
    Aligns with 30-day corrector and CQR calibration windows.
    """

    
    if is_residual:
        pred_col = "corrected_median"
    elif forecast_type == "probabilistic":
        pred_col = "tft_pred"
    elif point_forecast == "tft":
        pred_col = "tft_pred"
    elif point_forecast == "arima":
        pred_col = "arima_pred"
    else:
        pred_col = "tft_pred"
    
    window_size  = 30 * 24
    window_start = max(0, t - window_size)
    
    # warmup — less than 7 days of data
    if t - window_start < 7 * 24:
        max_back_off = (control_config.max_soc - control_config.min_soc) / 8
        return np.full(H, max_back_off * 0.5)
    
    past_errors = (
        results["actual"].values[window_start:t] -
        results[pred_col].values[window_start:t]
    ) * kwp
    
    Qw = np.var(past_errors)
    z  = norm.ppf(1 - control_config.beta)
    
    back_off = np.array([
        z * np.sqrt(i * Qw) / battery_capacity
        for i in range(1, H+1)
    ])
    
    max_back_off = (control_config.max_soc - control_config.min_soc) / 8
    return np.clip(back_off, 0, max_back_off)

# what fraction of total energy consumed came from solar/battery
# rather than grid
def compute_self_consumption(df: pd.DataFrame, load_total: float) -> dict:
    total_import = df["actual_import"].sum()
    total_export = df["actual_export"].sum()
    total_pv     = df["actual_pv"].sum()
    total_charge = df["charge"].sum()
    total_discharge = df["discharge"].sum()
    
    # self consumption = PV used directly for load + PV used to charge battery
    # NOT PV - export (because export includes battery discharge)
    
    # PV directly consumed = PV - amount exported that came from PV only
    # simpler: self_consumption = 1 - (solar_export / total_pv)
    # but solar_export = total_export - battery_export
    # battery_export = discharge - (import used to charge) ← complex
    
    # Simplest correct formula:
    # self_consumption = (total_pv - max(0, total_export - total_discharge)) / total_pv
    
    solar_export = max(0.0, total_export - total_discharge)
    pv_self_consumed = max(0.0, total_pv - solar_export)
    
    self_consumption = pv_self_consumed / total_pv if total_pv > 0 else 0.0
    self_sufficiency = max(0.0, 1 - (total_import / load_total)) if load_total > 0 else 0.0
    
    return {
        "self_consumption": float(np.clip(self_consumption, 0, 1)),
        "self_sufficiency": float(np.clip(self_sufficiency, 0, 1)),
        "total_pv_kwh":     float(total_pv),
        "total_import_kwh": float(total_import),
        "total_export_kwh": float(total_export),
    }
def compute_soc_violations(df: pd.DataFrame) -> dict:
    """Count how often battery hits SOC limits."""
    min_violations = (df["soc_next"] < control_config.min_soc + 0.01).sum()
    max_violations = (df["soc_next"] > control_config.max_soc - 0.01).sum()
    
    return {
        "min_soc_violations": int(min_violations),
        "max_soc_violations": int(max_violations),
        "total_violations":   int(min_violations + max_violations),
        "violation_rate":     (min_violations + max_violations) / len(df)
    }
def simulate_control(results, initial_soc, H, forecast_type,
                     is_residual, point_forecast, kwp, household_id, battery_capacity):
    
    import_prices, export_prices = get_synthetic_prices(len(results))

    load = get_single_household_aligned(
        file_config.processed_load_data_path, 
        household_id, 
        len(results)
    )
    # initial conservative back-off
    if point_forecast == "persistence":
        initial_back_off = np.zeros(H)
    else:
        max_back_off = (control_config.max_soc - control_config.min_soc) / 8
        initial_back_off = np.full(H, max_back_off * 0.5)
    
    mpc_controller = mpc(
        max_charge_rate=control_config.max_charge_rate,
        max_discharge_rate=control_config.max_discharge_rate,
        max_import_rate=control_config.max_import_rate,
        max_export_rate=control_config.max_export_rate,
        H=H,
        back_off=initial_back_off, battery_capacity=battery_capacity
    )
    
    current_soc = initial_soc
    control_actions = []
    start_t = 1 if point_forecast == "persistence" else 0
    
    for t in range(start_t, len(results)-H):
        
        # rebuild MPC daily with updated rolling back-off
        if t % 24 == 0:
            back_off = compute_back_off_rolling(
                results, t, kwp, H,
                forecast_type, is_residual, point_forecast, battery_capacity
            )
            mpc_controller = mpc(
                max_charge_rate=control_config.max_charge_rate,
                max_discharge_rate=control_config.max_discharge_rate,
                max_import_rate=control_config.max_import_rate,
                max_export_rate=control_config.max_export_rate,
                H=H,
                back_off=back_off, battery_capacity=battery_capacity
            )
        
        action = run_mpc(
            mpc_controller, t, results,
            import_prices, export_prices,
            load, current_soc, H,
            forecast_type, is_residual, point_forecast, kwp
        )
        
        actual_gen    = results["actual"].values[t] * kwp
        actual_net    = actual_gen - action["charge"] + action["discharge"] - load[t]
        uncurtailed_export = max(actual_net, 0)
        actual_export = min(uncurtailed_export, control_config.max_export_rate)
        curtailment   = uncurtailed_export - actual_export
        actual_import = max(-actual_net, 0)
        actual_revenue = (
            actual_export * export_prices[t]
            - actual_import * import_prices[t]
            )

        action["actual_revenue"] = actual_revenue
        action["actual_export"]  = actual_export
        action["actual_import"]  = actual_import
        action["actual_pv"]      = actual_gen
        action["curtailment"]    = curtailment
        
        control_actions.append(action)
        current_soc = action["soc_next"]
    
    return pd.DataFrame(control_actions)
def rule_based_scheduling(results: pd.DataFrame, load: np.ndarray, 
                          kwp: float, import_prices: np.ndarray,
                          export_prices: np.ndarray, battery_capacity: float) -> pd.DataFrame:
    """
    Simple rule-based battery scheduler.
    Rule: charge when PV > load (excess solar), 
          discharge when PV < load (evening peak)
    No forecast needed — purely reactive.
    """
    control_actions = []
    current_soc = 0.5
    
    for t in range(len(results) - 24):
        actual_gen = results["actual"].values[t] * kwp
        current_load = load[t]
        net = actual_gen - current_load
        
        charge = 0.0
        discharge = 0.0
        
        if net > 0:
            # excess PV — charge battery
            charge = min(
                net,
                control_config.max_charge_rate,
                (control_config.max_soc - current_soc) * battery_capacity
            )
        elif net < 0:
            # deficit — discharge battery
            discharge = min(
                abs(net),
                control_config.max_discharge_rate,
                (current_soc - control_config.min_soc) * battery_capacity
            )
        
        # update SOC
        current_soc = current_soc + (charge - discharge) / battery_capacity
        current_soc = np.clip(current_soc, control_config.min_soc, control_config.max_soc)
        
        # compute actual revenue
        actual_net = actual_gen - charge + discharge - current_load
        uncurtailed_export = max(actual_net, 0)
        actual_export = min(uncurtailed_export, control_config.max_export_rate)
        curtailment = uncurtailed_export - actual_export
        actual_import = max(-actual_net, 0)
        actual_revenue = (
            actual_export * export_prices[t]
            - actual_import * import_prices[t]
        )

        control_actions.append({
            "t": t, "charge": charge, "discharge": discharge,
            "actual_export": actual_export, "actual_import": actual_import,
            "actual_revenue": actual_revenue, "actual_pv": actual_gen,
            "curtailment": curtailment, "soc": current_soc,
            "forecast_type": "rule_based"
        })
    
    return pd.DataFrame(control_actions)
def get_single_household(load_path:str, household_id:int)->np.ndarray:
    """Get the load data for a single household from the processed load dataset.
    Args:
        load_path (str): The path to the processed load dataset (parquet file).
        household_id (int): The ID of the household to retrieve.
    Returns:
        np.ndarray: An array of load values for the specified household."""
    load_df = pd.read_parquet(load_path)
    household_load = load_df[load_df["id"] == household_id].sort_values(by="datetime")["load_kWh"].values
    return household_load

def get_single_household_aligned(load_path: str,
                                  household_id: int,
                                  n_steps: int) -> np.ndarray:
    """
    Get load data for household, tiled/truncated to match n_steps.
    Date alignment not possible if load covers different year than results.
    """
    load_df = pd.read_parquet(load_path)
    
    # get first household if id=0 doesn't exist
    available_ids = load_df["id"].unique()
    if household_id not in available_ids:
        logging.warning(f"household_id={household_id} not found, using {available_ids[0]}")
        household_id = available_ids[0]
    
    household = load_df[load_df["id"] == household_id].sort_values("datetime")
    load_array = household["load_kWh"].values
    
    logging.info(f"Load raw: {len(load_array)} hours "
                 f"mean={load_array.mean():.4f} kWh/h "
                 f"sum={load_array.sum():.1f} kWh/year")
    
    # tile or truncate to match n_steps
    if len(load_array) < n_steps:
        # tile to cover full period
        repeats = (n_steps // len(load_array)) + 1
        load_array = np.tile(load_array, repeats)
    
    load_array = load_array[:n_steps]
    logging.info(f"Load aligned to {n_steps} steps: sum={load_array.sum():.1f} kWh")
    
    return load_array
def compute_curtailment(df: pd.DataFrame) -> dict:
    """Energy generated but neither consumed, stored, nor exportable
    within the grid connection limit."""
    total_curtailment = df["curtailment"].sum()
    total_pv = df["actual_pv"].sum()
    return {
        "total_curtailment_kwh": float(total_curtailment),
        "curtailment_rate": float(total_curtailment / total_pv) if total_pv > 0 else 0.0,
        "curtailment_hours": int((df["curtailment"] > 0.01).sum()),
    }
def get_synthetic_prices(n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic import and export prices for the simulation.
    This creates a typical daily pattern of import prices with higher prices during the day and lower prices at night, and a flat export price representing the SEG.
    Args:        n_steps (int): The total number of time steps for which to generate prices.
    Returns:
        tuple[np.ndarray, np.ndarray]: Two arrays containing the import prices and export prices for each time step."""
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
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "pretrained"
    house = sys.argv[2] if len(sys.argv) > 2 else "1"
    if house == "1":
        load_id = 8
    elif house == "2":
        load_id = 7
    logging.info(f"Running control simulation for mode={mode}, house={house}, load_id={load_id}")
    if mode == "pretrained":
        
        run_name = "control-pretrained"
        if house =="1":
            ensemble_file = "predictions_rolling_pretrained_ensemble.csv"
            truth_path = file_config.test_set
        elif house == "2":
            ensemble_file = "predictions_rolling_pretrained_ensemble2.csv"
            truth_path = file_config.household2_test_set

    elif mode == "pretrained_drifted":
        run_name = "control-pretrained-drifted"
        if house =="1":
            ensemble_file = "predictions_rolling_pretrained_drifted_ensemble.csv"
            truth_path = file_config.test_set_drifted
        elif house == "2":
            ensemble_file = "predictions_rolling_pretrained_drifted_ensemble2.csv"
            truth_path = file_config.household2_test_set_drifted        

    elif mode == "pretrained_shaded":
        
        run_name = "control-pretrained-shaded"
        if house =="1":
            ensemble_file = "predictions_rolling_pretrained_shaded_ensemble.csv"
            truth_path = file_config.test_set_shaded
        elif house == "2":
            ensemble_file = "predictions_rolling_pretrained_shaded_ensemble2.csv"
            truth_path = file_config.household2_test_set_shaded
    elif mode == "battery_sensitivity":
        run_name = "control-battery-sensitivity"
        if house =="1":
            ensemble_file = "predictions_rolling_pretrained_ensemble.csv"
            truth_path = file_config.test_set
        elif house == "2":
            ensemble_file = "predictions_rolling_pretrained_ensemble2.csv"
            truth_path = file_config.household2_test_set
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if house == "1":
        _, _, _, kwp = get_pv_system()
    elif house =="2":
        _, _, _, kwp = get_pv_system(max_kwp=2.5)
    results = pd.read_csv(f"{file_config.results_dir}/{ensemble_file}")
    results_table = []
    scenarios = {
        "persistence":  dict(forecast_type="point",         is_residual=False, point_forecast="persistence"),
        "arima":        dict(forecast_type="point",         is_residual=False, point_forecast="arima"),
        "tft_prob":     dict(forecast_type="probabilistic", is_residual=False, point_forecast="tft"),
        "tft_residual": dict(forecast_type="probabilistic", is_residual=True,  point_forecast="tft"),
    }
    import_prices, export_prices = get_synthetic_prices(len(results))
    load = get_single_household_aligned(
    file_config.processed_load_data_path, load_id, len(results)
    )


   
    if mode == "battery_sensitivity":
        for battery in control_config.battery_sensitivity_test:
            logging.info(f"Running rule-based control with battery_capacity={battery} kWh")
            df_rule = rule_based_scheduling(results, load, kwp, import_prices, export_prices, battery)
            rule_degradation = df_rule["charge"].sum() * control_config.cycle_penalty
            results_table.append({
                "scenario":        "rule_based",
                "mode":            mode,
                "house":           house,
                "apparent_rev":    round(df_rule["actual_revenue"].sum(), 2),
                "degradation":     round(rule_degradation, 2),
                "true_revenue":    round(df_rule["actual_revenue"].sum() - rule_degradation, 2),
                "self_consumption": None,
                "violations":      None,
                "violation_rate":  None,
                "annual_charge":   round(df_rule["charge"].sum(), 1),
            })
            logging.info(f"rule_based: actual=£{df_rule['actual_revenue'].sum():.4f}")
            df_rule.to_csv(f"{file_config.results_dir}/control_rule_based_{mode}_h{house}_battery_{battery}.csv", index=False)
            for scenario_name, scenario in scenarios.items():
                forecast_type  = str(scenario["forecast_type"])
                is_residual    = bool(scenario["is_residual"])
                point_forecast = str(scenario["point_forecast"])
                
                logging.info(f"Running scenario: {scenario_name} with battery_capacity={battery} kWh")
                df = simulate_control(
                    results, initial_soc=0.5, H=24,
                    forecast_type=forecast_type,
                    is_residual=is_residual,
                    point_forecast=point_forecast,
                    kwp=kwp, household_id=load_id, battery_capacity=battery
                )
                degradation_cost   = df["charge"].sum() * control_config.cycle_penalty
                violations = compute_soc_violations(df)
                load_total = load[:len(df)].sum()
                sc_metrics = compute_self_consumption(df, load_total)
                curt_metrics = compute_curtailment(df)
                logging.info(f"{scenario_name}: curtailment={curt_metrics['total_curtailment_kwh']:.2f} kWh "
                f"({curt_metrics['curtailment_rate']*100:.2f}%)")
                
                logging.info(f"{scenario_name}: degradation_cost=£{degradation_cost:.4f}")
                logging.info(f"{scenario_name}: self_consumption={sc_metrics['self_consumption']:.3f}")
                logging.info(f"{scenario_name}: self_sufficiency={sc_metrics['self_sufficiency']:.3f}")
                logging.info(f"{scenario_name}: violations={violations['total_violations']} ({violations['violation_rate']*100:.1f}%)")
                # collect results for summary table
                results_table.append({
                    "scenario":        scenario_name,
                    "mode":            mode,
                    "house":           house,
                    "apparent_rev":    round(df["actual_revenue"].sum(), 2),
                    "degradation":     round(degradation_cost, 2),
                    "true_revenue":    round(df["actual_revenue"].sum() - degradation_cost, 2),
                    "self_consumption": round(sc_metrics["self_consumption"], 3),
                    "violations":      violations["total_violations"],
                    "violation_rate":  round(violations["violation_rate"] * 100, 1),
                    "annual_charge":   round(df["charge"].sum(), 1),
                    "curtailment_kwh":  round(curt_metrics["total_curtailment_kwh"], 2),
                    "curtailment_rate": round(curt_metrics["curtailment_rate"] * 100, 2),
                })
                df.to_csv(f"{file_config.results_dir}/control_{scenario_name}_{mode}_h{house}_battery_{battery}.csv", index=False)
    else:
        df_rule = rule_based_scheduling(results, load, kwp, import_prices, export_prices, control_config.battery_capacity)
        rule_degradation = df_rule["charge"].sum() * control_config.cycle_penalty
        results_table.append({
            "scenario":        "rule_based",
            "mode":            mode,
            "house":           house,
            "apparent_rev":    round(df_rule["actual_revenue"].sum(), 2),
            "degradation":     round(rule_degradation, 2),
            "true_revenue":    round(df_rule["actual_revenue"].sum() - rule_degradation, 2),
            "self_consumption": None,
            "violations":      None,
            "violation_rate":  None,
            "annual_charge":   round(df_rule["charge"].sum(), 1),
        })
        logging.info(f"rule_based: actual=£{df_rule['actual_revenue'].sum():.4f}")
        df_rule.to_csv(f"{file_config.results_dir}/control_rule_based_{mode}_h{house}.csv", index=False)

        for name, scenario in scenarios.items():
            forecast_type  = str(scenario["forecast_type"])
            is_residual    = bool(scenario["is_residual"])
            point_forecast = str(scenario["point_forecast"])
            
            logging.info(f"Running scenario: {name}")
            df = simulate_control(
                results, initial_soc=0.5, H=24,
                forecast_type=forecast_type,
                is_residual=is_residual,
                point_forecast=point_forecast,
                kwp=kwp, household_id=load_id, battery_capacity=control_config.battery_capacity
            )
            degradation_cost   = df["charge"].sum() * control_config.cycle_penalty
            violations = compute_soc_violations(df)
            load_total = load[:len(df)].sum()
            sc_metrics = compute_self_consumption(df, load_total)
            
            logging.info(df[["charge","discharge","actual_export","actual_import","actual_revenue"]].sum())
            logging.info(f"{name}: actual=£{df['actual_revenue'].sum():.4f}")
            logging.info(f"{name}: degradation_cost=£{degradation_cost:.4f}")
            logging.info(f"{name}: self_consumption={sc_metrics['self_consumption']:.3f}")
            logging.info(f"{name}: self_sufficiency={sc_metrics['self_sufficiency']:.3f}")
            logging.info(f"{name}: violations={violations['total_violations']} ({violations['violation_rate']*100:.1f}%)")

            # collect results for summary table
            results_table.append({
                "scenario":        name,
                "mode":            mode,
                "house":           house,
                "apparent_rev":    round(df["actual_revenue"].sum(), 2),
                "degradation":     round(degradation_cost, 2),
                "true_revenue":    round(df["actual_revenue"].sum() - degradation_cost, 2),
                "self_consumption": round(sc_metrics["self_consumption"], 3),
                "violations":      violations["total_violations"],
                "violation_rate":  round(violations["violation_rate"] * 100, 1),
                "annual_charge":   round(df["charge"].sum(), 1),
            })
            df.to_csv(f"{file_config.results_dir}/control_{name}_{mode}_h{house}.csv", index=False)
    summary_df = pd.DataFrame(results_table)
    summary_path = f"{file_config.results_dir}/control_summary_{mode}_h{house}.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"\n=== Summary Table ===")
    logging.info(f"\n{summary_df.to_string()}")
    logging.info(f"Saved to {summary_path}")