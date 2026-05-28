import cvxpy as cp # type: ignore[import]
import numpy as np# type: ignore[import]
import pandas as pd # type: ignore[import]
import logging

def mpc(max_charge_rate:float, max_discharge_rate:float, max_import_rate:float,
         max_export_rate:float, constraints:list, results:pd.DataFrame, H:int):
    forecast_median = cp.Parameter(shape=H, name="forecast_median", nonneg=True)
    forecast_lower = cp.Parameter(shape=H, name="forecast_lower", nonneg=True)
    forecast_upper = cp.Parameter(shape=H, name="forecast_upper", nonneg=True)
    soc_value = cp.Parameter(name="soc_value", nonneg=True)
    import_energy_price = cp.Parameter(shape = H, name="import_price", nonneg=True)
    export_energy_price = cp.Parameter(shape = H, name="export_price", nonneg=True)

    charge = cp.Variable(H, name="charge", nonneg=True)
    discharge = cp.Variable(H, name="discharge", nonneg=True)
    import_energy = cp.Variable(H, name="import_energy", nonneg=True)
    export_energy = cp.Variable(H, name="export_energy", nonneg=True)
    soc = cp.Variable(H+1, name="soc", nonneg=True)
    is_charging  = cp.Variable(H, boolean=True)  # 1=charging, 0=discharging
    is_importing = cp.Variable(H, boolean=True)  # 1=importing, 0=exporting

    constraints = constraints
    constraints+= [soc[0] == soc_value,
                   forecast_median + import_energy == export_energy + charge,
                   import_energy >= export_energy + charge - forecast_lower,
                    export_energy + charge >= forecast_upper,

                    charge    <= max_charge_rate    * is_charging,
                    discharge <= max_discharge_rate * (1 - is_charging),
                    
                    # import/export mutex
                    import_energy <= max_import_rate * is_importing,
                    export_energy <= max_export_rate * (1 - is_importing),
                   ]
    objective = cp.Maximize(cp.sum(export_energy * export_energy_price)-cp.sum(import_energy * import_energy_price))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI, verbose=False)