# --- START OF FILE ---

# %%
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.ticker as mtick  # Import ticker (though plots reverted to absolute)
import copy
import os

# --- General Configuration ---
style.use("seaborn-whitegrid")  # Keep specific version
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.titlesize"] = 18
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.format"] = "jpg"
SAVE_PDF_TOO = False

# --- Simulation Time & Assumptions ---
T_OBSERVATION_DAYS = 42
AVG_STAY_X_DAYS = 7
AVG_STAY_Y_DAYS = 10
T_SIM_END = T_OBSERVATION_DAYS * 3

# --- Non-Autonomous Parameters ---
DELTA_THETA = 0.15
LAMBDA_DECAY = 0.05
A_PEAK_DAY = 65
A_PEAK_WIDTH = 15
A_PEAK_SCALE = 1.5

# --- Sensitivity Analysis Configuration ---
SENSITIVITY_RANGE = np.linspace(0.7, 1.3, 7)

# --- Output Directories ---
OUTPUT_DIR_BASE = "simulation_results"
OUTPUT_DIR_PLOTS = os.path.join(OUTPUT_DIR_BASE, "plots")
OUTPUT_DIR_CSV = os.path.join(OUTPUT_DIR_BASE, "csv_data")

if not os.path.exists(OUTPUT_DIR_BASE):
    os.makedirs(OUTPUT_DIR_BASE)
if not os.path.exists(OUTPUT_DIR_PLOTS):
    os.makedirs(OUTPUT_DIR_PLOTS)
if not os.path.exists(OUTPUT_DIR_CSV):
    os.makedirs(OUTPUT_DIR_CSV)

# --- Helper Functions ---


def load_and_preprocess_data(filepath):
    """Loads and preprocesses patient data."""
    try:
        df = pd.read_csv(filepath, sep=',', skipinitialspace=True)
        df.columns = df.columns.str.strip()
        required_cols = ["SRNI", "UCI", "EXITUS"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"Missing columns: {[c for c in required_cols if c not in df.columns]}"
            )
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        print(f"Data loaded: {len(df)} patients from '{filepath}'.")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found '{filepath}'")
        return None
    except Exception as e:
        print(f"ERROR processing file '{filepath}': {e}")
        return None


def estimate_parameters_from_data(df, t_observation, avg_stay_x, avg_stay_y):
    """Estimates base model parameters from data, FORCING eta=0, nu=0."""
    # --- CORRECTED: Check df before calculating n_total ---
    if df is None:
        print("ERROR: Input DataFrame is None in estimate_parameters_from_data.")
        return None  # Return early if df is None

    n_total = len(df)
    if n_total == 0:
        print("Warning: No patients found after processing data (DataFrame was empty).")
        return None  # Return if df was empty

    n_niv = df["SRNI"].sum()
    df_went_to_niv = df[df["SRNI"] == 1].copy()
    n_icu_from_y = df_went_to_niv.loc[df_went_to_niv["UCI"] == 1, "UCI"].sum()
    n_exitus_from_y_no_uci = df_went_to_niv.loc[
        (df_went_to_niv["UCI"] == 0) & (df_went_to_niv["EXITUS"] == 1)
    ].shape[0]
    n_recovered_from_y = len(df_went_to_niv) - n_icu_from_y - n_exitus_from_y_no_uci
    df_stayed_no_niv = df[df["SRNI"] == 0].copy()
    # These are 0 by definition/observation in the provided context
    n_icu_from_x = 0
    n_exitus_from_x = 0
    n_recovered_from_x = len(df_stayed_no_niv)
    epsilon_rate = 1e-9  # Small value to avoid division by zero if needed

    if t_observation <= 0:
        A_avg = 0.0
        print("Warning: t_observation <= 0.")
    else:
        A_avg = n_total / t_observation

    # Rates from X
    if len(df_stayed_no_niv) > 0 and avg_stay_x > 0:
        total_out_rate_x = 1.0 / avg_stay_x
        frac_go_y = n_niv / n_total if n_total > 0 else 0
        frac_rec_x = (
            n_recovered_from_x / n_total if n_total > 0 else 0
        )  # == (n_total - n_niv) / n_total
        alpha = frac_go_y * total_out_rate_x
        rho = frac_rec_x * total_out_rate_x
        eta = 0.0  # Enforce eta=0
        nu = 0.0  # Enforce nu=0
    else:  # Fallback if no X patients or invalid avg_stay_x
        alpha, rho, eta, nu = epsilon_rate, epsilon_rate, 0.0, 0.0

    # Rates from Y
    if n_niv > 0 and avg_stay_y > 0:
        total_out_rate_y = 1.0 / avg_stay_y
        frac_icu_from_y = n_icu_from_y / n_niv if n_niv > 0 else 0
        frac_exitus_direct_from_y = n_exitus_from_y_no_uci / n_niv if n_niv > 0 else 0
        frac_rec_from_y = n_recovered_from_y / n_niv if n_niv > 0 else 0
        gamma = frac_icu_from_y * total_out_rate_y
        epsilon = frac_exitus_direct_from_y * total_out_rate_y
        theta_0 = frac_rec_from_y * total_out_rate_y
    else:  # Fallback if no Y patients or invalid avg_stay_y
        gamma, epsilon, theta_0 = epsilon_rate, epsilon_rate, epsilon_rate

    # Assemble parameters dictionary, ensuring fixed values
    params = {
        "A_avg": A_avg,
        "alpha": alpha,
        "rho": rho,
        "eta": 0.0,
        "nu": 0.0,  # Enforced 0s
        "gamma": gamma,
        "theta_0": theta_0,
        "epsilon": epsilon,
        "delta": 0.0,
        "gamma_0": 0.0,
        "epsilon_0": 0.0,  # Fixed to 0 by model def
        "Delta_theta": DELTA_THETA,
        "lambda_decay": LAMBDA_DECAY,
        "A_peak_day": A_PEAK_DAY,
        "A_peak_width": A_PEAK_WIDTH,
        "A_peak_scale": A_PEAK_SCALE,
    }
    print("\n--- Base Parameters Estimated (eta=0, nu=0, delta=0 enforced) ---")
    # Print key estimated and fixed parameters
    print(
        f"Using params: A_avg={params['A_avg']:.4f}, alpha={params['alpha']:.4f}, rho={params['rho']:.4f}, "
        f"eta={params['eta']:.1f}, nu={params['nu']:.1f}, gamma={params['gamma']:.4f}, "
        f"theta_0={params['theta_0']:.4f}, epsilon={params['epsilon']:.4f}, delta={params['delta']:.1f}, ..."
    )
    print("---------------------------------------------------------------------")
    return params


def gaussian_peak(t, base, center, width, scale):
    """Calculates admission rate with a Gaussian peak."""
    if width <= 0:
        return np.maximum(base, 0)
    peak_val = base * (1 + scale * np.exp(-((t - center) ** 2) / (2 * width**2)))
    return np.maximum(peak_val, 0)  # Ensure non-negative


def ode_system(t, y, params, A_func=None, use_theta_t=False):
    """Defines the ODE system. NOTE: Ensures eta=0, nu=0, delta=0 within function."""
    X, Y, Z, W, R = y
    p = params
    # Extract parameters safely
    A_avg = p.get("A_avg", 0)
    alpha = p.get("alpha", 0)
    rho = p.get("rho", 0)
    gamma = p.get("gamma", 0)
    epsilon = p.get("epsilon", 0)
    theta_0 = p.get("theta_0", 0)
    gamma_0 = p.get("gamma_0", 0)
    epsilon_0 = p.get("epsilon_0", 0)
    Delta_theta = p.get("Delta_theta", 0)
    lambda_decay = p.get("lambda_decay", 0)

    # --- Enforce model constraints directly here for safety ---
    eta = 0.0
    nu = 0.0
    delta = 0.0
    # Ensure modifiers are within bounds (redundant if params fixed, but safe)
    gamma_0 = max(0, min(1, p.get("gamma_0", 0)))
    epsilon_0 = max(0, min(1, p.get("epsilon_0", 0)))

    # Calculate time-dependent A(t) if applicable
    if callable(A_func):
        At = A_func(
            t,
            A_avg,
            p.get("A_peak_day", 0),
            p.get("A_peak_width", 1),
            p.get("A_peak_scale", 0),
        )
    else:
        At = A_avg
    At = max(0, At)  # Ensure non-negative admission rate

    # Calculate time-dependent theta(t) if applicable
    # Note: Delta_theta or lambda_decay might be 0 if running autonomous case
    if use_theta_t and p.get("Delta_theta", 0) != 0 and p.get("lambda_decay", 0) > 0:
        thetat = theta_0 + p["Delta_theta"] * np.exp(-p["lambda_decay"] * t)
    else:
        thetat = theta_0
    thetat = max(0, thetat)  # Ensure non-negative recovery rate

    # Ensure primary rates are non-negative
    alpha = max(0, alpha)
    rho = max(0, rho)
    gamma = max(0, gamma)
    epsilon = max(0, epsilon)

    # Define ODEs (using enforced eta, nu, delta, gamma_0, epsilon_0)
    dXdt = At - (alpha + rho + eta + nu) * X
    dYdt = alpha * X - (gamma * (1 - gamma_0) + thetat + epsilon * (1 - epsilon_0)) * Y
    dZdt = gamma * (1 - gamma_0) * Y + eta * X - delta * Z
    dWdt = epsilon * (1 - epsilon_0) * Y + nu * X
    dRdt = thetat * Y + rho * X
    return [dXdt, dYdt, dZdt, dWdt, dRdt]


def run_simulation(t_span, t_eval, y0, params, A_func=None, use_theta_t=False):
    """Runs solve_ivp with specified configurations."""
    # Make a deep copy to store the exact params used for this run
    sim_params = copy.deepcopy(params)
    # Add flags/functions used to the dict for easy checking later
    sim_params["_actual_A_func"] = A_func
    sim_params["_actual_use_theta_t"] = use_theta_t

    sol = solve_ivp(
        fun=ode_system,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=(sim_params, A_func, use_theta_t),  # Pass dict, func, flag
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )
    if not sol.success:
        print(f"Warning: ODE solver status {sol.status}, Msg: {sol.message}")
    # Attach the params dict used to the solution object
    sol.params_used = sim_params
    return sol


def save_plot(fig, filename, output_dir):
    """Saves the current figure."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    default_format = plt.rcParams["savefig.format"]
    fig.savefig(
        os.path.join(output_dir, f"{filename}.{default_format}"),
        dpi=plt.rcParams["savefig.dpi"],
    )
    if SAVE_PDF_TOO and default_format != "pdf":
        fig.savefig(
            os.path.join(output_dir, f"{filename}.pdf")
        )  # Save PDF also if requested


# --- Plotting Functions ---
# NOTE: Functions plot_time_series, plot_sensitivity, plot_sensitivity_single remain UNCHANGED
# (Their implementation relies on the data passed to them)


def plot_time_series(
    sol, title, filename, output_dir_plots, params_dict=None, proportions=None
):
    """Generates and saves a time series plot with ABSOLUTE numbers."""
    t = sol.t
    X, Y, Z, W, R = sol.y
    if not (len(t) == len(X) == len(Y) == len(Z) == len(W) == len(R)):
        print(f"Plot error '{title}': Inconsistent lengths.")
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    # Corrected labels to match figure example
    ax.plot(t, X, label="No NIV (X)", color=colors[0], lw=2.5)
    ax.plot(t, Y, label="NIV (Y)", color=colors[1], lw=2.5)
    ax.plot(t, Z, label="Cumulative ICU (Z)", color=colors[2], linestyle="--", lw=2)
    ax.plot(t, W, label="Cumulative Exitus (W)", color=colors[3], linestyle=":", lw=2)
    ax.plot(
        t, R, label="Cumulative Recovered (R)", color=colors[4], linestyle="-.", lw=2
    )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of Patients")
    ax.set_title(title)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        title="Compartments",
    )
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    # Parameter/Proportion text boxes
    if params_dict:
        param_text_lines = []
        p = params_dict
        # Use flags stored in params_dict by run_simulation
        if p.get("_actual_A_func") == gaussian_peak:
            param_text_lines.append(f"A(t): Gaussian Peak")
        elif "A_avg" in p:
            param_text_lines.append(f"$A(t) = A_{{avg}} = {p['A_avg']:.3f}$")
        else:
            param_text_lines.append("A(t): N/A")

        if p.get("_actual_use_theta_t"):
            param_text_lines.append(
                f"$\\theta(t) = \\theta_0 + {p.get('Delta_theta',0):.2f}e^{{-{p.get('lambda_decay',0):.2f} t}}$"
            )
        elif "theta_0" in p:
            param_text_lines.append(
                f"$\\theta(t) = \\theta_0 = {p.get('theta_0', 0):.3f}$"
            )
        else:
            param_text_lines.append("$\\theta(t)$: N/A")
        ax.text(
            0.03,
            0.97,
            "\n".join(param_text_lines),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="aliceblue", alpha=0.8),
        )
    if proportions:
        z = proportions.get("Z%", np.nan)
        w = proportions.get("W%", np.nan)
        r = proportions.get("R%", np.nan)
        if np.isnan(z) or np.isnan(w) or np.isnan(r):
            prop_text = "Final Props: N/A"
        # Label updated for clarity
        else:
            prop_text = f"Final % (vs Z+W+R):\n  ICU Outcome: {z:.1f}%\n  Mortality (Direct): {w:.1f}%\n  Recovery Outcome: {r:.1f}%"
        ax.text(
            0.97,
            0.03,
            prop_text,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.8),
        )
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    save_plot(fig, filename, output_dir_plots)
    # plt.close(fig)


def plot_sensitivity(
    results, param_name, param_values, title, filename, output_dir_plots
):
    """Generates and saves a sensitivity analysis plot (absolute values)."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(title, fontsize=16, y=0.98)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 3))
    outcomes = ["Z_final", "W_final", "R_final"]
    ylabels = [
        "Final Cum. ICU (Z)",
        "Final Cum. Exitus (W)",
        "Final Cum. Recovered (R)",
    ]
    # Identify base parameter value correctly from the middle of the *range*
    base_param_val = None
    base_param_idx = -1
    if len(param_values) > 0:
        # Find index closest to the center (handles odd/even lengths)
        center_target_val = np.median(param_values)  # Robust for float ranges
        base_param_idx = np.argmin(np.abs(np.array(param_values) - center_target_val))
        base_param_val = param_values[base_param_idx]

    for i, (ax, outcome, ylabel) in enumerate(zip(axes, outcomes, ylabels)):
        if (
            outcome not in results
            or not isinstance(results[outcome], (list, np.ndarray))
            or len(results[outcome]) != len(param_values)
        ):
            print(
                f"Plot {title} warn: {outcome} data issue. Len expected {len(param_values)}, got {len(results.get(outcome, []))}."
            )
            continue
        vals = results[outcome]
        ax.plot(
            param_values,
            vals,
            marker="o",
            linestyle="-",
            color=colors[i],
            lw=2,
            markersize=6,
        )
        if (
            base_param_val is not None
            and base_param_idx >= 0
            and base_param_idx < len(vals)
            and not np.isnan(vals[base_param_idx])
        ):
            ax.scatter(
                base_param_val,
                vals[base_param_idx],
                color="red",
                s=60,
                zorder=5,
                label=f"Base {param_name}={base_param_val:.3f}",
            )
            ax.axvline(base_param_val, color="grey", linestyle=":", lw=1.5)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.6)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0 and base_param_val is not None:
            ax.legend(fontsize=9)
        if i == len(axes) - 1:
            ax.set_xlabel(f"Param Value ({param_name})", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, filename, output_dir_plots)
    #plt.close(fig)


def plot_sensitivity_single(
    values, param_name, param_range, title, ylabel, filename, output_dir_plots, base_val
):
    """Plots sensitivity for a single variable (absolute values)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if not isinstance(values, (list, np.ndarray)) or len(values) != len(param_range):
        print(
            f"Plot sensitivity single '{title}' error: Data mismatch (len {len(values)} vs {len(param_range)})."
        )
        return
    ax.plot(param_range, values, marker="o", linestyle="-", color="purple", lw=2)

    if base_val is not None and len(param_range) > 0 and len(values) > 0:
        param_range_np = np.array(param_range)
        center_idx = np.argmin(
            np.abs(param_range_np - base_val)
        )  # Robust index finding
        if (
            center_idx >= 0
            and center_idx < len(values)
            and np.isclose(param_range_np[center_idx], base_val)
            and not np.isnan(values[center_idx])
        ):
            ax.scatter(
                param_range_np[center_idx],
                values[center_idx],
                color="red",
                s=60,
                zorder=5,
                label=f"Base {param_name}={base_val:.3f}",
            )
            ax.axvline(param_range_np[center_idx], color="grey", linestyle=":", lw=1.5)
            ax.legend()
        else:
            print(
                f"Warn plot_sens_single '{title}': Base value {base_val} not found or data NaN at index {center_idx}."
            )

    ax.set_xlabel(f"Param Value ({param_name})")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_plot(fig, filename, output_dir_plots)
    #plt.close(fig)


# --- UPDATED CSV Export Functions (Using versions from previous step) ---
def export_simulation_to_csv(sol, filename_base, output_dir_csv):
    """Exports comprehensive time series data to CSV, adding clinically relevant normalizations.
    Retrieves parameters and functions used from the `sol` object.
    """
    # --- This function remains IDENTICAL to the one provided in the previous answer ---
    # --- SNIP - Re-paste the full function code here if starting fresh ---
    if not os.path.exists(output_dir_csv):
        try:
            os.makedirs(output_dir_csv)
        except OSError as e:
            print(f"Error creating CSV dir '{output_dir_csv}': {e}")
            return
    t = sol.t
    X, Y, Z, W, R = sol.y
    if len(t) == 0:
        print(f"Warning: No time points export for {filename_base}.")
        return
    params = getattr(sol, "params_used", {})
    p = params
    A_func = p.get("_actual_A_func", None)
    use_theta_t_actual = p.get("_actual_use_theta_t", False)
    df_export = pd.DataFrame({"t": t, "X": X, "Y": Y, "Z": Z, "W": W, "R": R})
    Total = X + Y + Z + W + R
    df_export["Total_All_Compartments"] = Total
    Total_safe = np.maximum(Total, 1e-9)
    df_export["X_perc_vs_TotalAll"] = (X / Total_safe) * 100
    df_export["Y_perc_vs_TotalAll"] = (Y / Total_safe) * 100
    df_export["Z_perc_vs_TotalAll"] = (Z / Total_safe) * 100
    df_export["W_perc_vs_TotalAll"] = (W / Total_safe) * 100
    df_export["R_perc_vs_TotalAll"] = (R / Total_safe) * 100
    max_X = np.max(X) if X.size > 0 else 0
    max_X_safe = np.maximum(max_X, 1e-9)
    max_Y = np.max(Y) if Y.size > 0 else 0
    max_Y_safe = np.maximum(max_Y, 1e-9)
    max_Z = np.max(Z) if Z.size > 0 else 0
    max_Z_safe = np.maximum(max_Z, 1e-9)
    max_W = np.max(W) if W.size > 0 else 0
    max_W_safe = np.maximum(max_W, 1e-9)
    max_R = np.max(R) if R.size > 0 else 0
    max_R_safe = np.maximum(max_R, 1e-9)
    df_export["X_perc_vs_max"] = (X / max_X_safe) * 100
    df_export["Y_perc_vs_max"] = (Y / max_Y_safe) * 100
    df_export["Z_perc_vs_max"] = (Z / max_Z_safe) * 100
    df_export["W_perc_vs_max"] = (W / max_W_safe) * 100
    df_export["R_perc_vs_max"] = (R / max_R_safe) * 100
    A_avg = p.get("A_avg", 0)
    A_array = np.full_like(t, A_avg)
    if callable(A_func):
        try:
            A_array = A_func(
                t,
                A_avg,
                p.get("A_peak_day", 0),
                p.get("A_peak_width", 1),
                p.get("A_peak_scale", 0),
            )
        except TypeError:
            A_array = np.array(
                [
                    A_func(
                        ti,
                        A_avg,
                        p.get("A_peak_day", 0),
                        p.get("A_peak_width", 1),
                        p.get("A_peak_scale", 0),
                    )
                    for ti in t
                ]
            )
    A_array = np.maximum(A_array, 0)
    if len(t) > 1:
        dt = np.diff(t, prepend=t[0])
        dt[0] = t[1] - t[0] if t[0] == 0 else dt[0]
    elif len(t) == 1:
        dt = np.array([t[0]])
    else:
        dt = np.array([0])
    Cum_A = np.cumsum(A_array * dt)
    df_export["Admissions_A(t)"] = A_array
    df_export["Cumulative_Admissions"] = Cum_A
    Cum_A_safe = np.maximum(Cum_A, 1e-9)
    df_export["Z_perc_vs_Cum_Admissions"] = (Z / Cum_A_safe) * 100
    df_export["W_perc_vs_Cum_Admissions"] = (W / Cum_A_safe) * 100
    df_export["R_perc_vs_Cum_Admissions"] = (R / Cum_A_safe) * 100
    Active_Patients = X + Y
    Active_Patients_safe = np.maximum(Active_Patients, 1e-9)
    df_export["Active_Patients_XY"] = Active_Patients
    df_export["Y_perc_vs_ActiveXY"] = (Y / Active_Patients_safe) * 100
    gamma = p.get("gamma", 0)
    epsilon = p.get("epsilon", 0)
    rho = p.get("rho", 0)
    theta_0 = p.get("theta_0", 0)
    Delta_theta = p.get("Delta_theta", 0)
    lambda_decay = p.get("lambda_decay", 0)
    gamma_0 = p.get("gamma_0", 0)
    epsilon_0 = p.get("epsilon_0", 0)
    if use_theta_t_actual:
        thetat_array = np.maximum(theta_0 + Delta_theta * np.exp(-lambda_decay * t), 0)
    else:
        thetat_array = np.full_like(t, max(0, theta_0))
    df_export["Theta_t_used"] = thetat_array
    eta = 0.0
    nu = 0.0
    delta = 0.0
    gamma_0 = max(0, min(1, gamma_0))
    epsilon_0 = max(0, min(1, epsilon_0))
    Incidence_Z = max(0, gamma) * max(0, 1 - gamma_0) * Y + max(0, eta) * X
    Incidence_W = max(0, epsilon) * max(0, 1 - epsilon_0) * Y + max(0, nu) * X
    Incidence_R = thetat_array * Y + max(0, rho) * X
    df_export["Incidence_Z"] = Incidence_Z
    df_export["Incidence_W"] = Incidence_W
    df_export["Incidence_R"] = Incidence_R
    ordered_cols = [
        "t",
        "X",
        "Y",
        "Z",
        "W",
        "R",
        "Active_Patients_XY",
        "Y_perc_vs_ActiveXY",
        "Admissions_A(t)",
        "Cumulative_Admissions",
        "Z_perc_vs_Cum_Admissions",
        "W_perc_vs_Cum_Admissions",
        "R_perc_vs_Cum_Admissions",
        "Total_All_Compartments",
        "X_perc_vs_TotalAll",
        "Y_perc_vs_TotalAll",
        "Z_perc_vs_TotalAll",
        "W_perc_vs_TotalAll",
        "R_perc_vs_TotalAll",
        "X_perc_vs_max",
        "Y_perc_vs_max",
        "Z_perc_vs_max",
        "W_perc_vs_max",
        "R_perc_vs_max",
        "Incidence_Z",
        "Incidence_W",
        "Incidence_R",
        "Theta_t_used",
    ]
    valid_ordered_cols = [col for col in ordered_cols if col in df_export.columns]
    df_export = df_export[valid_ordered_cols]
    filepath = os.path.join(output_dir_csv, f"{filename_base}_data_full_export.csv")
    try:
        df_export.to_csv(filepath, index=False, float_format="%.5f")
    except Exception as e:
        print(f"ERROR: Failed export to '{filepath}': {e}")
    final_cum_A = Cum_A[-1] if len(Cum_A) > 0 else 0
    return final_cum_A


def export_sensitivity_results_to_csv(
    results_dict,
    param_name_str,
    param_values,
    filename_base,
    output_dir_csv,
    Cum_A_final_values=None,
):
    """Exports summarized sensitivity results, adding Y peak per total admission.
    Uses 'Z_final_perc_vs_Outcomes' etc. for final proportion naming.
    """
    # --- This function remains IDENTICAL to the one provided in the previous answer ---
    # --- SNIP - Re-paste the full function code here if starting fresh ---
    if not os.path.exists(output_dir_csv):
        try:
            os.makedirs(output_dir_csv)
        except OSError as e:
            print(f"Error creating CSV dir '{output_dir_csv}': {e}")
            return
    data_to_export = {"parameter_value": param_values, "parameter_name": param_name_str}
    required_length = len(param_values)
    for outcome_key, outcome_values in results_dict.items():
        if (
            isinstance(outcome_values, (list, np.ndarray))
            and len(outcome_values) == required_length
        ):
            data_to_export[outcome_key] = np.array(outcome_values)
        else:
            print(f"Warn: Mismatch '{outcome_key}' sens export {filename_base}.")
            data_to_export[outcome_key] = np.full(required_length, np.nan)
    req_final_keys = ["Z_final", "W_final", "R_final"]
    if all(
        key in data_to_export and not np.all(np.isnan(data_to_export[key]))
        for key in req_final_keys
    ):
        Zf = data_to_export["Z_final"]
        Wf = data_to_export["W_final"]
        Rf = data_to_export["R_final"]
        Totf = Zf + Wf + Rf
        Totf_s = np.maximum(Totf, 1e-9)
        data_to_export["Z_final_perc_vs_Outcomes"] = (Zf / Totf_s) * 100
        data_to_export["W_final_perc_vs_Outcomes"] = (Wf / Totf_s) * 100
        data_to_export["R_final_perc_vs_Outcomes"] = (Rf / Totf_s) * 100
        data_to_export["Total_Outcomes_ZWR"] = Totf
    else:
        print(f"-> Skip final % calc {filename_base}")
        [
            data_to_export.update({k: np.full(required_length, np.nan)})
            for k in [
                "Z_final_perc_vs_Outcomes",
                "W_final_perc_vs_Outcomes",
                "R_final_perc_vs_Outcomes",
                "Total_Outcomes_ZWR",
            ]
        ]
    if (
        "Y_peak" in data_to_export
        and Cum_A_final_values is not None
        and isinstance(Cum_A_final_values, (list, np.ndarray))
        and len(Cum_A_final_values) == required_length
        and not np.all(np.isnan(data_to_export["Y_peak"]))
    ):
        Cum_A_final_array = np.array(Cum_A_final_values)
        valid_mask = ~np.isnan(data_to_export["Y_peak"]) & ~np.isnan(Cum_A_final_array)
        Cum_A_final_safe = np.maximum(Cum_A_final_array[valid_mask], 1e-9)
        y_peak_norm = np.full(required_length, np.nan)
        y_peak_norm[valid_mask] = (
            data_to_export["Y_peak"][valid_mask] / Cum_A_final_safe
        )
        data_to_export["Y_peak_per_Total_Admission"] = y_peak_norm
        data_to_export["Cumulative_Admissions_Final"] = Cum_A_final_array
    else:
        print(f"-> Skip Y_peak_norm calc {filename_base}")
        data_to_export["Y_peak_per_Total_Admission"] = np.full(required_length, np.nan)
    try:
        df_sensitivity = pd.DataFrame(data_to_export)
    except ValueError as e:
        print(f"ERR creating DF {filename_base}: {e}")
        return
    ordered_cols = (
        ["parameter_value", "parameter_name"]
        + sorted([k for k in results_dict.keys() if k in df_sensitivity.columns])
        + [
            "Z_final_perc_vs_Outcomes",
            "W_final_perc_vs_Outcomes",
            "R_final_perc_vs_Outcomes",
        ]
        + ["Total_Outcomes_ZWR", "Cumulative_Admissions_Final"]
        + ["Y_peak_per_Total_Admission"]
    )
    ordered_cols = [c for c in ordered_cols if c in df_sensitivity.columns] + sorted(
        [c for c in df_sensitivity.columns if c not in ordered_cols]
    )
    valid_ordered_cols = [col for col in ordered_cols if col in df_sensitivity.columns]
    df_sensitivity = df_sensitivity[valid_ordered_cols]
    filepath = os.path.join(output_dir_csv, f"{filename_base}_summary_full.csv")
    try:
        df_sensitivity.to_csv(filepath, index=False, float_format="%.5f")
    except Exception as e:
        print(f"ERROR: Failed sens export to '{filepath}': {e}")


# --- UPDATED Main Execution ---
if __name__ == "__main__":
    print(f"Starting simulations...")
    print(f" -> Plots saved in: '{OUTPUT_DIR_PLOTS}/'")
    print(f" -> CSV data saved in: '{OUTPUT_DIR_CSV}/'")

    df_patients = load_and_preprocess_data("patient_data.csv")
    if df_patients is None:
        exit("Exiting due to data loading error.")
    base_params = estimate_parameters_from_data(
        df_patients, T_OBSERVATION_DAYS, AVG_STAY_X_DAYS, AVG_STAY_Y_DAYS
    )
    if base_params is None:
        exit("Exiting due to parameter estimation error.")

    t_span = [0, T_SIM_END]
    t_eval = np.linspace(
        t_span[0], t_span[1], int(T_SIM_END * 10) + 1
    )  # Increased points
    y0 = [0, 0, 0, 0, 0]  # Start with empty system

    # --- UPDATED: Added C3 and C4 filenames ---
    fnames = {
        "A1": "A1_base_autonomous",
        "B1": "B1_variable_A",
        "B2": "B2_variable_theta",
        "B3": "B3_variable_A_theta",
        "C1": "C1_sensitivity_theta0",
        "C2_out": "C2_sensitivity_alpha_outcomes",
        "C2_peak": "C2_sensitivity_alpha_Ypeak",
        "C3_out": "C3_sensitivity_gamma_outcomes",  # New
        "C3_peak": "C3_sensitivity_gamma_Ypeak",  # New
        "C4_out": "C4_sensitivity_epsilon_outcomes",  # New
        "C4_peak": "C4_sensitivity_epsilon_Ypeak",  # New
    }

    print("\n--- Running Core Simulations & Exporting Comprehensive Data ---")
    # --- This loop remains UNCHANGED ---
    scenarios = {
        "A1": {"use_A": False, "use_T": False},
        "B1": {"use_A": True, "use_T": False},
        "B2": {"use_A": False, "use_T": True},
        "B3": {"use_A": True, "use_T": True},
    }
    final_cum_A_scenarios = {}
    for sc_key, sc_config in scenarios.items():
        print(f"Running Simulation {fnames[sc_key]}...")
        params = copy.deepcopy(base_params)
        A_function_to_use = gaussian_peak if sc_config["use_A"] else None
        theta_t_enabled = (
            sc_config["use_T"]
            and params.get("Delta_theta", 0) != 0
            and params.get("lambda_decay", 0) > 0
        )
        if not theta_t_enabled:
            params["Delta_theta"] = 0.0
            params["lambda_decay"] = 0.0
        if not sc_config["use_A"]:
            params["A_peak_scale"] = 0.0  # Ensure peak scale is 0 if not used
        sol = run_simulation(
            t_span,
            t_eval,
            y0,
            params,
            A_func=A_function_to_use,
            use_theta_t=theta_t_enabled,
        )
        final_cum_A = export_simulation_to_csv(sol, fnames[sc_key], OUTPUT_DIR_CSV)
        final_cum_A_scenarios[sc_key] = final_cum_A
        params_for_plot = sol.params_used
        proportions_for_plot = None
        if len(sol.y[0]) > 0:
            final_Z = sol.y[2][-1]
            final_W = sol.y[3][-1]
            final_R = sol.y[4][-1]
            final_total_outcomes = final_Z + final_W + final_R
            if final_total_outcomes > 1e-9:
                proportions_for_plot = {
                    "Z%": (final_Z / final_total_outcomes) * 100,
                    "W%": (final_W / final_total_outcomes) * 100,
                    "R%": (final_R / final_total_outcomes) * 100,
                }
            else:
                proportions_for_plot = {}
        else:
            proportions_for_plot = {}
        title = f"Sim {fnames[sc_key].replace('_',' ')}"
        plot_time_series(
            sol,
            title,
            fnames[sc_key],
            OUTPUT_DIR_PLOTS,
            params_for_plot,
            proportions_for_plot,
        )
    # --- End Core Simulations Loop ---

    print("\n--- Running Sensitivity Analyses & Exporting Summaries ---")

    # --- C1: Sensitivity to theta_0 --- (UNCHANGED LOGIC) ---
    print(f"Running Sensitivity {fnames['C1']}...")
    theta_0_base = base_params["theta_0"]
    theta_0_range = np.maximum(theta_0_base * SENSITIVITY_RANGE, 1e-9)
    results_C1 = {
        "Z_final": [],
        "W_final": [],
        "R_final": [],
        "Y_peak": [],
    }  # Also collect Y_peak for consistency
    cum_A_finals_C1 = []
    param_name_C1 = "theta_0"
    for theta_val in theta_0_range:
        params_C1 = copy.deepcopy(base_params)
        params_C1["theta_0"] = theta_val
        params_C1["Delta_theta"] = 0.0
        sol_C1_run = run_simulation(
            t_span, t_eval, y0, params_C1, A_func=None, use_theta_t=False
        )
        results_C1["Z_final"].append(
            sol_C1_run.y[2][-1] if len(sol_C1_run.t) > 0 else np.nan
        )
        results_C1["W_final"].append(
            sol_C1_run.y[3][-1] if len(sol_C1_run.t) > 0 else np.nan
        )
        results_C1["R_final"].append(
            sol_C1_run.y[4][-1] if len(sol_C1_run.t) > 0 else np.nan
        )
        results_C1["Y_peak"].append(
            np.max(sol_C1_run.y[1]) if sol_C1_run.y[1].size > 0 else np.nan
        )  # Collect Y_peak
        t_sol = sol_C1_run.t
        A_avg_C1 = sol_C1_run.params_used.get("A_avg", 0)
        A_array_C1 = np.full_like(t_sol, A_avg_C1)
        if len(t_sol) > 1:
            dt_C1 = np.diff(t_sol, prepend=t_sol[0])
            dt_C1[0] = t_sol[1] - t_sol[0] if t_sol[0] == 0 else dt_C1[0]
            Cum_A_final_run = np.sum(A_array_C1 * dt_C1)
        elif len(t_sol) == 1:
            Cum_A_final_run = A_array_C1[0] * t_sol[0]
        else:
            Cum_A_final_run = 0
        cum_A_finals_C1.append(Cum_A_final_run)
    export_sensitivity_results_to_csv(
        results_C1,
        param_name_C1,
        theta_0_range,
        fnames["C1"],
        OUTPUT_DIR_CSV,
        Cum_A_final_values=cum_A_finals_C1,
    )
    plot_sensitivity(
        results_C1,
        "$\\theta_0$",
        theta_0_range,
        f"Sim {fnames['C1']}: Sensitivity to $\\theta_0$ (Outcomes)",
        fnames["C1"],
        OUTPUT_DIR_PLOTS,
    )
    # plot_sensitivity_single(results_C1['Y_peak'], "$\\theta_0$", theta_0_range, f"Sim C1: Sensitivity of Peak NIV Load to $\\theta_0$", "Peak NIV Patients (Y)", f"{fnames['C1']}_Ypeak", OUTPUT_DIR_PLOTS, theta_0_base) # Optional Y_peak plot

    # --- C2: Sensitivity to alpha --- (UNCHANGED LOGIC) ---
    print(f"Running Sensitivity {fnames['C2_out']}...")
    alpha_base = base_params["alpha"]
    alpha_range = np.maximum(alpha_base * SENSITIVITY_RANGE, 1e-9)
    results_C2 = {"Z_final": [], "W_final": [], "R_final": [], "Y_peak": []}
    cum_A_finals_C2 = []
    param_name_C2 = "alpha"
    for alpha_val in alpha_range:
        params_C2 = copy.deepcopy(base_params)
        params_C2["alpha"] = alpha_val
        params_C2["Delta_theta"] = 0.0
        sol_C2_run = run_simulation(
            t_span, t_eval, y0, params_C2, A_func=None, use_theta_t=False
        )
        results_C2["Z_final"].append(
            sol_C2_run.y[2][-1] if len(sol_C2_run.t) > 0 else np.nan
        )
        results_C2["W_final"].append(
            sol_C2_run.y[3][-1] if len(sol_C2_run.t) > 0 else np.nan
        )
        results_C2["R_final"].append(
            sol_C2_run.y[4][-1] if len(sol_C2_run.t) > 0 else np.nan
        )
        results_C2["Y_peak"].append(
            np.max(sol_C2_run.y[1]) if sol_C2_run.y[1].size > 0 else np.nan
        )
        t_sol = sol_C2_run.t
        A_avg_C2 = sol_C2_run.params_used.get("A_avg", 0)
        A_array_C2 = np.full_like(t_sol, A_avg_C2)
        if len(t_sol) > 1:
            dt_C2 = np.diff(t_sol, prepend=t_sol[0])
            dt_C2[0] = t_sol[1] - t_sol[0] if t_sol[0] == 0 else dt_C2[0]
            Cum_A_final_run = np.sum(A_array_C2 * dt_C2)
        elif len(t_sol) == 1:
            Cum_A_final_run = A_array_C2[0] * t_sol[0]
        else:
            Cum_A_final_run = 0
        cum_A_finals_C2.append(Cum_A_final_run)
    export_sensitivity_results_to_csv(
        results_C2,
        param_name_C2,
        alpha_range,
        fnames["C2_out"],
        OUTPUT_DIR_CSV,
        Cum_A_final_values=cum_A_finals_C2,
    )
    plot_sensitivity(
        results_C2,
        "$\\alpha$",
        alpha_range,
        f"Sim {fnames['C2_out']}: Sensitivity to $\\alpha$ (Outcomes)",
        fnames["C2_out"],
        OUTPUT_DIR_PLOTS,
    )
    plot_sensitivity_single(
        results_C2["Y_peak"],
        "$\\alpha$",
        alpha_range,
        f"Sim {fnames['C2_peak']}: Sensitivity of Peak NIV Load to $\\alpha$",
        "Peak NIV Patients (Y)",
        fnames["C2_peak"],
        OUTPUT_DIR_PLOTS,
        alpha_base,
    )

    # --- C3: Sensitivity to gamma --- (NEW BLOCK) ---
    print(f"Running Sensitivity {fnames['C3_out']}...")
    gamma_base = base_params["gamma"]
    gamma_range = np.maximum(gamma_base * SENSITIVITY_RANGE, 1e-9)
    results_C3 = {"Z_final": [], "W_final": [], "R_final": [], "Y_peak": []}
    cum_A_finals_C3 = []
    param_name_C3 = "gamma"
    for gamma_val in gamma_range:
        params_C3 = copy.deepcopy(base_params)
        params_C3["gamma"] = gamma_val
        params_C3["Delta_theta"] = 0.0
        sol_C3_run = run_simulation(
            t_span, t_eval, y0, params_C3, A_func=None, use_theta_t=False
        )
        results_C3["Z_final"].append(
            sol_C3_run.y[2][-1] if len(sol_C3_run.t) > 0 else np.nan
        )
        results_C3["W_final"].append(
            sol_C3_run.y[3][-1] if len(sol_C3_run.t) > 0 else np.nan
        )
        results_C3["R_final"].append(
            sol_C3_run.y[4][-1] if len(sol_C3_run.t) > 0 else np.nan
        )
        results_C3["Y_peak"].append(
            np.max(sol_C3_run.y[1]) if sol_C3_run.y[1].size > 0 else np.nan
        )
        t_sol = sol_C3_run.t
        A_avg_C3 = sol_C3_run.params_used.get("A_avg", 0)
        A_array_C3 = np.full_like(t_sol, A_avg_C3)
        if len(t_sol) > 1:
            dt_C3 = np.diff(t_sol, prepend=t_sol[0])
            dt_C3[0] = t_sol[1] - t_sol[0] if t_sol[0] == 0 else dt_C3[0]
            Cum_A_final_run = np.sum(A_array_C3 * dt_C3)
        elif len(t_sol) == 1:
            Cum_A_final_run = A_array_C3[0] * t_sol[0]
        else:
            Cum_A_final_run = 0
        cum_A_finals_C3.append(Cum_A_final_run)
    export_sensitivity_results_to_csv(
        results_C3,
        param_name_C3,
        gamma_range,
        fnames["C3_out"],
        OUTPUT_DIR_CSV,
        Cum_A_final_values=cum_A_finals_C3,
    )
    plot_sensitivity(
        results_C3,
        "$\\gamma$",
        gamma_range,
        f"Sim {fnames['C3_out']}: Sensitivity to $\\gamma$ (Outcomes)",
        fnames["C3_out"],
        OUTPUT_DIR_PLOTS,
    )
    plot_sensitivity_single(
        results_C3["Y_peak"],
        "$\\gamma$",
        gamma_range,
        f"Sim {fnames['C3_peak']}: Sensitivity of Peak NIV Load to $\\gamma$",
        "Peak NIV Patients (Y)",
        fnames["C3_peak"],
        OUTPUT_DIR_PLOTS,
        gamma_base,
    )

    # --- C4: Sensitivity to epsilon --- (NEW BLOCK) ---
    print(f"Running Sensitivity {fnames['C4_out']}...")
    epsilon_base = base_params["epsilon"]
    epsilon_range = np.maximum(epsilon_base * SENSITIVITY_RANGE, 1e-9)
    results_C4 = {"Z_final": [], "W_final": [], "R_final": [], "Y_peak": []}
    cum_A_finals_C4 = []
    param_name_C4 = "epsilon"
    for epsilon_val in epsilon_range:
        params_C4 = copy.deepcopy(base_params)
        params_C4["epsilon"] = epsilon_val
        params_C4["Delta_theta"] = 0.0
        sol_C4_run = run_simulation(
            t_span, t_eval, y0, params_C4, A_func=None, use_theta_t=False
        )
        results_C4["Z_final"].append(
            sol_C4_run.y[2][-1] if len(sol_C4_run.t) > 0 else np.nan
        )
        results_C4["W_final"].append(
            sol_C4_run.y[3][-1] if len(sol_C4_run.t) > 0 else np.nan
        )
        results_C4["R_final"].append(
            sol_C4_run.y[4][-1] if len(sol_C4_run.t) > 0 else np.nan
        )
        results_C4["Y_peak"].append(
            np.max(sol_C4_run.y[1]) if sol_C4_run.y[1].size > 0 else np.nan
        )
        t_sol = sol_C4_run.t
        A_avg_C4 = sol_C4_run.params_used.get("A_avg", 0)
        A_array_C4 = np.full_like(t_sol, A_avg_C4)
        if len(t_sol) > 1:
            dt_C4 = np.diff(t_sol, prepend=t_sol[0])
            dt_C4[0] = t_sol[1] - t_sol[0] if t_sol[0] == 0 else dt_C4[0]
            Cum_A_final_run = np.sum(A_array_C4 * dt_C4)
        elif len(t_sol) == 1:
            Cum_A_final_run = A_array_C4[0] * t_sol[0]
        else:
            Cum_A_final_run = 0
        cum_A_finals_C4.append(Cum_A_final_run)
    export_sensitivity_results_to_csv(
        results_C4,
        param_name_C4,
        epsilon_range,
        fnames["C4_out"],
        OUTPUT_DIR_CSV,
        Cum_A_final_values=cum_A_finals_C4,
    )
    plot_sensitivity(
        results_C4,
        "$\\epsilon$",
        epsilon_range,
        f"Sim {fnames['C4_out']}: Sensitivity to $\\epsilon$ (Outcomes)",
        fnames["C4_out"],
        OUTPUT_DIR_PLOTS,
    )
    plot_sensitivity_single(
        results_C4["Y_peak"],
        "$\\epsilon$",
        epsilon_range,
        f"Sim {fnames['C4_peak']}: Sensitivity of Peak NIV Load to $\\epsilon$",
        "Peak NIV Patients (Y)",
        fnames["C4_peak"],
        OUTPUT_DIR_PLOTS,
        epsilon_base,
    )

    print(f"\nAll simulations complete.")
    print(f" -> Plots saved in: '{OUTPUT_DIR_PLOTS}/'")
    print(f" -> Comprehensive CSV data saved in: '{OUTPUT_DIR_CSV}/'")

# %%
# --- END OF FILE ircu_simulator_export_v3.py ---
