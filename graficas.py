import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for generating plots without a display
import matplotlib.pyplot as plt
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

# ========================================================
# 1. CONFIGURACIÓN DEL USUARIO
# ========================================================
# MODIFICADO: Detección automática o configuración manual
# Para mantener la compatibilidad con ajuste_general.py, usamos MODEL_NAME
MODEL_NAME = "PEDE"   # <--- CAMBIA ESTO: "LCDM", "CPL", "PEDE"

# Rutas Base (DINÁMICA Y RELATIVA)
BASE_PATH = Path.cwd() / "Ajuste_Modelos"

# Carpeta de salida global para gráficas del modelo actual
OUTDIR = BASE_PATH / MODEL_NAME / "GRAFICAS_GLOBALES"
OUTDIR.mkdir(parents=True, exist_ok=True)

# --- Configuración para Comparativa Multi-Modelo ---
# Carpeta donde se guardarán los plots comparativos (LCDM vs CPL vs PEDE)
OUTDIR_COMPARISON = BASE_PATH / "COMPARATIVA_MODELOS"
OUTDIR_COMPARISON.mkdir(parents=True, exist_ok=True)

# Lista de modelos a buscar y comparar
MODELS_TO_COMPARE = [
    {"name": "LCDM", "color": "black",   "label": r"$\Lambda$CDM"},
    {"name": "CPL",  "color": "#1f77b4", "label": "CPL"},
    {"name": "PEDE", "color": "#d62728", "label": "PEDE"}
]

# Rutas a los datos observacionales (RELATIVAS)
PATH_HZ_DATA  = "HzTable_MM_BC32.txt"
CATALOGO_FRBS = "localized_FRBs(1).txt"

print(f">>> Generando gráficas para: {MODEL_NAME}")
print(f">>> Guardando en: {OUTDIR}")

# ------------------------------
# Estilo global "paper" (ALTA DEFINICIÓN)
# ------------------------------
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "font.size": 12,
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "axes.linewidth": 1.2,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.6",
    "figure.constrained_layout.use": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "path.simplify": False,
    "axes.unicode_minus": False
})

LEVELS_2D = [0.68, 0.95]

COLOR_CASES = {
    "Hz":        "#1f77b4",  # azul
    "FRBs":      "#2ca02c",  # verde
    "FRBs+Hz":   "#d62728",  # rojo (Joint)
}

def color_for_title(title: str) -> str:
    for k, v in COLOR_CASES.items():
        if k.lower() in str(title).lower():
            return v
    return "black"

# ------------------------------
# Generación dinámica de rutas
# ------------------------------
def get_path(tipo, archivo):
    return BASE_PATH / MODEL_NAME / tipo / archivo

# Definir rutas a las cadenas y estadísticas
PATH_FRBs     = get_path("FRBs", f"chain_{MODEL_NAME}_FRBs.csv")
PATH_Hz       = get_path("Hz",   f"chain_{MODEL_NAME}_Hz.csv")
PATH_FRBs_Hz  = get_path("FRBs_Hz", f"chain_{MODEL_NAME}_FRBs_Hz.csv")

PATH_QJ_FRBs     = get_path("FRBs", f"qj_stats_{MODEL_NAME}.csv")
PATH_QJ_Hz       = get_path("Hz",   f"qj_stats_{MODEL_NAME}.csv")
PATH_QJ_FRBs_Hz  = get_path("FRBs_Hz", f"qj_stats_{MODEL_NAME}.csv")


# ------------------------------
# Utilidades comunes
# ------------------------------
def add_outside_legend(fig, labels, colors, where=(0.66, 0.76, 0.25, 0.20), use_lines=True, lw=2.2):
    if use_lines:
        handles = [Line2D([0],[0], color=c, lw=lw, label=lab) for lab,c in zip(labels, colors)]
    else:
        handles = [Patch(facecolor='none', edgecolor=c, linewidth=lw, label=lab) for lab,c in zip(labels, colors)]
    ax_leg = fig.add_axes(where)
    ax_leg.axis('off')
    ax_leg.legend(handles=handles, loc="center", frameon=True)

def remove_local_legends(fig):
    for ax in fig.get_axes():
        lg = ax.get_legend()
        if lg is not None:
            lg.remove()

def load_chain_df(path):
    return pd.read_csv(path)

def _normalize_colnames(df):
    colmap = {}
    for c in df.columns:
        k = c.strip().lower().replace(" ", "")
        if k == "h": colmap[c] = "h"
        elif ("ωb" in k) or ("omegab" in k) or (k == "ob"): colmap[c] = "Ωb"
        elif ("ωm" in k) or ("omegam" in k) or (k == "om"): colmap[c] = "Ωm"
        elif k == "w0": colmap[c] = "w0"
        elif k == "wa": colmap[c] = "wa"
        elif ("dm_host" in k) or ("dmhost" in k): colmap[c] = "DM_host"
        elif ("dm_halo" in k) or ("dmhalo" in k): colmap[c] = "DM_halo"
        elif k == "h0": colmap[c] = "H0"
    return df.rename(columns=colmap)

def chain_cosmo_params(df):
    """Extrae parámetros cosmológicos disponibles."""
    df = _normalize_colnames(df)
    wanted = ["h", "Ωb", "Ωm", "w0", "wa"]
    found = [c for c in wanted if c in df.columns]
    return df[found].to_numpy(float), found

def get_latex_labels(cols):
    lab_map = {
        "h": r"$h$", "Ωb": r"$\Omega_b$", "Ωm": r"$\Omega_m$",
        "w0": r"$w_0$", "wa": r"$w_a$"
    }
    return [lab_map.get(c, c) for c in cols]

def padded_ranges(arrays, qlo=0.1, qhi=99.9, pad_frac=0.15):
    data = np.vstack(arrays)
    mins = np.percentile(data, qlo, axis=0)
    maxs = np.percentile(data, qhi, axis=0)
    diff = maxs - mins
    diff[diff == 0] = 1.0
    return [(mins[i] - diff[i]*pad_frac, maxs[i] + diff[i]*pad_frac) for i in range(data.shape[1])]

def get_contour_colors_simple(color):
    color_center = mpl.colors.to_rgba(color, alpha=0.5)
    color_periphery = mpl.colors.to_rgba(color, alpha=0.2)
    color_fondo = (0.0, 0.0, 0.0, 0.0)
    return [color_fondo, color_periphery, color_center]

def _overlay_kde_on_diagonal(fig, data, color, lw=2.2, smooth_pad=0.04):
    K = data.shape[1]
    axes = np.array(fig.axes).reshape(K, K)
    for i in range(K):
        ax = axes[i, i]
        x = data[:, i]
        if np.std(x) < 1e-6: continue
        xmin, xmax = np.quantile(x, 0.001), np.quantile(x, 0.999)
        dx = (xmax - xmin) * smooth_pad
        grid = np.linspace(xmin - dx, xmax + dx, 600)
        kde = gaussian_kde(x)
        y = kde(grid)
        ax.plot(grid, y, color=color, lw=lw)

# ------------------------------
# Corner plots
# ------------------------------
def corner_single(data, labels, title, outfile_pdf):
    ranges = padded_ranges([data])
    col = color_for_title(title)
    contour_colors = get_contour_colors_simple(col)

    fig = corner.corner(
        data,
        labels=labels,
        range=ranges,
        bins=40,
        smooth=1.5,
        max_n_ticks=4,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        plot_contours=True,
        levels=LEVELS_2D,
        color=col,
        show_titles=True,
        title_kwargs={"fontsize": 14},
        title_fmt=".3f",
        hist_kwargs=dict(density=True, alpha=0.0),
        contour_kwargs=dict(colors=[col], linewidths=0.5),
        contourf_kwargs=dict(colors=contour_colors, extend="neither", antialiased=True)
    )

    _overlay_kde_on_diagonal(fig, data, col, lw=2.2)
    fig.patch.set_facecolor("white")
    for ax in fig.get_axes(): ax.set_facecolor("white")
    fig.suptitle(f"Posteriors: {title} ({MODEL_NAME})", fontsize=16, y=1.02)
    plt.savefig(outfile_pdf, bbox_inches="tight", pad_inches=0.1, facecolor='white')
    plt.close(fig)

def corner_triple(data_frbs, data_hz, data_both, labels, outfile_pdf):
    ranges = padded_ranges([data_frbs, data_hz, data_both], qlo=0.0, qhi=100.0, pad_frac=0.2)
    col_hz, col_frbs, col_joint = COLOR_CASES["Hz"], COLOR_CASES["FRBs"], COLOR_CASES["FRBs+Hz"]
    colors_hz = get_contour_colors_simple(col_hz)
    colors_frbs = get_contour_colors_simple(col_frbs)
    colors_joint = get_contour_colors_simple(col_joint)

    fig = corner.corner(
        data_hz, labels=labels, range=ranges, bins=40, smooth=2.0, max_n_ticks=4,
        plot_datapoints=False, plot_density=False, fill_contours=True, plot_contours=True,
        levels=LEVELS_2D, color=col_hz, hist_kwargs=dict(density=True, alpha=0.0),
        contour_kwargs=dict(colors=[col_hz], linewidths=0.5),
        contourf_kwargs=dict(colors=colors_hz, extend="neither", antialiased=True)
    )
    corner.corner(
        data_frbs, range=ranges, bins=40, smooth=2.0, max_n_ticks=4,
        plot_datapoints=False, plot_density=False, fill_contours=True, plot_contours=True,
        levels=LEVELS_2D, color=col_frbs, hist_kwargs=dict(density=True, alpha=0.0),
        fig=fig, contour_kwargs=dict(colors=[col_frbs], linewidths=0.5),
        contourf_kwargs=dict(colors=colors_frbs, extend="neither", antialiased=True)
    )
    corner.corner(
        data_both, range=ranges, bins=40, smooth=2.0, max_n_ticks=4,
        plot_datapoints=False, plot_density=False, fill_contours=True, plot_contours=True,
        levels=LEVELS_2D, color=col_joint, hist_kwargs=dict(density=True, alpha=0.0),
        fig=fig, contour_kwargs=dict(colors=[col_joint], linewidths=0.5),
        contourf_kwargs=dict(colors=colors_joint, extend="neither", antialiased=True)
    )

    _overlay_kde_on_diagonal(fig, data_hz, col_hz, lw=2.0)
    _overlay_kde_on_diagonal(fig, data_frbs, col_frbs, lw=2.0)
    _overlay_kde_on_diagonal(fig, data_both, col_joint, lw=2.0)

    ndim = data_hz.shape[1]
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        ax = axes[i, i]
        ax.set_title(labels[i], fontsize=16, pad=10)
        max_y_found = 0.0
        lines = ax.get_lines()
        for line in lines:
            y_data = line.get_ydata()
            if len(y_data) > 0:
                local_max = np.max(y_data)
                if local_max > max_y_found: max_y_found = local_max
        if max_y_found > 0: ax.set_ylim(0, max_y_found * 1.1)

    fig.patch.set_facecolor("white")
    for ax in fig.get_axes(): ax.set_facecolor("white")
    remove_local_legends(fig)
    add_outside_legend(fig, labels=["Hz", "FRBs", "FRBs+Hz"], colors=[col_hz, col_frbs, col_joint],
                       where=(0.66, 0.76, 0.25, 0.20), use_lines=False, lw=2.0)
    plt.savefig(outfile_pdf, bbox_inches="tight", pad_inches=0.1, facecolor='white')
    plt.close(fig)

def plot_convergence(df, label, outfile_pdf):
    df = _normalize_colnames(df)
    wanted = ["h", "Ωb", "Ωm", "w0", "wa"]
    cols = [c for c in wanted if c in df.columns]
    vals = df[cols].to_numpy(float)
    labels = get_latex_labels(cols)
    n_vars = len(cols)
    fig, axes = plt.subplots(n_vars, 1, figsize=(8, 2*n_vars), sharex=True)
    if n_vars == 1: axes = [axes]
    x = np.arange(vals.shape[0])
    for i in range(n_vars):
        y = vals[:, i]
        q16, q50, q84 = np.percentile(y, [16, 50, 84])
        ax = axes[i]
        ax.plot(x, y, color='black', alpha=0.35, linewidth=0.6)
        ax.axhline(q50, color='red', linestyle='--', linewidth=1.2)
        ax.axhline(q16, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.axhline(q84, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.set_ylabel(labels[i])
        leg_text = f"{labels[i]} = {q50:.3f}"
        ax.legend([leg_text], loc='upper right', fontsize=9, frameon=True)
        ax.tick_params(axis='both', which='major', direction='in')
    axes[-1].set_xlabel("Índice de muestra (paso)")
    fig.suptitle(f"Convergencia (traza) – {label}", y=0.99, fontsize=12)
    plt.tight_layout()
    plt.savefig(outfile_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ------------------------------
# Cosmología y DM_IGM
# ------------------------------
f_IGM, f_e = 0.84, 0.88
G, m_p, c_ms = 6.67430e-11, 1.67262e-27, 299792458.0
Mpc_m, pc_m = 3.085677581491367e22, 3.085677581491367e16

def E_Theoretical(z, params_dict):
    """Calcula E(z) soportando LCDM, CPL y PEDE."""
    z = np.asarray(z, dtype=float)
    Om = params_dict.get("Ωm", 0.3)
    E2 = Om*(1.0+z)**3 + (1.0-Om)
    if MODEL_NAME == "CPL":
        w0 = params_dict.get("w0", -1.0)
        wa = params_dict.get("wa", 0.0)
        de_term = (1-Om) * (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z))
        E2 = Om*(1+z)**3 + de_term
    elif MODEL_NAME == "PEDE":
        arg_tanh = np.log10(1+z)
        de_term = (1-Om) * (1 - np.tanh(arg_tanh))
        E2 = Om*(1+z)**3 + de_term
    return np.sqrt(np.abs(E2))

def H_of_z(params_dict, z):
    h = params_dict.get("h", 0.7)
    return 100.0 * float(h) * E_Theoretical(z, params_dict)

def DM_IGM_model_params(params_dict, z_eval):
    h  = params_dict.get("h", 0.7)
    Ob = params_dict.get("Ωb", 0.05)
    
    z_max = np.max(z_eval) if np.ndim(z_eval)>0 else z_eval
    z_grid = np.linspace(0.0, float(z_max)*1.05, 500)
    E_vals = E_Theoretical(z_grid, params_dict)
    integrand = (1.0 + z_grid) / E_vals
    I_cum = cumulative_trapezoid(integrand, z_grid, initial=0)
    I_target = np.interp(z_eval, z_grid, I_cum)
    
    H0_SI  = (100.0 * h * 1000.0) / Mpc_m
    rho_c0 = 3.0 * H0_SI**2 / (8.0 * np.pi * G)
    n_e0_cm3 = (Ob * rho_c0 / m_p) * f_IGM * 1e-6
    c_over_H0_pc = (c_ms / H0_SI) / pc_m
    return c_over_H0_pc * n_e0_cm3 * I_target * f_e

# ------------------------------
# Carga de Datos
# ------------------------------
def load_cc_data():
    if not Path(CATALOGO_FRBS).exists(): return None
    try:
        df = pd.read_csv(CATALOGO_FRBS, sep=r'\s+', header=1,
            names=['Names','Redshift','DM_obs','RA','Dec','DM_MW_NE2001','DM_MW_YMW16','Type'])
        for c in ['Redshift','DM_obs','DM_MW_NE2001']: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        mask = ((df['DM_obs'] - df['DM_MW_NE2001']) > 80.0)
        valid = df[mask].copy()
        z = valid['Redshift'].values
        
        med_host, med_halo = 100.0, 65.0
        if Path(PATH_FRBs_Hz).exists():
            d = pd.read_csv(PATH_FRBs_Hz)
            d = _normalize_colnames(d)
            if "DM_host" in d: med_host = d["DM_host"].median()
            if "DM_halo" in d: med_halo = d["DM_halo"].median()
            
        y = valid['DM_obs'].values - valid['DM_MW_NE2001'].values - med_halo - med_host/(1+z)
        y_err = np.sqrt(30**2 + 30**2 + (80/(1+z))**2)
        return pd.DataFrame({"z": z, "DM": y, "DM_err": y_err})
    except Exception as e:
        print(f"Error cargando FRBs: {e}")
        return None

def load_hz_points():
    if not Path(PATH_HZ_DATA).exists(): return None
    try:
        return pd.read_csv(PATH_HZ_DATA, sep=r'\s+', comment="#", names=["z", "H", "eH"])
    except Exception as e:
        print(f"Error cargando Hz: {e}")
        return None

# ------------------------------
# Gráficas Overlay
# ------------------------------
def plot_overlay_Hz(df_hz_pts, paths_chains):
    print("   -> Generando Overlay H(z)...")
    fig, ax = plt.subplots(figsize=(8, 6))
    if df_hz_pts is not None:
        ax.errorbar(df_hz_pts.z, df_hz_pts.H, yerr=df_hz_pts.eH, fmt='s', ms=5, color='black', alpha=0.7, elinewidth=1.5, capsize=3, label="Datos CC", zorder=5)

    labels = ["FRBs", "Hz", "Joint"]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    z_grid = np.linspace(0, 2.5, 200)

    for lbl, path, col in zip(labels, paths_chains, colors):
        if not path.exists(): continue
        chain = _normalize_colnames(pd.read_csv(path))
        sample = chain.sample(n=min(300, len(chain)), random_state=42)
        h_lines = []
        for _, row in sample.iterrows():
            h_lines.append(H_of_z(row.to_dict(), z_grid))
        h_med = np.median(h_lines, axis=0)
        h_lo = np.percentile(h_lines, 16, axis=0)
        h_hi = np.percentile(h_lines, 84, axis=0)
        ax.plot(z_grid, h_med, color=col, lw=2.5, label=f"{lbl}")
        ax.fill_between(z_grid, h_lo, h_hi, color=col, alpha=0.25)

    ax.set_xlabel("Redshift z", fontsize=14)
    ax.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=14)
    ax.set_xlim(0, 2.5)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.set_title(f"Historia de Expansión $H(z)$ - {MODEL_NAME}", fontsize=14)
    ax.grid(True, alpha=0.2)
    plt.savefig(OUTDIR / "overlay_Hz.pdf", bbox_inches="tight")
    plt.close()

def plot_overlay_DM(df_frb, paths_chains):
    print("   -> Generando Overlay DM_IGM...")
    fig, ax = plt.subplots(figsize=(8, 6))
    if df_frb is not None:
        ax.errorbar(df_frb.z, df_frb.DM, yerr=df_frb.DM_err, fmt='o', ms=4, color='darkviolet', alpha=0.7, elinewidth=1.0, capsize=2, label="Datos FRBs", zorder=1)

    labels = ["FRBs", "Hz", "Joint"]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    z_grid = np.linspace(0, 2.5, 200)

    for lbl, path, col in zip(labels, paths_chains, colors):
        if not path.exists(): continue
        chain = _normalize_colnames(pd.read_csv(path))
        sample = chain.sample(n=min(300, len(chain)), random_state=42)
        dm_lines = []
        for _, row in sample.iterrows():
            dm_lines.append(DM_IGM_model_params(row.to_dict(), z_grid))
        dm_med = np.median(dm_lines, axis=0)
        dm_lo = np.percentile(dm_lines, 16, axis=0)
        dm_hi = np.percentile(dm_lines, 84, axis=0)
        ax.plot(z_grid, dm_med, color=col, lw=2.5, label=f"{lbl}", zorder=10)
        ax.fill_between(z_grid, dm_lo, dm_hi, color=col, alpha=0.25, zorder=9)

    ax.set_xlabel("Redshift z", fontsize=14)
    ax.set_ylabel(r"$DM_{IGM}$ [pc cm$^{-3}$]", fontsize=14)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 3000)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.set_title(f"Relación de Dispersión $DM(z)$ - {MODEL_NAME}", fontsize=14)
    ax.grid(True, alpha=0.2)
    plt.savefig(OUTDIR / "overlay_DM.pdf", bbox_inches="tight")
    plt.close()

# ------------------------------
# FÍSICA: CÁLCULO DE q(z) y j(z)
# ------------------------------
def calculate_kinematics(params_dict, z_arr):
    Om = params_dict.get("Ωm", params_dict.get("Om", 0.3))
    if MODEL_NAME == "CPL":
        w0 = params_dict.get("w0", -1.0)
        wa = params_dict.get("wa", 0.0)
        de_term = (1-Om) * (1+z_arr)**(3*(1+w0+wa)) * np.exp(-3*wa*z_arr/(1+z_arr))
        E = np.sqrt(Om*(1+z_arr)**3 + de_term)
    elif MODEL_NAME == "PEDE":
        arg = np.log10(1+z_arr)
        de_term = (1-Om) * (1 - np.tanh(arg))
        E = np.sqrt(Om*(1+z_arr)**3 + de_term)
    else:
        E = np.sqrt(Om*(1+z_arr)**3 + (1-Om))
    dE_dz = np.gradient(E, z_arr)
    q = ((1 + z_arr) / E) * dE_dz - 1
    dq_dz = np.gradient(q, z_arr)
    j = q * (2 * q + 1) + (1 + z_arr) * dq_dz
    return q, j

def plot_qj_extended(paths_chains, paths_save_stats):
    print("   -> Generando cinemática extendida...")
    z_full = np.linspace(0, 2.5, 206)
    fig_q, ax_q = plt.subplots(figsize=(8, 5))
    fig_j, ax_j = plt.subplots(figsize=(8, 5))
    labels = ["FRBs", "Hz", "Joint"]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]

    for lbl, path, col, save_path in zip(labels, paths_chains, colors, paths_save_stats):
        if not path.exists(): continue
        chain = _normalize_colnames(pd.read_csv(path))
        sample = chain.sample(n=min(300, len(chain)), random_state=42)
        q_list, j_list = [], []
        for _, row in sample.iterrows():
            q, j = calculate_kinematics(row.to_dict(), z_full)
            q_list.append(q)
            j_list.append(j)
        q_med = np.median(q_list, 0)
        q_lo, q_hi = np.percentile(q_list, 16, 0), np.percentile(q_list, 84, 0)
        j_med = np.median(j_list, 0)
        j_lo, j_hi = np.percentile(j_list, 16, 0), np.percentile(j_list, 84, 0)

        sl = slice(3, -3)
        z_c = z_full[sl]
        df_new = pd.DataFrame({
            "z": z_c, "q_med": q_med[sl], "q_lo": (q_med - q_lo)[sl], "q_hi": (q_hi - q_med)[sl],
            "j_med": j_med[sl], "j_lo": (j_med - j_lo)[sl], "j_hi": (j_hi - j_med)[sl]
        })
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(save_path, index=False)

        ax_q.plot(z_c, q_med[sl], label=lbl, color=col, lw=2)
        ax_q.fill_between(z_c, q_lo[sl], q_hi[sl], color=col, alpha=0.2)
        ax_j.plot(z_c, j_med[sl], label=lbl, color=col, lw=2)
        ax_j.fill_between(z_c, j_lo[sl], j_hi[sl], color=col, alpha=0.2)

    ax_q.axhline(0, ls=":", c="k"); ax_q.set_xlabel("z"); ax_q.set_ylabel("q(z)")
    ax_q.set_xlim(0, 2.5); ax_q.legend(loc="upper right")
    ax_q.set_title(f"Desaceleración {MODEL_NAME}")
    fig_q.savefig(OUTDIR / "q_z_evolution.pdf", bbox_inches="tight")
    plt.close(fig_q)

    ax_j.axhline(1, ls=":", c="k"); ax_j.set_xlabel("z"); ax_j.set_ylabel("j(z)")
    ax_j.set_xlim(0, 2.5); ax_j.legend(loc="best")
    ax_j.set_ylim(-5, 5)
    ax_j.set_title(f"Jerk {MODEL_NAME} (Zoom)")
    fig_j.savefig(OUTDIR / "j_z_evolution_zoom.pdf", bbox_inches="tight")

    if MODEL_NAME == "CPL":
        ax_j.set_ylim(-35, 35)
        ax_j.set_title(f"Jerk {MODEL_NAME} (Rango Completo)")
        fig_j.savefig(OUTDIR / "j_z_evolution_full.pdf", bbox_inches="tight")
    plt.close(fig_j)

# ========================================================
# COMPARATIVAS MULTI-MODELO
# ========================================================
def plot_all_model_comparisons():
    print(f"\n>>> Generando Comparativas Multi-Modelo en: {OUTDIR_COMPARISON}")
    SL = slice(4, -4)
    casos = [("Hz", "Hz", r"Comparativa con Datos $H(z)$"),
             ("FRBs", "FRBs", r"Comparativa con Datos FRBs"),
             ("FRBs_Hz", "Joint", r"Comparativa Conjunta (FRBs+$H(z)$)")]

    for carpeta_tipo, etiqueta_archivo, titulo_base in casos:
        print(f"   -> Procesando caso: {etiqueta_archivo}")
        fig_q, ax_q = plt.subplots(figsize=(8, 6))
        fig_j, ax_j = plt.subplots(figsize=(8, 6))
        found_data = False
        all_j_values = []
        all_q_values = []

        for mod in MODELS_TO_COMPARE:
            mname = mod["name"]
            path = BASE_PATH / mname / carpeta_tipo / f"qj_stats_{mname}.csv"
            if path.exists():
                try:
                    df = pd.read_csv(path).iloc[SL]
                    def smooth_curve(y_vals):
                        return savgol_filter(y_vals, window_length=15, polyorder=3) if len(y_vals) > 15 else y_vals
                    
                    q_med_smooth = smooth_curve(df["q_med"].values)
                    ax_q.plot(df["z"], q_med_smooth, color=mod["color"], lw=2.5, label=mod["label"], antialiased=True)
                    all_q_values.append(q_med_smooth)
                    
                    j_med_smooth = smooth_curve(df["j_med"].values)
                    ax_j.plot(df["z"], j_med_smooth, color=mod["color"], lw=2.5, label=mod["label"], antialiased=True)
                    all_j_values.append(j_med_smooth)
                    found_data = True
                except Exception as e:
                    print(f"      [Error] Leyendo {mname}: {e}")

        if not found_data:
            print(f"      [!] No hay datos para {etiqueta_archivo}.")
            plt.close(fig_q); plt.close(fig_j)
            continue

        q_ymin, q_ymax = -1.5, 1.0
        if all_q_values:
            all_q_data = np.concatenate(all_q_values)
            margin = (np.max(all_q_data) - np.min(all_q_data)) * 0.1
            q_ymin = min(q_ymin, np.min(all_q_data) - margin)
            q_ymax = max(q_ymax, np.max(all_q_data) + margin)
        ax_q.set_ylim(q_ymin, q_ymax)

        j_ymin, j_ymax = -0.5, 3.5
        if all_j_values:
            all_j_data = np.concatenate(all_j_values)
            margin = (np.max(all_j_data) - np.min(all_j_data)) * 0.1
            j_ymin = min(j_ymin, np.min(all_j_data) - margin)
            j_ymax = max(j_ymax, np.max(all_j_data) + margin)
        ax_j.set_ylim(j_ymin, j_ymax)

        ax_q.axhline(0, color="gray", ls="--", lw=1.5)
        ax_q.set_xlabel("Redshift z", fontsize=14)
        ax_q.set_ylabel(r"Desaceleración $q(z)$", fontsize=14)
        ax_q.set_xlim(0, 2.5)
        ax_q.legend(loc="upper right", fontsize=12)
        ax_q.grid(True, alpha=0.15)
        ax_q.set_title(f"{titulo_base} - Desaceleración", fontsize=14)
        fig_q.savefig(OUTDIR_COMPARISON / f"Comparativa_{etiqueta_archivo}_q.pdf", bbox_inches="tight")
        plt.close(fig_q)

        ax_j.axhline(1, color="gray", ls="--", lw=1.5, label=r"$\Lambda$CDM (ref)")
        ax_j.set_xlabel("Redshift z", fontsize=14)
        ax_j.set_ylabel(r"Jerk $j(z)$", fontsize=14)
        ax_j.set_xlim(0, 2.5)
        ax_j.legend(loc="best", fontsize=12)
        ax_j.grid(True, alpha=0.15)
        ax_j.set_title(f"{titulo_base} - Jerk", fontsize=14)
        fig_j.savefig(OUTDIR_COMPARISON / f"Comparativa_{etiqueta_archivo}_j.pdf", bbox_inches="tight")
        plt.close(fig_j)
        print(f"      [OK] Guardados ({etiqueta_archivo}).")

# ------------------------------
# MAIN
# ------------------------------
def main():
    chains = {}
    cols_found = None
    paths = {"FRBs": PATH_FRBs, "Hz": PATH_Hz, "Joint": PATH_FRBs_Hz}

    for lbl, p in paths.items():
        if p.exists():
            print(f"Cargando {lbl}...")
            df = load_chain_df(p)
            data, cnames = chain_cosmo_params(df)
            chains[lbl] = data
            if cols_found is None or len(cnames) > len(cols_found):
                cols_found = cnames
            corner_single(data, get_latex_labels(cnames), lbl, OUTDIR / f"corner_{lbl}.pdf")
            plot_convergence(df, lbl, OUTDIR / f"trace_{lbl}.pdf")

    if len(chains) == 3:
        print("Generando Corner Triple...")
        corner_triple(chains["FRBs"], chains["Hz"], chains["Joint"],
                      get_latex_labels(cols_found), OUTDIR / "corner_triple.pdf")

    print("Generando Overlays separados...")
    df_frb = load_cc_data()
    df_hz = load_hz_points()
    mis_cadenas = [PATH_FRBs, PATH_Hz, PATH_FRBs_Hz]
    plot_overlay_Hz(df_hz, mis_cadenas)
    plot_overlay_DM(df_frb, mis_cadenas)

    print("Generando y actualizando cinemática extendida...")
    plot_qj_extended([PATH_FRBs, PATH_Hz, PATH_FRBs_Hz],
                     [PATH_QJ_FRBs, PATH_QJ_Hz, PATH_QJ_FRBs_Hz])

    plot_all_model_comparisons()

if __name__ == "__main__":
    main()