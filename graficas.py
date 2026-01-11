import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import corner
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ========================================================
# 1. CONFIGURACIÓN DEL USUARIO
# ========================================================
MODEL_NAME = "PEDE"   # <--- CAMBIA ESTO: "LCDM", "CPL", "PEDE"

# Rutas Base
WIN_BASE = r"D:\Ingeniería física\Cosmología\Tésis 1\Ajuste paradiferentes modelos"
WSL_BASE = "/mnt/d/Ingeniería física/Cosmología/Tésis 1/Ajuste paradiferentes modelos"
BASE = WIN_BASE if os.path.isdir(WIN_BASE) else WSL_BASE

# Carpetas de salida
OUTDIR = os.path.join(BASE, MODEL_NAME, "GRAFICAS_GLOBALES")
os.makedirs(OUTDIR, exist_ok=True)
OUTDIR_COMPARISON = os.path.join(BASE, "COMPARATIVA_MODELOS")
os.makedirs(OUTDIR_COMPARISON, exist_ok=True)

# Modelos
MODELS_TO_COMPARE = [
    {"name": "LCDM", "color": "black",   "label": r"$\Lambda$CDM"},
    {"name": "CPL",  "color": "#1f77b4", "label": "CPL"},
    {"name": "PEDE", "color": "#d62728", "label": "PEDE"}
]

def get_path(tipo, archivo):
    return os.path.join(BASE, MODEL_NAME, tipo, archivo)

PATH_FRBs     = get_path("FRBs", f"chain_{MODEL_NAME}_FRBs.csv")
PATH_Hz       = get_path("Hz",   f"chain_{MODEL_NAME}_Hz.csv")
PATH_FRBs_Hz  = get_path("FRBs_Hz", f"chain_{MODEL_NAME}_FRBs_Hz.csv")
PATH_QJ_FRBs     = get_path("FRBs", f"qj_stats_{MODEL_NAME}.csv")
PATH_QJ_Hz       = get_path("Hz",   f"qj_stats_{MODEL_NAME}.csv")
PATH_QJ_FRBs_Hz  = get_path("FRBs_Hz", f"qj_stats_{MODEL_NAME}.csv")
PATH_HZ_DATA  = "/mnt/d/Ingeniería física/Cosmología/Tésis 1/Artículo/Código de réplica de figura/HzTable_MM_BC32.txt"
CATALOGO_FRBS = "/mnt/d/Ingeniería física/Cosmología/Tésis 1/Artículo/Código de réplica de figura/localized_FRBs(1).txt"

print(f">>> Generando gráficas para: {MODEL_NAME}")
print(f">>> Guardando en: {OUTDIR}")

# ------------------------------
# ESTILO VISUAL
# ------------------------------
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 14,
    "axes.labelsize": 24,          
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "figure.constrained_layout.use": False,
    "axes.unicode_minus": False
})

LEVELS_2D = [0.68, 0.95]
COLOR_CASES = {"Hz": "#1f77b4", "FRBs": "#2ca02c", "FRBs+Hz": "#d62728"}

def color_for_title(title: str) -> str:
    for k, v in COLOR_CASES.items():
        if k.lower() in str(title).lower(): return v
    return "black"

# ------------------------------
# Utilidades
# ------------------------------
def add_outside_legend(fig, labels, colors, where=(0.66, 0.76, 0.25, 0.20), use_lines=True, lw=2.5):
    if use_lines:
        handles = [Line2D([0],[0], color=c, lw=lw, label=lab) for lab,c in zip(labels, colors)]
    else:
        handles = [Patch(facecolor='none', edgecolor=c, linewidth=lw, label=lab) for lab,c in zip(labels, colors)]
    ax_leg = fig.add_axes(where)
    ax_leg.axis('off')
    ax_leg.legend(handles=handles, loc="center", frameon=True, fontsize=18)

def remove_local_legends(fig):
    for ax in fig.get_axes():
        lg = ax.get_legend()
        if lg is not None: lg.remove()

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
    df = _normalize_colnames(df)
    wanted = ["h", "Ωb", "Ωm", "w0", "wa"]
    found = [c for c in wanted if c in df.columns]
    return df[found].to_numpy(float), found

def get_latex_labels(cols):
    lab_map = {"h": r"$h$", "Ωb": r"$\Omega_b$", "Ωm": r"$\Omega_m$", "w0": r"$w_0$", "wa": r"$w_a$"}
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
    return [(0,0,0,0), color_periphery, color_center]

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
# CORNER PLOTS (Corrección de Ejes)
# ------------------------------
def format_axes_ticks(fig, ndim):
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            ax.tick_params(axis='both', which='major', labelsize=12)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            
            # --- CORRECCIÓN: SOLO NÚMEROS EN LA CELDA (0,0) ---
            if i == j: 
                if i == 0 and j == 0: 
                    ax.yaxis.set_visible(True)
                    ax.yaxis.set_ticks_position('left')
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='lower'))
                    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                    ax.tick_params(axis='y', which='major', left=True, labelleft=True, labelsize=12, pad=5)
                else: 
                    ax.yaxis.set_visible(False)
                    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            else:
                plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

def corner_single(data, labels, title, outfile_pdf):
    ranges = padded_ranges([data])
    col = color_for_title(title)
    ndim = data.shape[1]
    
    fig = corner.corner(
        data,
        labels=labels,
        range=ranges,
        bins=40,
        smooth=1.5,
        max_n_ticks=3,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        plot_contours=True,
        levels=LEVELS_2D,
        color=col,
        show_titles=True,
        title_kwargs={"fontsize": 16},
        label_kwargs={"fontsize": 28, "labelpad": 25},
        title_fmt=".3f",
        hist_kwargs=dict(density=True, alpha=0.0),
        contour_kwargs=dict(colors=[col], linewidths=1.5),
        contourf_kwargs=dict(colors=get_contour_colors_simple(col), extend="neither")
    )
    _overlay_kde_on_diagonal(fig, data, col, lw=2.5)
    format_axes_ticks(fig, ndim)
    fig.suptitle(f"{title} ({MODEL_NAME})", fontsize=20, y=1.02)
    plt.savefig(outfile_pdf, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

def corner_triple(data_frbs, data_hz, data_both, labels, outfile_pdf):
    ranges = padded_ranges([data_frbs, data_hz, data_both], qlo=0.0, qhi=100.0, pad_frac=0.2)
    col_hz, col_frbs, col_joint = COLOR_CASES["Hz"], COLOR_CASES["FRBs"], COLOR_CASES["FRBs+Hz"]
    ndim = data_hz.shape[1]
    
    fig = corner.corner(
        data_hz, labels=labels, range=ranges, bins=40, smooth=2.0, max_n_ticks=3,
        plot_datapoints=False, plot_density=False, fill_contours=True, plot_contours=True,
        levels=LEVELS_2D, color=col_hz, hist_kwargs=dict(density=True, alpha=0.0),
        contour_kwargs=dict(colors=[col_hz], linewidths=1.5),
        contourf_kwargs=dict(colors=get_contour_colors_simple(col_hz), extend="neither"),
        label_kwargs={"fontsize": 30, "labelpad": 30}, 
        show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 16}
    )
    
    for dat, col in [(data_frbs, col_frbs), (data_both, col_joint)]:
        corner.corner(
            dat, range=ranges, bins=40, smooth=2.0, max_n_ticks=3,
            plot_datapoints=False, plot_density=False, fill_contours=True, plot_contours=True,
            levels=LEVELS_2D, color=col, hist_kwargs=dict(density=True, alpha=0.0),
            fig=fig,
            contour_kwargs=dict(colors=[col], linewidths=1.5),
            contourf_kwargs=dict(colors=get_contour_colors_simple(col), extend="neither")
        )

    _overlay_kde_on_diagonal(fig, data_hz, col_hz)
    _overlay_kde_on_diagonal(fig, data_frbs, col_frbs)
    _overlay_kde_on_diagonal(fig, data_both, col_joint)

    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        ax = axes[i, i]
        ax.set_title(labels[i], fontsize=20, pad=15)
        max_y = 0
        for line in ax.get_lines():
            if len(line.get_ydata()) > 0: max_y = max(max_y, np.max(line.get_ydata()))
        if max_y > 0: ax.set_ylim(0, max_y * 1.3)

    format_axes_ticks(fig, ndim)
    remove_local_legends(fig)
    add_outside_legend(fig, ["Hz", "FRBs", "FRBs+Hz"], [col_hz, col_frbs, col_joint], lw=3.0)
    plt.savefig(outfile_pdf, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)

def plot_convergence(df, label, outfile_pdf):
    df = _normalize_colnames(df)
    cols = [c for c in ["h", "Ωb", "Ωm", "w0", "wa"] if c in df.columns]
    vals = df[cols].to_numpy(float)
    labels = get_latex_labels(cols)
    n_vars = len(cols)
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 2.5*n_vars), sharex=True)
    if n_vars == 1: axes = [axes]
    
    x = np.arange(vals.shape[0])
    for i in range(n_vars):
        y = vals[:, i]
        q16, q50, q84 = np.percentile(y, [16, 50, 84])
        ax = axes[i]
        ax.plot(x, y, 'k', alpha=0.3, lw=0.8)
        ax.axhline(q50, c='r', ls='--', lw=1.5)
        ax.axhline(q16, c='r', ls='--', lw=1.2, alpha=0.7)
        ax.axhline(q84, c='r', ls='--', lw=1.2, alpha=0.7)
        ax.set_ylabel(labels[i], fontsize=24)
        ax.legend([f"{labels[i]} = {q50:.3f}"], loc='upper right', fontsize=12)
    
    axes[-1].set_xlabel("Steps", fontsize=22)
    plt.tight_layout()
    plt.savefig(outfile_pdf, dpi=300)
    plt.close(fig)

# ------------------------------
# FÍSICA Y OVERLAYS
# ------------------------------
f_IGM, f_e = 0.84, 0.88
G, m_p, c_ms = 6.67430e-11, 1.67262e-27, 299792458.0
Mpc_m, pc_m = 3.085677581491367e22, 3.085677581491367e16

def E_Theoretical(z, params_dict):
    z = np.asarray(z, dtype=float)
    Om = params_dict.get("Ωm", 0.3)
    E2 = Om*(1.0+z)**3 + (1.0-Om)
    if MODEL_NAME == "CPL":
        w0, wa = params_dict.get("w0", -1.0), params_dict.get("wa", 0.0)
        de_term = (1-Om) * (1+z)**(3*(1+w0+wa)) * np.exp(-3*wa*z/(1+z))
        E2 = Om*(1+z)**3 + de_term
    elif MODEL_NAME == "PEDE":
        de_term = (1-Om) * (1 - np.tanh(np.log10(1+z))) 
        E2 = Om*(1+z)**3 + de_term
    return np.sqrt(np.abs(E2))

def H_of_z(params_dict, z):
    return 100.0 * float(params_dict.get("h", 0.7)) * E_Theoretical(z, params_dict)

def DM_IGM_model_params(params_dict, z_eval):
    h, Ob = params_dict.get("h", 0.7), params_dict.get("Ωb", 0.05)
    z_grid = np.linspace(0.0, np.max(z_eval)*1.05, 500)
    integrand = (1.0 + z_grid) / E_Theoretical(z_grid, params_dict)
    I_cum = cumulative_trapezoid(integrand, z_grid, initial=0)
    I_target = np.interp(z_eval, z_grid, I_cum)
    H0_SI  = (100.0 * h * 1000.0) / Mpc_m
    rho_c0 = 3.0 * H0_SI**2 / (8.0 * np.pi * G)
    n_e0_cm3 = (Ob * rho_c0 / m_p) * f_IGM * 1e-6
    return (c_ms / H0_SI / pc_m) * n_e0_cm3 * I_target * f_e

def load_cc_data():
    if not os.path.exists(CATALOGO_FRBS): return None
    df = pd.read_csv(CATALOGO_FRBS, sep=r'\s+', header=1, names=['Names','z','DM_obs','RA','Dec','DM_NE2001','DM_YMW16','Type'])
    for c in ['z','DM_obs','DM_NE2001']: df[c] = pd.to_numeric(df[c], errors='coerce')
    valid = df[(df['DM_obs'] - df['DM_NE2001']) > 80.0].copy()
    z = valid['z'].values
    med_host, med_halo = 100.0, 65.0
    if os.path.exists(PATH_FRBs_Hz):
        d = _normalize_colnames(pd.read_csv(PATH_FRBs_Hz))
        if "DM_host" in d: med_host = d["DM_host"].median()
        if "DM_halo" in d: med_halo = d["DM_halo"].median()
    y = valid['DM_obs'].values - valid['DM_NE2001'].values - med_halo - med_host/(1+z)
    return pd.DataFrame({"z": z, "DM": y, "DM_err": np.sqrt(1800 + (80/(1+z))**2)})

def load_hz_points():
    if not os.path.exists(PATH_HZ_DATA): return None
    return pd.read_csv(PATH_HZ_DATA, sep=r'\s+', comment="#", names=["z", "H", "eH"])

def plot_overlay_Hz(df_hz_pts, paths_chains):
    print("   -> Generando Overlay H(z)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if df_hz_pts is not None:
        ax.errorbar(df_hz_pts.z, df_hz_pts.H, yerr=df_hz_pts.eH, fmt='s', ms=6, color='k', capsize=3, label="Datos CC")

    labels, colors = ["FRBs", "Hz", "Joint"], ["#2ca02c", "#1f77b4", "#d62728"]
    z_grid = np.linspace(0, 2.5, 200)
    max_h = 0
    for lbl, path, col in zip(labels, paths_chains, colors):
        if not os.path.exists(path): continue
        chain = _normalize_colnames(pd.read_csv(path)).sample(n=300, random_state=42)
        h_lines = np.array([H_of_z(r.to_dict(), z_grid) for _, r in chain.iterrows()])
        med, lo, hi = np.median(h_lines, 0), np.percentile(h_lines, 16, 0), np.percentile(h_lines, 84, 0)
        ax.plot(z_grid, med, color=col, lw=3, label=lbl)
        ax.fill_between(z_grid, lo, hi, color=col, alpha=0.25)
        max_h = max(max_h, np.max(hi))

    ax.set_xlabel(r"Redshift $z$", fontsize=22)
    ax.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=22)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, max_h * 1.35) 
    ax.legend(fontsize=18, loc="upper left")
    ax.set_title(f"Historia de Expansión - {MODEL_NAME}", fontsize=24)
    plt.savefig(os.path.join(OUTDIR, "overlay_Hz.pdf"), bbox_inches="tight", pad_inches=0.3)
    plt.close()

def plot_overlay_DM(df_frb, paths_chains):
    print("   -> Generando Overlay DM_IGM...")
    fig, ax = plt.subplots(figsize=(10, 8))
    if df_frb is not None:
        ax.errorbar(df_frb.z, df_frb.DM, yerr=df_frb.DM_err, fmt='o', ms=5, color='darkviolet', capsize=2, label="Datos FRBs")

    labels, colors = ["FRBs", "Hz", "Joint"], ["#2ca02c", "#1f77b4", "#d62728"]
    z_grid = np.linspace(0, 2.5, 200)
    for lbl, path, col in zip(labels, paths_chains, colors):
        if not os.path.exists(path): continue
        chain = _normalize_colnames(pd.read_csv(path)).sample(n=300, random_state=42)
        dm_lines = np.array([DM_IGM_model_params(r.to_dict(), z_grid) for _, r in chain.iterrows()])
        med, lo, hi = np.median(dm_lines, 0), np.percentile(dm_lines, 16, 0), np.percentile(dm_lines, 84, 0)
        ax.plot(z_grid, med, color=col, lw=3, label=lbl)
        ax.fill_between(z_grid, lo, hi, color=col, alpha=0.25)

    ax.set_xlabel(r"Redshift $z$", fontsize=22)
    ax.set_ylabel(r"$DM_{IGM}$ [pc cm$^{-3}$]", fontsize=22)
    ax.set_xlim(0, 2.5); ax.set_ylim(0, 3200)
    ax.legend(fontsize=18, loc="upper left")
    ax.set_title(f"Relación de Dispersión - {MODEL_NAME}", fontsize=24)
    plt.savefig(os.path.join(OUTDIR, "overlay_DM.pdf"), bbox_inches="tight", pad_inches=0.3)
    plt.close()

def calculate_kinematics(params, z):
    Om = params.get("Ωm", params.get("Om", 0.3))
    E = E_Theoretical(z, params)
    dE = np.gradient(E, z)
    q = ((1+z)/E)*dE - 1
    j = q*(2*q+1) + (1+z)*np.gradient(q, z)
    return q, j

def plot_qj_extended(paths_chains, paths_save_stats):
    print("   -> Generando cinemática...")
    z_full = np.linspace(0, 2.5, 206)
    fig_q, ax_q = plt.subplots(figsize=(10, 6))
    fig_j, ax_j = plt.subplots(figsize=(10, 6))
    
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    for lbl, path, col, save_path in zip(["FRBs", "Hz", "Joint"], paths_chains, colors, paths_save_stats):
        if not os.path.exists(path): continue
        chain = _normalize_colnames(pd.read_csv(path)).sample(n=300, random_state=42)
        qs, js = [], []
        for _, row in chain.iterrows():
            q, j = calculate_kinematics(row.to_dict(), z_full)
            qs.append(q); js.append(j)
        q_med, q_lo, q_hi = np.median(qs, 0), np.percentile(qs, 16, 0), np.percentile(qs, 84, 0)
        j_med, j_lo, j_hi = np.median(js, 0), np.percentile(js, 16, 0), np.percentile(js, 84, 0)
        
        sl = slice(3, -3)
        pd.DataFrame({"z": z_full[sl], "q_med": q_med[sl], "q_lo": (q_med-q_lo)[sl], "q_hi": (q_hi-q_med)[sl],
                      "j_med": j_med[sl], "j_lo": (j_med-j_lo)[sl], "j_hi": (j_hi-j_med)[sl]}).to_csv(save_path, index=False)

        for ax, med, lo, hi in [(ax_q, q_med, q_lo, q_hi), (ax_j, j_med, j_lo, j_hi)]:
            ax.plot(z_full[sl], med[sl], label=lbl, color=col, lw=3)
            ax.fill_between(z_full[sl], lo[sl], hi[sl], color=col, alpha=0.2)

    for ax, ylab, title in [(ax_q, "q(z)", "Desaceleración"), (ax_j, "j(z)", "Jerk")]:
        ax.axhline(0 if "q" in ylab else 1, c='k', ls=':', lw=1.5)
        # --- AUMENTO ---
        ax.set_xlabel("Redshift z", fontsize=26)
        ax.set_ylabel(ylab, fontsize=26)
        ax.set_xlim(0, 2.5)
        ax.legend(fontsize=16)
        ax.set_title(f"{title} - {MODEL_NAME}", fontsize=22)

    fig_q.savefig(os.path.join(OUTDIR, "q_z_evolution.pdf"), bbox_inches="tight", pad_inches=0.2)
    ax_j.set_ylim(-5, 5)
    fig_j.savefig(os.path.join(OUTDIR, "j_z_evolution_zoom.pdf"), bbox_inches="tight", pad_inches=0.2)
    if MODEL_NAME == "CPL":
        ax_j.set_ylim(-35, 35)
        fig_j.savefig(os.path.join(OUTDIR, "j_z_evolution_full.pdf"), bbox_inches="tight")
    plt.close('all')

# --- MODIFICADO: RANGOS DINÁMICOS PARA EVITAR CORTES ---
def plot_all_model_comparisons():
    print(f"\n>>> Generando Comparativas Multi-Modelo...")
    for tipo, label_file, title in [("Hz", "Hz", r"Datos $H(z)$"), ("FRBs", "FRBs", "Datos FRBs"), ("FRBs_Hz", "Joint", "Joint (FRBs+Hz)")]:
        fig_q, ax_q = plt.subplots(figsize=(10, 8))
        fig_j, ax_j = plt.subplots(figsize=(10, 8))
        has_data = False
        
        all_q_data = []
        all_j_data = []
        
        for mod in MODELS_TO_COMPARE:
            path = os.path.join(BASE, mod["name"], tipo, f"qj_stats_{mod['name']}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                z = df["z"].values
                q = savgol_filter(df["q_med"], 15, 3)
                j = savgol_filter(df["j_med"], 15, 3)
                ax_q.plot(z, q, color=mod["color"], lw=3, label=mod["label"])
                ax_j.plot(z, j, color=mod["color"], lw=3, label=mod["label"])
                all_q_data.append(q)
                all_j_data.append(j)
                has_data = True
        
        if not has_data: continue
        
        for ax, ylab, tit in [(ax_q, r"$q(z)$", "Desaceleración"), (ax_j, r"$j(z)$", "Jerk")]:
            ax.axhline(0 if "q" in ylab else 1, c='gray', ls='--', lw=1.5)
            ax.set_xlabel("Redshift z", fontsize=26)
            ax.set_ylabel(ylab, fontsize=26)
            ax.legend(fontsize=18)
            ax.set_title(f"{title} - {tit}", fontsize=24)
            ax.grid(alpha=0.15)
        
        # --- CALCULO DE LÍMITES DINÁMICOS ---
        # Q
        if all_q_data:
            flat_q = np.concatenate(all_q_data)
            q_min, q_max = np.min(flat_q), np.max(flat_q)
            q_pad = (q_max - q_min) * 0.1
            ax_q.set_ylim(q_min - q_pad, q_max + q_pad)
        else:
            ax_q.set_ylim(-2.5, 1.5)

        # J
        if all_j_data:
            flat_j = np.concatenate(all_j_data)
            j_min, j_max = np.min(flat_j), np.max(flat_j)
            j_pad = (j_max - j_min) * 0.1
            ax_j.set_ylim(j_min - j_pad, j_max + j_pad)
        else:
            ax_j.set_ylim(-5, 10)

        fig_q.savefig(os.path.join(OUTDIR_COMPARISON, f"Comparativa_{label_file}_q.pdf"), bbox_inches="tight")
        fig_j.savefig(os.path.join(OUTDIR_COMPARISON, f"Comparativa_{label_file}_j.pdf"), bbox_inches="tight")
        plt.close('all')

def main():
    chains, cols_found = {}, None
    paths = {"FRBs": PATH_FRBs, "Hz": PATH_Hz, "Joint": PATH_FRBs_Hz}
    for lbl, p in paths.items():
        if os.path.exists(p):
            print(f"Cargando {lbl}...")
            df = load_chain_df(p)
            data, cnames = chain_cosmo_params(df)
            chains[lbl] = data
            if cols_found is None or len(cnames) > len(cols_found): cols_found = cnames
            corner_single(data, get_latex_labels(cnames), lbl, os.path.join(OUTDIR, f"corner_{lbl}.pdf"))
            plot_convergence(df, lbl, os.path.join(OUTDIR, f"trace_{lbl}.pdf"))

    if len(chains) == 3:
        print("Generando Corner Triple...")
        corner_triple(chains["FRBs"], chains["Hz"], chains["Joint"], get_latex_labels(cols_found), os.path.join(OUTDIR, "corner_triple.pdf"))

    print("Generando Overlays...")
    df_frb, df_hz = load_cc_data(), load_hz_points()
    plot_overlay_Hz(df_hz, [PATH_FRBs, PATH_Hz, PATH_FRBs_Hz])
    plot_overlay_DM(df_frb, [PATH_FRBs, PATH_Hz, PATH_FRBs_Hz])
    plot_qj_extended([PATH_FRBs, PATH_Hz, PATH_FRBs_Hz], [PATH_QJ_FRBs, PATH_QJ_Hz, PATH_QJ_FRBs_Hz])
    plot_all_model_comparisons()

if __name__ == "__main__":
    main()
