# 1) Instalación e imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from iminuit import Minuit
import os
import corner
import emcee
import multiprocessing as mp
from IPython.display import display, Math
import sympy as sp
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid # Para Scipy reciente
from tqdm import tqdm # Barra de progreso
import time
from multiprocessing import Pool
from pathlib import Path

# Si te da error, usa: from scipy.integrate import cumtrapz
# Constantes SI
f_IGM = 0.84#fracción de bariones que están en el medio intergaláctico (IGM).
f_e=0.88 #fracción de electrones libres por barión en el IGM
G   = 6.67430e-11           # m^3 kg^-1 s^-2
m_p = 1.67262192369e-27     # kg
c_ms = 299792458.0          # m/s
Mpc_m = 3.085677581491367e22  # m/Mpc
pc_m  = 3.085677581491367e16  # m/pc

# ===== Variables globales para el ajuste =====
Z = None          # aquí se guardará el array de redshifts de los FRBs
DM_OBS = None     # aquí se guardará el array con las DM observadas (medidas de los FRBs)
DM_MW_USE = None  # aquí se guardará la contribución de la Vía Láctea (según NE2001 o YMW16)
DM_ERR_BASE = None # aquí se guardarán los errores base (σ_i) de cada FRB
NPTS = 0          # número de puntos de datos (FRBs)
OM_PRIOR_GAUSS = True           # activa si quieres prior directo en Ωm
WM_PRIOR = True                  # prior en ωm = Ωm h² (más físico)
WM_MU, WM_SIGMA = 0.143, 0.003   # afloja σ si no quieres que domine

# ============================
# 2) CONFIG (mínima y coherente)
# ============================

# --- 2.1 SELECCIÓN DEL MODELO ---
MODEL_NAME = "PEDE"   # Opciones: "LCDM", "CPL", "PEDE"

# --- 2.2 SELECCIÓN DE DATOS (Define qué carpetas se crearán) ---
RUN_FRBS = True      # ¿Usar FRBs?
RUN_HZ   = True      # ¿Usar H(z)?

# Determinamos la etiqueta de la subcarpeta automáticamente
if RUN_FRBS and RUN_HZ:
    DATA_LABEL = "FRBs_Hz"
elif RUN_FRBS:
    DATA_LABEL = "FRBs"
elif RUN_HZ:
    DATA_LABEL = "Hz"
else:
    raise ValueError("¡Debes activar al menos RUN_FRBS o RUN_HZ!")

# --- 2.3 GESTIÓN DE CARPETAS (DINÁMICA Y RELATIVA) ---
# Usamos el directorio actual como base
BASE_PATH = Path.cwd() / "Ajuste_Modelos"

# Construimos la ruta final: Base / Modelo / Tipo_Datos
ruta_pdfs = BASE_PATH / MODEL_NAME / DATA_LABEL
csv_path = ruta_pdfs

# Creamos la carpeta (y subcarpetas) si no existen
ruta_pdfs.mkdir(parents=True, exist_ok=True)
print(f"\n>>> Los resultados se guardarán en: {ruta_pdfs}\n")

# --- 2.4 ARCHIVOS DE ENTRADA (RELATIVOS) ---
# Se asume que los archivos .txt están en el mismo directorio que el script
# o ajusta según tu estructura de carpetas
file_path = "localized_FRBs(1).txt"
hz_file_path = "HzTable_MM_BC32.txt"

# Modelo MW por defecto
MW_MODEL = 'NE2001'

# Errores base
USE_REALISTIC_ERRORS = True
SIGMA_HOST_BASE = 80.0

# Límites de parámetros (Minuit)
H_LIMS  = (0.4, 1.0)
OM_LIMS = (0.0, 1.0)
OB_LIMS = (0.0, 0.2)

# Prior suave en Ω_b
OMEGAB_PRIOR_MU    = 0.0486
OMEGAB_PRIOR_SIGMA = 0.0060
# Priors suaves en h
h_PRIOR_MU    = 0.6764
h_PRIOR_SIGMA = 0.0052
conv = 1000.0 / Mpc_m

# DM_host
HOST_LIMS    = (0.0, 250.0)
DM_HOST_INIT = 100.0

# DM_halo
DM_HALO_LIMS    = (0.0, 200.0)
DM_HALO_INIT    = 65.0
DM_HALO_MU      = 65.0
DM_HALO_SIGMA   = 30.0

# Escala global de errores
SCALE_ERR_INIT = 1.2
SCALE_ERR_LIMS = (0.7, 3.5)

# ============= Configuracion EMCEE (OPTIMIZADA) ===============
ndim      = 5
nwalkers  = 32 # Reducido a múltiplo de cpu count (usualmente 8 o 16)
nsteps_wu = 500  # Warmup más corto pero suficiente
nsteps    = 5000 # Pasos suficientes para convergencia razonable

# Configuración inicial MCMC
OM_MU, OM_SIGMA = 0.3, 0.1
rng    = np.random.default_rng()

# Definimos centro y escala inicial según modelo
if MODEL_NAME == "CPL":
    # h, Ob, Om, w0, wa, host, halo
    center = np.array([0.676, 0.049, 0.30, -1.0, 0.0, 100.0, 65.0])
    scale  = np.array([0.02,  0.005, 0.05,  0.1, 0.1,  40.0, 30.0])
    ndim   = 7
else: # LCDM y PEDE
    # h, Ob, Om, host, halo
    center = np.array([0.676, 0.049, 0.30, 100.0, 65.0])
    scale  = np.array([0.02,  0.005, 0.05,  40.0, 30.0])
    ndim   = 5

pos0 = center + rng.normal(0, 1, size=(nwalkers, ndim)) * scale

# ============================
# 3) Leer datos
# ============================
try:
    df = pd.read_csv(
        file_path, sep=r'\s+', header=1,
        names=['Names', 'Redshift', 'DM_obs', 'RA', 'Dec',
               'DM_MW_NE2001', 'DM_MW_YMW16', 'Type']
    )
    df['Type'] = pd.to_numeric(df['Type'], errors='coerce').astype('Int64')
    df['DM_MW_NE2001'] = pd.to_numeric(df['DM_MW_NE2001'], errors='coerce')
    df['DM_MW_YMW16']  = pd.to_numeric(df['DM_MW_YMW16'],  errors='coerce')
    # ΔDM_MW = NE2001 - YMW16
    df['dDM_MW'] = df['DM_MW_NE2001'] - df['DM_MW_YMW16']
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de FRBs: {file_path}")
    if RUN_FRBS: exit()

try:
    #Datos de cronometros cósmicos
    df_Hz = pd.read_csv(
        hz_file_path,
        delim_whitespace=True,   # separador = espacios
        comment="#",             # ignora líneas que empiecen con "#"
        names=["z", "H", "sigma_H"]
    )

    HZ_Z    = df_Hz["z"].to_numpy(float)
    HZ_HOBS = conv*df_Hz["H"].to_numpy(float)
    HZ_HERR = conv*df_Hz["sigma_H"].to_numpy(float)
    print("Datos H(z):", len(HZ_Z), "puntos cargados")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de Hz: {hz_file_path}")
    if RUN_HZ: exit()


# ============================
# 4) MODELO UNIVERSAL (ΛCDM / CPL / PEDE)
# ============================

# Definimos todos los símbolos posibles
z_sym, h_sym, Om_sym, Ob_sym = sp.symbols('z h Omega_m Omega_b')
w0_sym, wa_sym = sp.symbols('w0 wa')

# Lógica de selección (¡Solo cambias MODEL_NAME arriba y esto se ajusta solo!)
if MODEL_NAME == "LCDM":
    # Parametros cosmológicos esperados: [h, Ob, Om]
    SYMBOLS_COSMO = [h_sym, Ob_sym, Om_sym]
    E_ARG_FORMULA = Om_sym * (1 + z_sym)**3 + (1 - Om_sym)

elif MODEL_NAME == "CPL":
    # Parametros: [h, Ob, Om, w0, wa]
    SYMBOLS_COSMO = [h_sym, Ob_sym, Om_sym, w0_sym, wa_sym]
    dark_energy = (1 - Om_sym) * (1 + z_sym)**(3 * (1 + w0_sym + wa_sym)) * sp.exp(-3 * wa_sym * z_sym / (1 + z_sym))
    E_ARG_FORMULA = Om_sym * (1 + z_sym)**3 + dark_energy

elif MODEL_NAME == "PEDE":
    # Parametros: [h, Ob, Om]
    SYMBOLS_COSMO = [h_sym, Ob_sym, Om_sym]
    term_pede = (1 - Om_sym) * (1 - sp.tanh(sp.log(1 + z_sym) / sp.log(10)))
    E_ARG_FORMULA = Om_sym * (1 + z_sym)**3 + term_pede

# --- 4.2 Compilación "Mágica" (No tocar) ---
ALL_ARGS = SYMBOLS_COSMO + [z_sym]

print(f"Generando función numérica para modelo {MODEL_NAME}...")
_E_arg_numeric = sp.lambdify(ALL_ARGS, E_ARG_FORMULA, 'numpy')


# --- 4.3 Función E_Modelo UNIVERSAL ---
def E_Modelo(z, theta_cosmo):
    arg = _E_arg_numeric(*theta_cosmo, z)
    if np.ndim(arg) == 0:
        if arg < 0: return np.nan
    else:
        arg[arg < 0] = np.nan
    return arg**0.5

# --- Funciones auxiliares adaptadas ---
def H0_SI_from_h(h):
    return (100.0 * h * 1000.0) / Mpc_m

def H_of_z_SI(z, theta_cosmo):
    h = theta_cosmo[0]
    return H0_SI_from_h(h) * E_Modelo(z, theta_cosmo)

# Modelo DM_IGM actualizado para aceptar theta_cosmo
def modelo_dm_igm(z_array, theta_cosmo):
    """
    VERSIÓN VECTORIZADA DE ALTA VELOCIDAD.
    """
    z_targets = np.atleast_1d(z_array)
    z_max = np.max(z_targets)

    h = theta_cosmo[0]
    Ob = theta_cosmo[1]

    H0_SI  = H0_SI_from_h(h)
    rho_c0 = 3.0 * H0_SI**2 / (8.0 * np.pi * G)
    n_e0_m3  = (Ob * rho_c0 / m_p) * f_IGM
    n_e0_cm3 = n_e0_m3 * 1e-6
    const_fisica = ((c_ms / H0_SI) / pc_m) * n_e0_cm3 * f_e

    z_grid = np.linspace(0.0, z_max * 1.05, 500)
    E_vals = E_Modelo(z_grid, theta_cosmo)

    if np.any(np.isnan(E_vals)):
        return np.full_like(z_targets, np.nan)

    integrand = (1.0 + z_grid) / E_vals
    I_cum = cumulative_trapezoid(integrand, z_grid, initial=0)
    I_targets = np.interp(z_targets, z_grid, I_cum)

    return const_fisica * I_targets

# PASO 4: Función que crea q(z) y j(z)
def compilar_q_j_universal(E_arg_sym, z_s, lista_simbolos_cosmo):
    print("="*30)
    print(f"Derivando fórmulas simbólicas para: {MODEL_NAME}...")

    E_sym = sp.sqrt(E_arg_sym)
    dE_dz_sym = sp.diff(E_sym, z_s)
    q_sym = -1 + (1 + z_s) / E_sym * dE_dz_sym

    if MODEL_NAME in ["PEDE", "CPL"]:
        print(f">>> Modo Rápido activado para {MODEL_NAME}: Saltando simplificación algebraica.")
        dq_dz_sym = sp.diff(q_sym, z_s)
        j_sym = q_sym * (2 * q_sym + 1) + (1 + z_s) * dq_dz_sym
    else:
        print(">>> Simplificando expresiones (LCDM)...")
        q_sym = sp.simplify(q_sym)
        dq_dz_sym = sp.diff(q_sym, z_s)
        j_sym = sp.simplify(q_sym * (2 * q_sym + 1) + (1 + z_s) * dq_dz_sym)

    # --- GUARDAR EN ARCHIVO DE TEXTO ---
    nombre_txt = f"formulas_matematicas_{MODEL_NAME}.txt"
    ruta_archivo = ruta_pdfs / nombre_txt

    try:
        with open(ruta_archivo, "w", encoding="utf-8") as f:
            f.write(f"RESULTADOS ANALÍTICOS EXACTOS PARA: {MODEL_NAME}\n")
            f.write("="*60 + "\n\n")
            f.write("PARAMETRO DE DESACELERACION q(z):\n")
            f.write("-" * 30 + "\n")
            f.write("Código LaTeX:\n")
            f.write(sp.latex(q_sym))
            f.write("\n\n" + "="*60 + "\n\n")
            f.write("PARAMETRO JERK j(z):\n")
            f.write("-" * 30 + "\n")
            f.write("Código LaTeX:\n")
            f.write(sp.latex(j_sym))
        print(f"Fórmulas simbólicas guardadas en:\n -> {ruta_archivo}")
    except Exception as e:
        print(f"Advertencia: No se pudo guardar el archivo de texto: {e}")

    print("Compilando funciones numéricas optimizadas (CSE)...")
    args_lambda = [z_s] + lista_simbolos_cosmo
    q_func_num = sp.lambdify(args_lambda, q_sym, 'numpy', cse=True)
    j_func_num = sp.lambdify(args_lambda, j_sym, 'numpy', cse=True)
    print("Funciones compiladas correctamente.")
    print("="*30)
    return q_func_num, j_func_num

# PASO 5: Ejecución (Creación de funciones)
q_func, j_func = compilar_q_j_universal(E_ARG_FORMULA, z_sym, SYMBOLS_COSMO)

# ============================
# 5) Selección y errores
# ============================
def seleccionar_validos(df, mw_model='NE2001', case=0, dm_cut=80.0):
    if mw_model.upper() == 'NE2001':
        dm_mw_use = df['DM_MW_NE2001']
    else:
        dm_mw_use = df['DM_MW_YMW16']

    df = df.copy()
    df['DM_MW_USE'] = dm_mw_use
    base_mask = (df['DM_obs'] - df['DM_MW_USE']) > dm_cut

    if case == 1:
        extra_mask = ~(df['dDM_MW'] > 50)
    elif case == 2:
        extra_mask = ~(np.abs(df['dDM_MW']) > 50)
    elif case == 3:
        extra_mask = (df['DM_MW_USE'] < 100)
    else:
        extra_mask = np.ones(len(df), dtype=bool)

    mask = base_mask & extra_mask
    return df[mask].copy(), df[~mask].copy(), int(base_mask.sum()), int(mask.sum())

# ============================
# 6) CHI-CUADRADO Y PRIORS
# ============================
def chi2_FRBS(theta, scale_err):
    DM_host = theta[-2]
    DM_halo = theta[-1]
    theta_cosmo = theta[:-2]

    dm_model = modelo_dm_igm(Z, theta_cosmo)
    if np.any(np.isnan(dm_model)): return np.inf

    resid = DM_OBS - DM_MW_USE - DM_halo - DM_host/(1.0+Z) - dm_model
    s = scale_err if scale_err > 1e-6 else 1e-6
    sigma = s * DM_ERR_BASE
    return np.sum((resid/sigma)**2)

def chi2_Hz(theta, scale_err):
    theta_cosmo = theta[:-2]
    H_model = np.array([H_of_z_SI(zi, theta_cosmo) for zi in HZ_Z])
    return np.sum(((HZ_HOBS - H_model) / HZ_HERR) ** 2)

def prepriors(theta, scale_err):
    theta_cosmo = theta[:-2]
    DM_host     = theta[-2]
    DM_halo     = theta[-1]
    h  = theta_cosmo[0]
    Ob = theta_cosmo[1]
    Om = theta_cosmo[2]

    if not (H_LIMS[0]     <= h       <= H_LIMS[1]):     return -np.inf
    if not (OB_LIMS[0]    <= Ob      <= OB_LIMS[1]):    return -np.inf
    if not (OM_LIMS[0]    <= Om      <= OM_LIMS[1]):    return -np.inf
    if not (HOST_LIMS[0]  <= DM_host <= HOST_LIMS[1]):  return -np.inf
    if not (DM_HALO_LIMS[0] <= DM_halo <= DM_HALO_LIMS[1]): return -np.inf

    if MODEL_NAME == "CPL":
        w0 = theta_cosmo[3]
        wa = theta_cosmo[4]
        if not (-10.0 <= w0 <= 10.0): return -np.inf
        if not (-50.0 <= wa <= 50.0): return -np.inf
    return 0

def priors(theta, scale_err):
    if prepriors(theta, scale_err) != 0:
        return -np.inf

    theta_cosmo = theta[:-2]
    DM_host     = theta[-2]
    DM_halo     = theta[-1]
    h  = theta_cosmo[0]
    Ob = theta_cosmo[1]
    Om = theta_cosmo[2]

    chilog  = 2.0 * NPTS * np.log(scale_err if scale_err > 1e-6 else 1e-6)
    chilog += ((h - h_PRIOR_MU)/h_PRIOR_SIGMA)**2
    chilog += ((Om - OM_MU)/OM_SIGMA)**2
    chilog += ((Ob - OMEGAB_PRIOR_MU)/OMEGAB_PRIOR_SIGMA)**2
    chilog += ((DM_halo - DM_HALO_MU)/DM_HALO_SIGMA)**2
    chilog += ((DM_host - 100.0)/80.0)**2

    if WM_PRIOR:
        chilog += ((Om*(h**2) - WM_MU)/WM_SIGMA)**2
    return chilog

def set_datos_para_fit(z, DM_obs, DM_MW, dm_errors_base):
    global Z, DM_OBS, DM_MW_USE, DM_ERR_BASE, NPTS
    Z = np.asarray(z, float)
    DM_OBS = np.asarray(DM_obs, float)
    DM_MW_USE = np.asarray(DM_MW, float)
    DM_ERR_BASE = np.asarray(dm_errors_base, float)
    NPTS = len(Z)

# =========================================
# DEFINICIÓN DE LA FUNCIÓN DE PROBABILIDAD
# =========================================
def log_prob_4d(theta, scale_err):
    chi2 = 0.0
    if RUN_FRBS:
       chi2 += chi2_FRBS(theta, scale_err)
    if RUN_HZ:
       chi2 += chi2_Hz(theta, scale_err)

    prior = priors(theta, scale_err)
    total = chi2 + prior

    if not np.isfinite(total):
        return -np.inf
    return -0.5 * total

# =========================================
# PASO 1.5) PREPARACIÓN DE DATOS (GLOBAL)
# =========================================
if RUN_FRBS:
    print(">>> Procesando datos FRBs...")
    valid, invalid, base_n, final_n = seleccionar_validos(df, mw_model=MW_MODEL, case=2)
    z_frb  = valid['Redshift'].to_numpy(float)
    yO_frb = valid['DM_obs'].to_numpy(float)
    yM_frb = valid['DM_MW_USE'].to_numpy(float)

    if USE_REALISTIC_ERRORS:
        sigma_MW, sigma_halo = 30.0, 30.0
        dm_err_frb = np.sqrt((SIGMA_HOST_BASE/(1.0+z_frb))**2 + sigma_MW**2 + sigma_halo**2)
    else:
        DM_halo_assume = 65.0
        dm_err_frb = np.maximum(0.1*(yO_frb - yM_frb - DM_halo_assume), 20.0)

    set_datos_para_fit(z_frb, yO_frb, yM_frb, dm_err_frb)
    print(f"   -> FRBs cargados: {NPTS} puntos.")
elif RUN_HZ:
    print(">>> Modo solo H(z) activo.")
    Z = None

# =========================================
# PASO 2) Definir log_prob de emcee
# =========================================
print("Probando log_prob con pos0[0]...")
test_prob = log_prob_4d(pos0[0], SCALE_ERR_INIT)
print(f"log_prob(test) = {test_prob}")

# =========================================
# PASO 3 y 4) Inicializar y Ejecución (MCMC)
# =========================================
label = DATA_LABEL
print(f"\n>>> Ejecutando MCMC para modelo: {MODEL_NAME} ({label})")

if __name__ == "__main__":
    # Autodetección de CPUs
    ncpu = mp.cpu_count()
    print(f"Usando {ncpu} núcleos de CPU")

    with Pool(processes=ncpu) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob_4d, args=(SCALE_ERR_INIT,), pool=pool
        )

        print(f"Iniciando Warmup ({nsteps_wu} pasos)...")
        # tqdm para barra de progreso
        for _ in tqdm(sampler.sample(pos0, iterations=nsteps_wu, progress=False), total=nsteps_wu):
            pass
        
        pos = sampler.get_last_sample().coords
        sampler.reset()
        print(f"Warmup terminado.")

        print(f"Iniciando Producción ({nsteps} pasos)...")
        for _ in tqdm(sampler.sample(pos, iterations=nsteps, progress=False), total=nsteps):
            pass

    # =========================================
    # PASO 5) Post-proceso DINÁMICO
    # =========================================
    if MODEL_NAME == "CPL":
        param_labels = ["h", "Ob", "Om", "w0", "wa", "DM_host", "DM_halo"]
        fmts         = [".4f", ".4f", ".4f", ".3f", ".3f", ".1f", ".2f"]
    elif MODEL_NAME == "LCDM" or MODEL_NAME == "PEDE":
        param_labels = ["h", "Ob", "Om", "DM_host", "DM_halo"]
        fmts         = [".4f", ".4f", ".4f", ".1f", ".2f"]

    try:
        tau = sampler.get_autocorr_time(tol=0, quiet=True)
        if not np.all(np.isfinite(tau)):
            print("Advertencia: tau no convergió, usando defaults.")
            burnin, thin = 500, 5
        else:
            burnin = int(2 * np.max(tau))
            thin   = max(1, int(0.5 * np.min(tau)))
            print(f"Tau estimado: {tau}")
    except:
        print("No se pudo calcular tau. Usando defaults.")
        burnin, thin = 500, 5

    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    print(f"Muestras planas finales: {flat_samples.shape}")

    stats = []
    for i in range(ndim):
        q16, q50, q84 = np.percentile(flat_samples[:, i], [16, 50, 84])
        stats.append([q50, q50-q16, q84-q50])
    stats = np.array(stats)

    print("\n=== Resultados del Ajuste ===")
    for i, name in enumerate(param_labels):
        med, lo, hi = stats[i]
        fmt = fmts[i] if i < len(fmts) else ".4f"
        print(f"{name} = {format(med, fmt)} (+{format(hi, fmt)} / -{format(lo, fmt)})")

    # =============================================================================
    # 6. GUARDAR RESULTADOS EN CSV
    # =============================================================================
    data_dict = {
        "Parametro": param_labels + ["H0"],
        "Mediana":   list(stats[:,0]) + [stats[0,0]*100],
        "Err_-":     list(stats[:,1]) + [stats[0,1]*100],
        "Err_+":     list(stats[:,2]) + [stats[0,2]*100]
    }
    df_results = pd.DataFrame(data_dict)

    theta_med = stats[:, 0]
    chi2_val = 0.0
    if RUN_FRBS: chi2_val += chi2_FRBS(theta_med, SCALE_ERR_INIT)
    if RUN_HZ:   chi2_val += chi2_Hz(theta_med, SCALE_ERR_INIT)

    n_frb = int(NPTS) if RUN_FRBS else 0
    n_hz  = int(len(HZ_Z)) if RUN_HZ else 0
    N_total = n_frb + n_hz
    k = ndim
    dof = max(1, N_total - k)
    chi2_red = chi2_val / dof
    aic_val = chi2_val + 2 * k
    bic_val = chi2_val + k * np.log(N_total)

    extras = pd.DataFrame([
        {"Parametro": "chi2_total", "Mediana": chi2_val},
        {"Parametro": "dof",        "Mediana": dof},
        {"Parametro": "chi2_red",   "Mediana": chi2_red},
        {"Parametro": "AIC",        "Mediana": aic_val},
        {"Parametro": "BIC",        "Mediana": bic_val},
        {"Parametro": "N_datos",    "Mediana": N_total}
    ])

    df_final = pd.concat([df_results, extras], ignore_index=True)
    nombre_archivo_res = f"resultados_{MODEL_NAME}_{label}.csv"
    ruta_completa_res = ruta_pdfs / nombre_archivo_res
    df_final.to_csv(ruta_completa_res, index=False)
    print(f"\nResultados guardados en:\n -> {ruta_completa_res}")

    chain_path = ruta_pdfs / f"chain_{MODEL_NAME}_{label}.csv"
    pd.DataFrame(flat_samples, columns=param_labels).to_csv(chain_path, index=False)

    # ==========================================================
    # ===== CÁLCULO DINÁMICO DE q(z) y j(z) =====
    # ==========================================================
    print("\nCalculando dinámica q(z), j(z)...")
    if Z is not None:
        z_max_ref = float(Z.max())
    elif 'HZ_Z' in globals() and len(HZ_Z) > 0:
        z_max_ref = float(HZ_Z.max())
    else:
        z_max_ref = 2.5

    z_grid = np.linspace(0.0, z_max_ref * 1.2, 100)
    cosmo_samples = flat_samples[:, :-2]
    q_trajectories = []
    j_trajectories = []
    indices_random = np.random.choice(len(cosmo_samples), size=min(2000, len(cosmo_samples)), replace=False)

    for idx in indices_random:
        theta_c = cosmo_samples[idx]
        try:
            qq = q_func(z_grid, *theta_c)
            jj = j_func(z_grid, *theta_c)
            if np.ndim(qq) == 0: qq = np.full_like(z_grid, qq)
            if np.ndim(jj) == 0: jj = np.full_like(z_grid, jj)
            q_trajectories.append(qq)
            j_trajectories.append(jj)
        except Exception as e:
            if len(q_trajectories) == 0: print(f"ERROR en q(z): {e}")
            continue

    q_trajectories = np.array(q_trajectories)
    j_trajectories = np.array(j_trajectories)

    q_16, q_50, q_84 = np.percentile(q_trajectories, [16, 50, 84], axis=0)
    j_16, j_50, j_84 = np.percentile(q_trajectories, [16, 50, 84], axis=0)

    df_qj = pd.DataFrame({
        "z": z_grid,
        "q_med": q_50, "q_lo": q_50-q_16, "q_hi": q_84-q_50,
        "j_med": j_50, "j_lo": j_50-j_16, "j_hi": j_84-j_50,
    })
    df_qj.to_csv(ruta_pdfs / f"qj_stats_{MODEL_NAME}.csv", index=False)
    print("Estadísticas q(z) guardadas.")

    # ========================================================
    # ===== TRANSICIÓN z (q=0) =====
    # ========================================================
    print("Calculando z de transición...")
    if MODEL_NAME == "CPL" and RUN_FRBS:
        z_search_min, z_search_max = 0.5, 1.0
    else:
        z_search_min, z_search_max = 0.0, 5.0

    z_trans_list = []
    for idx in indices_random:
        theta_c = cosmo_samples[idx]
        def f_root(z_val): return q_func(z_val, *theta_c)
        try:
            if f_root(z_search_min) * f_root(z_search_max) < 0:
                root = brentq(f_root, z_search_min, z_search_max)
                z_trans_list.append(root)
        except: pass

    txt_path = ruta_pdfs / f"z_transition_{MODEL_NAME}_{label}.txt"
    if len(z_trans_list) > 0:
        zt_16, zt_50, zt_84 = np.percentile(z_trans_list, [16, 50, 84])
        err_minus = zt_50 - zt_16
        err_plus  = zt_84 - zt_50
        resultado_str = f"{zt_50:.3f} (+{err_plus:.3f} / -{err_minus:.3f})"
        print(f"z_transition = {resultado_str}")

        with open(txt_path, "w") as f:
            f.write(f"RESULTADOS DE TRANSICION (q=0)\n================================\n")
            f.write(f"Modelo: {MODEL_NAME}\nDatos:  {label}\n--------------------------------\n")
            f.write(f"Mediana (z_50): {zt_50:.6f}\nLímite Inf (z_16): {zt_16:.6f}\n")
            f.write(f"Límite Sup (z_84): {zt_84:.6f}\nError (+): {err_plus:.6f}\nError (-): {err_minus:.6f}\n")
            f.write(f"\nFormato Paper:\nz_t = {resultado_str}\n")
        print(f"--> Archivo guardado en: {txt_path}")
    else:
        print("No se encontró transición en el rango [0, 5].")
        with open(txt_path, "w") as f:
            f.write(f"Modelo: {MODEL_NAME} - {label}\nRESULTADO: No se encontró transición.\n")