import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==============================================================================
# 1. CÀRREGA I PREPARACIÓ DE LES DADES
# ==============================================================================
x_grid = np.load("x.npy")                     # Vector 1D de 64 píxels espacials
times = np.load("times.npy")                  # Vector 1D de 12 fotogrames temporals
noise_std = float(np.load("noise_std.npy"))   # Desviació estàndard del soroll (sigma = 0.025)

nu_train = np.load("train_frequencies.npy")   # Forma: (12 temps, 9 freqüències)
nu_val = np.load("validation_frequencies.npy")# Forma: (12 temps, 5 freqüències)
vis_train = np.load("vis_train.npy")          # Forma: (12 temps, 9 freqüències)
vis_val = np.load("vis_validation.npy")       # Forma: (12 temps, 5 freqüències)

# Unifiquem totes les mesures per a cada instant (total de 14 canals)
nu_obs = np.concatenate([nu_train, nu_val], axis=1)    # Forma: (12, 14)
vis_obs = np.concatenate([vis_train, vis_val], axis=1) # Forma: (12, 14)

n_t = len(times)            # 12 fotogrames
n_pix = len(x_grid)         # 64 píxels
n_canals = vis_obs.shape[1] # 14 canals de freqüència
xi = x_grid                 # Coordenades espacials x_p
m_val = 1.0 / n_pix         # Prior uniforme pla

# Nombre total de dades reals (part real i part imaginària): 2 * (12 * 14) = 336
n_data = 2 * vis_obs.size

print(f"[OK] Dades carregades correctament:")
print(f" - Nombre de temps (n_t): {n_t}")
print(f" - Nombre de píxels (n_pix): {n_pix}")
print(f" - Nombre de canals de freqüència (n_canals): {n_canals}")
print(f" - Nombre total de dades reals (N_dades): {n_data}")

# ==============================================================================
# 2. PRECÀLCUL DE LA MATRIU DE FOURIER
# ==============================================================================
# Matriu 3D de Fourier: F(t, k, p) = exp(-2 * pi * i * nu[t, k] * x[p])
# Calculada mitjançant bucles explícits per a màxima claredat:
F_matrix = np.zeros((n_t, n_canals, n_pix), dtype=complex)

for t in range(n_t):
    for k in range(n_canals):
        freq = nu_obs[t, k]
        for p in range(n_pix):
            posicio = xi[p]
            angle = -2.0 * np.pi * freq * posicio
            F_matrix[t, k, p] = np.exp(1j * angle)


# ==============================================================================
# 3. FUNCIONS MATEMÀTIQUES DEFINIDES AMB BUCLES EXPLÍCITS
# ==============================================================================

def calcular_visibilitats_model(movie):
    """
    Calcula les visibilitats predites pel model mitjançant la Transformada de Fourier discreta 1D.

    Fórmula matemàtica:
        V_mod(t, k) = sum_{p=0}^{N_pix - 1} I(t, p) * exp(-2 * pi * i * nu(t, k) * x(p))
                    = sum_{p=0}^{N_pix - 1} I(t, p) * F_matrix(t, k, p)

    Paràmetres:
        movie: Matriu 2D de mida (n_t, n_pix) amb les intensitats I(t, p).

    Retorna:
        v_mod: Matriu 2D complexa de mida (n_t, n_canals) amb les visibilitats.
    """
    v_mod = np.zeros((n_t, n_canals), dtype=complex)

    for t in range(n_t):
        for k in range(n_canals):
            suma_fourier = 0.0 + 0.0j
            for p in range(n_pix):
                intensitat = movie[t, p]
                factor_exponencial = F_matrix[t, k, p]
                suma_fourier += intensitat * factor_exponencial
            v_mod[t, k] = suma_fourier

    return v_mod


def calcular_chi2_total(v_mod, v_obs):
    """
    Calcula el cost de fidelitat a les dades mitjançant l'estadístic Khi-quadrat (chi2).

    Fórmula matemàtica:
        chi2 = sum_{t=0}^{N_t - 1} sum_{k=0}^{N_canals - 1} |V_obs(t, k) - V_mod(t, k)|^2 / sigma^2

    On el mòdul al quadrat d'un nombre complex z = a + b*i és:
        |z|^2 = Re(z)^2 + Im(z)^2

    Paràmetres:
        v_mod: Visibilitats predites pel model (n_t, n_canals).
        v_obs: Visibilitats observades (n_t, n_canals).

    Retorna:
        suma_chi2: Valor escalar del chi2 total.
    """
    suma_chi2 = 0.0
    variancia = noise_std ** 2

    for t in range(n_t):
        for k in range(n_canals):
            dif = v_obs[t, k] - v_mod[t, k]
            residu_quadrat = (dif.real ** 2) + (dif.imag ** 2)
            suma_chi2 += residu_quadrat / variancia

    return suma_chi2


def calcular_gradient_chi2(v_mod, v_obs):
    """
    Calcula la derivada analítica del chi2 respecte a la brillantor de cada píxel I(t, p).

    Fórmula matemàtica:
        d(chi2) / d(I[t, p]) = - (2 / sigma^2) * sum_{k=0}^{N_canals - 1} Re[ (V_obs(t, k) - V_mod(t, k)) * conj(F_matrix(t, k, p)) ]

    On:
        - conj(F_matrix[t, k, p]) és el conjugat complex: exp(+2 * pi * i * nu * x)
        - Re[...] indica agafar la part real del producte complex.

    Paràmetres:
        v_mod: Visibilitats predites pel model (n_t, n_canals).
        v_obs: Visibilitats observades (n_t, n_canals).

    Retorna:
        grad_chi2: Matriu 2D de mida (n_t, n_pix) amb les derivades parcials.
    """
    grad_chi2 = np.zeros((n_t, n_pix))
    factor = 2.0 / (noise_std ** 2)

    for t in range(n_t):
        for p in range(n_pix):
            suma_canals = 0.0
            for k in range(n_canals):
                dif = v_obs[t, k] - v_mod[t, k]
                f_conj = np.conj(F_matrix[t, k, p])
                producte_complex = dif * f_conj
                suma_canals += producte_complex.real

            grad_chi2[t, p] = -factor * suma_canals

    return grad_chi2


def calcular_regularitzador_espacial_ngmem(movie):
    """
    Calcula el regularitzador espacial d'entropia ngMeM (Non-Global Maximum Entropy Method).

    Fórmula matemàtica:
        S_ngmem = sum_{t=0}^{N_t - 1} sum_{p=0}^{N_pix - 1} [ I(t, p) * ln(I(t, p) / m) - I(t, p) + m ]

    On:
        - I(t, p) és la intensitat del píxel.
        - m és el model a priori pla de referència uniforme (m_val = 1 / n_pix).
        - La funció és sempre positiva o zero, i assoleix el mínim quan I(t, p) = m.

    Paràmetres:
        movie: Matriu 2D d'intensitats (n_t, n_pix).

    Retorna:
        suma_ngmem: Valor escalar del regularitzador espacial.
    """
    suma_ngmem = 0.0

    for t in range(n_t):
        for p in range(n_pix):
            val = movie[t, p]
            terme = val * np.log(val / m_val) - val + m_val
            suma_ngmem += terme

    return suma_ngmem


def calcular_gradient_espacial_ngmem(movie):
    """
    Calcula la derivada analítica de ngMeM respecte a cada píxel I(t, p).

    Fórmula matemàtica:
        d(S_ngmem) / d(I[t, p]) = ln(I(t, p) / m)

    Derivació:
        d/dI [ I * ln(I/m) - I + m ] = ln(I/m) + I * (1/I) - 1 = ln(I/m)

    Paràmetres:
        movie: Matriu 2D d'intensitats (n_t, n_pix).

    Retorna:
        grad_ngmem: Matriu 2D de mida (n_t, n_pix) amb les derivades.
    """
    grad_ngmem = np.zeros((n_t, n_pix))

    for t in range(n_t):
        for p in range(n_pix):
            val = movie[t, p]
            grad_ngmem[t, p] = np.log(val / m_val)

    return grad_ngmem


def calcular_regularitzador_temporal_L2(movie):
    """
    Calcula el regularitzador de continuïtat temporal L2 entre tots els fotogrames.

    Fórmula matemàtica:
        R_temporal = 0.5 * sum_{t=0}^{N_t - 1} sum_{s=0}^{N_t - 1} sum_{p=0}^{N_pix - 1} (I(t, p) - I(s, p))^2

    On:
        - Es penalitzen les variacions temporals d'un mateix píxel p entre els instants t i s.
        - El factor 0.5 compensa la duplicitat de sumar tant (t, s) com (s, t).

    Paràmetres:
        movie: Matriu 2D d'intensitats (n_t, n_pix).

    Retorna:
        cost_temporal: Valor escalar del regularitzador temporal.
    """
    suma_diferencies_quadrat = 0.0

    for t in range(n_t):
        for s in range(n_t):
            for p in range(n_pix):
                diferencia = movie[t, p] - movie[s, p]
                suma_diferencies_quadrat += diferencia ** 2

    return 0.5 * suma_diferencies_quadrat


def calcular_gradient_temporal_L2(movie):
    """
    Calcula la derivada analítica del regularitzador temporal L2 respecte a I(t, p).

    Fórmula matemàtica:
        d(R_temporal) / d(I[t, p]) = 2 * sum_{s=0}^{N_t - 1} (I(t, p) - I(s, p))

    Derivació:
        Com que I(t, p) apareix tant quan l'índex és t com quan és s,
        la suma de les dues derivades parcials multiplicada per 0.5 dóna exactament:
        0.5 * [ 2 * (I(t,p) - I(s,p)) + 2 * (I(t,p) - I(s,p)) ] = 2 * (I(t,p) - I(s,p)).

    Paràmetres:
        movie: Matriu 2D d'intensitats (n_t, n_pix).

    Retorna:
        grad_temporal: Matriu 2D de mida (n_t, n_pix).
    """
    grad_temporal = np.zeros((n_t, n_pix))

    for t in range(n_t):
        for p in range(n_pix):
            suma_s = 0.0
            for s in range(n_t):
                suma_s += (movie[t, p] - movie[s, p])
            grad_temporal[t, p] = 2.0 * suma_s

    return grad_temporal


# ==============================================================================
# 4. FUNCIÓ OBJECTIU I RESOLUCIÓ DE LA RECONSTRUCCIÓ
# ==============================================================================

def funcio_objectiu(vector_pla, lmbda, mu):
    """
    Avalua la funció de cost total i el seu vector de gradients (2 pesos).

    Fórmula matemàtica del cost total:
        J(I) = chi2(I) + lambda * S_ngmem(I) + mu * R_temporal(I)

    Fórmula matemàtica del gradient total:
        grad_J(I) = grad_chi2(I) + lambda * grad_ngmem(I) + mu * grad_temporal(I)

    Paràmetres:
        vector_pla: Vector 1D de longitud (n_t * n_pix) amb les variables d'optimització.
        lmbda: Pes del terme espacial d'entropia ngMeM.
        mu: Pes del terme de continuïtat temporal L2.

    Retorna:
        cost_total: Valor escalar del cost.
        grad_total_pla: Vector 1D amb el gradient total.
    """
    movie = vector_pla.reshape((n_t, n_pix))

    # 1. Visibilitats del model
    v_mod = calcular_visibilitats_model(movie)

    # 2. Termes de cost
    terme_dades = calcular_chi2_total(v_mod, vis_obs)
    terme_espai = calcular_regularitzador_espacial_ngmem(movie)
    terme_temps = calcular_regularitzador_temporal_L2(movie)

    cost_total = terme_dades + (lmbda * terme_espai) + (mu * terme_temps)

    # 3. Gradients individuals
    grad_dades = calcular_gradient_chi2(v_mod, vis_obs)
    grad_espai = calcular_gradient_espacial_ngmem(movie)
    grad_temps = calcular_gradient_temporal_L2(movie)

    grad_total = grad_dades + (lmbda * grad_espai) + (mu * grad_temps)

    return cost_total, grad_total.ravel()


limits = [(1e-8, None)] * (n_t * n_pix)

def resoldre_reconstruccio(lmbda, mu, x_init=None, maxiter=300):
    """
    Optimitza la pel·lícula d'imatges donats els pesos (lambda, mu).
    Retorna un diccionari complet amb la solució, mètriques i temps d'execució.
    """
    t0 = time.time()
    if x_init is None:
        x0 = np.full(n_t * n_pix, m_val)
    else:
        x0 = x_init.copy()

    opt = minimize(
        funcio_objectiu,
        x0,
        args=(lmbda, mu),
        jac=True,
        method="L-BFGS-B",
        bounds=limits,
        options={"maxiter": maxiter}
    )
    movie = opt.x.reshape(n_t, n_pix)
    elapsed = time.time() - t0

    v_mod = calcular_visibilitats_model(movie)
    chi2_val = calcular_chi2_total(v_mod, vis_obs)
    chi2_red = chi2_val / n_data
    ngmem_val = calcular_regularitzador_espacial_ngmem(movie)
    l2t_val = calcular_regularitzador_temporal_L2(movie)

    return {
        "movie": movie,
        "flat": opt.x,
        "chi2_obs": float(chi2_val),
        "chi2_red": float(chi2_red),
        "ngmem": float(ngmem_val),
        "l2_temporal": float(l2t_val),
        "time_seconds": round(elapsed, 2),
        "converged": bool(opt.success)
    }


# ==============================================================================
# 5. TEORIA DE JOCS COOPERATIUS: VALORS DE SHAPLEY PER A 2 JUGADORS
# ==============================================================================
# Plantegem la selecció dels hiperparàmetres com un Joc Cooperatiu amb 2 jugadors:
#   - Jugador 1: lambda (promou la simplicitat/entropia espacial ngMeM)
#   - Jugador 2: mu     (promou la continuïtat temporal L2)
#
# Cada coalició S d'hiperparàmetres coopera per reduir les penalitzacions respecte
# a l'estat base sense regularització, respectant el Principi de Discrepància (chi2_red ~ 1).

print("\n" + "=" * 70)
print(" DETERMINACIÓ DELS PESOS MITJANÇANT ELS VALORS DE SHAPLEY (2 PESOS)")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 Coalició buida S = {} (sense regularitzadors: lambda = 0, mu = 0)
# ------------------------------------------------------------------------------
print("\n[1/4] Avaluant coalició buida {} (lambda=0, mu=0)...")
res_empty = resoldre_reconstruccio(0.0, 0.0)
s0 = res_empty["ngmem"]
t0 = res_empty["l2_temporal"]
print(f" -> Coalició buida: chi2_red={res_empty['chi2_red']:.3f} | ngMeM={s0:.3f} | L2_t={t0:.3f}")

# ------------------------------------------------------------------------------
# 5.2 Coalició individual {lambda}: calibració de lambda* tal que chi2_red ~ 1 amb mu = 0
# ------------------------------------------------------------------------------
def calibrar_lambda_sol():
    """Cerca dicotòmica (bisecció en escala logarítmica) de lambda* tal que chi2_red ~ 1.0"""
    low, high = 1.0, 500.0
    x_prev = res_empty["flat"]
    for _ in range(12):
        mid = np.sqrt(low * high)
        sol = resoldre_reconstruccio(mid, 0.0, x_init=x_prev)
        x_prev = sol["flat"]
        if sol["chi2_red"] < 1.0:
            low = mid
        else:
            high = mid
    opt_l = np.sqrt(low * high)
    return opt_l, resoldre_reconstruccio(opt_l, 0.0, x_init=x_prev)

print("\n[2/4] Calibrant coalició {lambda} (ngMeM sol fins a chi2_red ~ 1)...")
l_star, res_l = calibrar_lambda_sol()
s_l = res_l["ngmem"]
t_l = res_l["l2_temporal"]
print(f" -> lambda* = {l_star:.3f} | chi2_red={res_l['chi2_red']:.3f} | ngMeM={s_l:.3f} | L2_t={t_l:.3f}")

# ------------------------------------------------------------------------------
# 5.3 Coalició individual {mu}: calibració de mu* tal que chi2_red ~ 1 amb lambda = 0
# ------------------------------------------------------------------------------
def calibrar_mu_sol():
    """Cerca dicotòmica de mu* tal que chi2_red ~ 1.0"""
    low, high = 100.0, 20000.0
    x_prev = res_empty["flat"]
    for _ in range(12):
        mid = np.sqrt(low * high)
        sol = resoldre_reconstruccio(0.0, mid, x_init=x_prev)
        x_prev = sol["flat"]
        if sol["chi2_red"] < 1.0:
            low = mid
        else:
            high = mid
    opt_m = np.sqrt(low * high)
    return opt_m, resoldre_reconstruccio(0.0, opt_m, x_init=x_prev)

print("\n[3/4] Calibrant coalició {mu} (L2 temporal sol fins a chi2_red ~ 1)...")
m_star, res_m = calibrar_mu_sol()
s_m = res_m["ngmem"]
t_m = res_m["l2_temporal"]
print(f" -> mu* = {m_star:.3f} | chi2_red={res_m['chi2_red']:.3f} | ngMeM={s_m:.3f} | L2_t={t_m:.3f}")

# ------------------------------------------------------------------------------
# 5.4 Gran coalició {lambda, mu}: cooperació simultània
# ------------------------------------------------------------------------------
def calibrar_gran_coalicio(l_base, m_base):
    """Calibra l'escala de cooperació perquè la gran coalició compleixi chi2_red ~ 1"""
    low_c, high_c = 0.2, 3.0
    x_prev = res_empty["flat"]
    for _ in range(10):
        mid_c = np.sqrt(low_c * high_c)
        sol = resoldre_reconstruccio(mid_c * 0.5 * l_base, mid_c * 0.5 * m_base, x_init=x_prev)
        x_prev = sol["flat"]
        if sol["chi2_red"] < 1.0:
            low_c = mid_c
        else:
            high_c = mid_c
    c_opt = np.sqrt(low_c * high_c)
    return resoldre_reconstruccio(c_opt * 0.5 * l_base, c_opt * 0.5 * m_base, x_init=x_prev)

print("\n[4/4] Avaluant gran coalició {lambda, mu}...")
res_lm = calibrar_gran_coalicio(l_star, m_star)
s_lm = res_lm["ngmem"]
t_lm = res_lm["l2_temporal"]
print(f" -> Gran coalició: chi2_red={res_lm['chi2_red']:.3f} | ngMeM={s_lm:.3f} | L2_t={t_lm:.3f}")

# ------------------------------------------------------------------------------
# 5.5 Funció característica del joc v(S)
# ------------------------------------------------------------------------------
# Definim la utilitat normalitzada obtinguda per cada regularitzador en cooperació:
#   u_lambda = (s0 - s_lm) / (s0 - s_l)
#   u_mu     = (t0 - t_lm) / (t0 - t_m)
v_empty = 0.0
v_lambda = 1.0
v_mu = 1.0

u_lambda_coop = (s0 - s_lm) / (s0 - s_l) if abs(s0 - s_l) > 1e-12 else 1.0
u_mu_coop = (t0 - t_lm) / (t0 - t_m) if abs(t0 - t_m) > 1e-12 else 1.0
v_lambda_mu = u_lambda_coop + u_mu_coop

print("\n--- Valors de la funció característica v(S) ---")
print(f" v(Ø)            = {v_empty:.4f}")
print(f" v({{lambda}})     = {v_lambda:.4f}")
print(f" v({{mu}})         = {v_mu:.4f}")
print(f" v({{lambda, mu}}) = {v_lambda_mu:.4f} (u_lambda={u_lambda_coop:.3f}, u_mu={u_mu_coop:.3f})")

# ------------------------------------------------------------------------------
# 5.6 Càlcul dels Valors de Shapley per a 2 jugadors
# ------------------------------------------------------------------------------
# Fórmula teòrica de Shapley:
#   phi_i = sum_{S subseteq N \ {i}} [ |S|! * (|N| - |S| - 1)! / |N|! ] * [ v(S U {i}) - v(S) ]
#
# Per a 2 jugadors (|N| = 2):
#   phi_lambda = 0.5 * (v({lambda}) - v(Ø)) + 0.5 * (v({lambda, mu}) - v({mu}))
#   phi_mu     = 0.5 * (v({mu}) - v(Ø))     + 0.5 * (v({lambda, mu}) - v({lambda}))
phi_lambda = 0.5 * (v_lambda - v_empty) + 0.5 * (v_lambda_mu - v_mu)
phi_mu = 0.5 * (v_mu - v_empty) + 0.5 * (v_lambda_mu - v_lambda)

print("\n--- Valors de Shapley Calculats ---")
print(f" phi(lambda) = {phi_lambda:.4f}")
print(f" phi(mu)     = {phi_mu:.4f}")

# Pesos relatius normalitzats:
#   w_i = phi_i / (phi_lambda + phi_mu)
suma_phi = phi_lambda + phi_mu
w_lambda = max(phi_lambda / suma_phi, 1e-4)
w_mu = max(phi_mu / suma_phi, 1e-4)

# ------------------------------------------------------------------------------
# 5.7 Calibració de l'escala global c per complir chi2_red ~ 1
# ------------------------------------------------------------------------------
low_c, high_c = 0.2, 3.0
x_prev = res_lm["flat"]
for _ in range(10):
    mid_c = np.sqrt(low_c * high_c)
    sol = resoldre_reconstruccio(mid_c * w_lambda * l_star, mid_c * w_mu * m_star, x_init=x_prev)
    x_prev = sol["flat"]
    if sol["chi2_red"] < 1.0:
        low_c = mid_c
    else:
        high_c = mid_c

c_final = np.sqrt(low_c * high_c)
best_lambda = float(c_final * w_lambda * l_star)
best_mu = float(c_final * w_mu * m_star)

res_final = resoldre_reconstruccio(best_lambda, best_mu, x_init=x_prev)
best_movie = res_final["movie"]

print("\n" + "=" * 70)
print(f" PESOS ÒPTIMS PER SHAPLEY (2 PESOS):")
print(f"   lambda_Shapley = {best_lambda:.4f}")
print(f"   mu_Shapley     = {best_mu:.4f}")
print(f"   chi2_red final = {res_final['chi2_red']:.4f}")
print("=" * 70)

# ==============================================================================
# 6. TAULA DE RESULTATS
# ==============================================================================
df_shapley = pd.DataFrame([
    {"coalicio": "Buida Ø", "lambda": 0.0, "mu": 0.0, "v_S": v_empty, "chi2_red": res_empty["chi2_red"], "ngmem": s0, "l2_temporal": t0},
    {"coalicio": "{lambda}", "lambda": l_star, "mu": 0.0, "v_S": v_lambda, "chi2_red": res_l["chi2_red"], "ngmem": s_l, "l2_temporal": t_l},
    {"coalicio": "{mu}", "lambda": 0.0, "mu": m_star, "v_S": v_mu, "chi2_red": res_m["chi2_red"], "ngmem": s_m, "l2_temporal": t_m},
    {"coalicio": "{lambda, mu} Shapley", "lambda": best_lambda, "mu": best_mu, "v_S": v_lambda_mu, "chi2_red": res_final["chi2_red"], "ngmem": res_final["ngmem"], "l2_temporal": res_final["l2_temporal"]}
])
df_shapley.to_csv("resultats_shapley_lambda_mu.csv", index=False)
print("\n[OK] Taula guardada correctament a 'resultats_shapley_lambda_mu.csv'.")


# ==============================================================================
# 7. GENERACIÓ DELS GRÀFICS FINALS
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Diagrama de barres dels Valors de Shapley
categories = [r"$\phi_\lambda$ (ngMeM)", r"$\phi_\mu$ (L2 temporal)"]
shapley_vals = [phi_lambda, phi_mu]
bars = axes[0].bar(categories, shapley_vals, color=["#2b7bba", "#e67300"], width=0.45, edgecolor="black", alpha=0.85)

for bar in bars:
    h = bar.get_height()
    axes[0].annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

axes[0].set_ylim(0, max(shapley_vals) * 1.25)
axes[0].set_title(rf"Valors de Shapley ($\lambda$={best_lambda:.2f}, $\mu$={best_mu:.2f})")
axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

# (b) Pel·lícula reconstruïda al llarg dels 12 instants de temps
colors = plt.cm.viridis(np.linspace(0, 1, n_t))
for i, t in enumerate(times):
    axes[1].plot(xi, best_movie[i], color=colors[i], lw=1.8, label=f"t={t:.2f}")

axes[1].set_title("Pel·lícula Reconstruïda (Shapley 2 pesos)")
axes[1].set_xlabel(r"Posició $\xi$")
axes[1].set_ylabel("Intensitat I(xi, t)")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(fontsize=7, ncol=2)

# (c) Cinemàtica de la font (Centroide calculat amb bucle explícit)
#   centroide(t) = sum_{p} [ I(t, p) * x[p] ] / sum_{p} [ I(t, p) ]
centroids = np.zeros(n_t)
for t in range(n_t):
    suma_posicio_ponderada = 0.0
    suma_intensitat_total = 0.0
    for p in range(n_pix):
        suma_posicio_ponderada += best_movie[t, p] * xi[p]
        suma_intensitat_total += best_movie[t, p]
    centroids[t] = suma_posicio_ponderada / suma_intensitat_total

axes[2].plot(centroids, times, "o-", color="crimson", lw=2)
axes[2].set_title("Cinemàtica de la font (Centroide)")
axes[2].set_xlabel("Posició centroide")
axes[2].set_ylabel("Temps (t)")
axes[2].set_xlim(-1, 1)
axes[2].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("shapley_lambda_mu_pellicula.png", dpi=150)
print("[OK] Gràfic guardat correctament a 'shapley_lambda_mu_pellicula.png'.")
plt.show()
