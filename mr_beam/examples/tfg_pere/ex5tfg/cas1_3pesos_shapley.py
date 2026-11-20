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
# 4. FUNCIÓ OBJECTIU I RESOLUCIÓ DE LA RECONSTRUCCIÓ (3 PESOS)
# ==============================================================================

def funcio_objectiu(vector_pla, delta, lmbda, mu):
    """
    Avalua la funció de cost total i el seu vector de gradients (3 pesos).

    Fórmula matemàtica del cost total:
        J(I) = delta * chi2(I) + lambda * S_ngmem(I) + mu * R_temporal(I)

    Fórmula matemàtica del gradient total:
        grad_J(I) = delta * grad_chi2(I) + lambda * grad_ngmem(I) + mu * grad_temporal(I)

    Paràmetres:
        vector_pla: Vector 1D de longitud (n_t * n_pix) amb les variables d'optimització.
        delta: Pes del terme de dades (chi2).
        lmbda: Pes del terme espacial d'entropia ngMeM.
        mu: Pes del terme de continuïtat temporal L2.

    Retorna:
        cost_total: Valor escalar del cost.
        grad_total_pla: Vector 1D amb el gradient total.
    """
    movie = vector_pla.reshape((n_t, n_pix))

    v_mod = calcular_visibilitats_model(movie)

    terme_dades = calcular_chi2_total(v_mod, vis_obs)
    terme_espai = calcular_regularitzador_espacial_ngmem(movie)
    terme_temps = calcular_regularitzador_temporal_L2(movie)

    cost_total = (delta * terme_dades) + (lmbda * terme_espai) + (mu * terme_temps)

    grad_dades = calcular_gradient_chi2(v_mod, vis_obs)
    grad_espai = calcular_gradient_espacial_ngmem(movie)
    grad_temps = calcular_gradient_temporal_L2(movie)

    grad_total = (delta * grad_dades) + (lmbda * grad_espai) + (mu * grad_temps)

    return cost_total, grad_total.ravel()


limits = [(1e-8, None)] * (n_t * n_pix)

def resoldre_reconstruccio(delta, lmbda, mu, x_init=None, maxiter=300):
    """
    Optimitza la pel·lícula d'imatges donats els 3 pesos (delta, lambda, mu).
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
        args=(delta, lmbda, mu),
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
# 5. TEORIA DE JOCS COOPERATIUS: VALORS DE SHAPLEY PER A 3 JUGADORS
# ==============================================================================
# Formulació com a joc cooperatiu amb 3 jugadors:
#   - Jugador 1 (delta):  Ajust a les dades observades (minimitzar chi2)
#   - Jugador 2 (lambda): Suavitat espacial / Entropia (minimitzar ngMeM)
#   - Jugador 3 (mu):     Continuïtat temporal (minimitzar L2 temporal)

print("\n" + "=" * 70)
print(" DETERMINACIÓ DELS PESOS MITJANÇANT ELS VALORS DE SHAPLEY (3 PESOS)")
print("=" * 70)

# ------------------------------------------------------------------------------
# 5.1 Referència base: Prior Pla (màxima entropia)
# ------------------------------------------------------------------------------
movie_pla = np.full((n_t, n_pix), m_val)
v_mod_pla = calcular_visibilitats_model(movie_pla)
chi2_base = float(calcular_chi2_total(v_mod_pla, vis_obs))
chi2_red_base = chi2_base / n_data
s0 = float(calcular_regularitzador_espacial_ngmem(movie_pla))
t0 = float(calcular_regularitzador_temporal_L2(movie_pla))

print(f"\n[Estat Base / Prior Pla Uniforme]")
print(f" -> chi2_red = {chi2_red_base:.2f} | ngMeM = {s0:.2f} | L2_t = {t0:.2f}")

# ------------------------------------------------------------------------------
# 5.2 Coalicions individuals (un sol jugador actiu):
# ------------------------------------------------------------------------------
# (a) {delta}: només dades actives (delta=1.0, lambda=0, mu=0)
print("\n[1/7] Avaluant coalició {delta} (només dades)...")
res_delta = resoldre_reconstruccio(1.0, 0.0, 0.0)
chi2_min_dades = res_delta["chi2_obs"]
s_delta = res_delta["ngmem"]
t_delta = res_delta["l2_temporal"]
print(f" -> chi2_red={res_delta['chi2_red']:.3f} | ngMeM={s_delta:.2f} | L2_t={t_delta:.2f}")

# (b) {lambda}: regularització espacial amb dades
def calibrar_lambda_sol():
    low, high = 1.0, 500.0
    x_prev = res_delta["flat"]
    for _ in range(12):
        mid = np.sqrt(low * high)
        sol = resoldre_reconstruccio(1.0, mid, 0.0, x_init=x_prev)
        x_prev = sol["flat"]
        if sol["chi2_red"] < 1.0:
            low = mid
        else:
            high = mid
    opt_l = np.sqrt(low * high)
    return opt_l, resoldre_reconstruccio(1.0, opt_l, 0.0, x_init=x_prev)

print("\n[2/7] Calibrant coalició {lambda} (ngMeM fins a chi2_red ~ 1)...")
l_star, res_l = calibrar_lambda_sol()
s_star = res_l["ngmem"]
print(f" -> lambda*={l_star:.2f} | chi2_red={res_l['chi2_red']:.3f} | ngMeM={s_star:.2f}")

# (c) {mu}: regularització temporal amb dades
def calibrar_mu_sol():
    low, high = 100.0, 20000.0
    x_prev = res_delta["flat"]
    for _ in range(12):
        mid = np.sqrt(low * high)
        sol = resoldre_reconstruccio(1.0, 0.0, mid, x_init=x_prev)
        x_prev = sol["flat"]
        if sol["chi2_red"] < 1.0:
            low = mid
        else:
            high = mid
    opt_m = np.sqrt(low * high)
    return opt_m, resoldre_reconstruccio(1.0, 0.0, opt_m, x_init=x_prev)

print("\n[3/7] Calibrant coalició {mu} (L2 temporal fins a chi2_red ~ 1)...")
m_star, res_m = calibrar_mu_sol()
t_star = res_m["l2_temporal"]
print(f" -> mu*={m_star:.2f} | chi2_red={res_m['chi2_red']:.3f} | L2_t={t_star:.2f}")

# ------------------------------------------------------------------------------
# 5.3 Coalicions de 2 jugadors:
# ------------------------------------------------------------------------------
print("\n[4/7] Avaluant coalició {delta, lambda}...")
res_dl = res_l
print(f" -> chi2_red={res_dl['chi2_red']:.3f} | ngMeM={res_dl['ngmem']:.2f}")

print("\n[5/7] Avaluant coalició {delta, mu}...")
res_dm = res_m
print(f" -> chi2_red={res_dm['chi2_red']:.3f} | L2_t={res_dm['l2_temporal']:.2f}")

print("\n[6/7] Avaluant coalició {lambda, mu}...")
res_lm = resoldre_reconstruccio(0.1, l_star, m_star, x_init=res_delta["flat"])
print(f" -> chi2_red={res_lm['chi2_red']:.3f} | ngMeM={res_lm['ngmem']:.2f} | L2_t={res_lm['l2_temporal']:.2f}")

# ------------------------------------------------------------------------------
# 5.4 Gran Coalició {delta, lambda, mu}:
# ------------------------------------------------------------------------------
def calibrar_gran_coalicio():
    low_c, high_c = 0.2, 3.0
    x_prev = res_delta["flat"]
    for _ in range(10):
        mid_c = np.sqrt(low_c * high_c)
        sol = resoldre_reconstruccio(1.0, mid_c * 0.5 * l_star, mid_c * 0.5 * m_star, x_init=x_prev)
        x_prev = sol["flat"]
        if sol["chi2_red"] < 1.0:
            low_c = mid_c
        else:
            high_c = mid_c
    c_opt = np.sqrt(low_c * high_c)
    return resoldre_reconstruccio(1.0, c_opt * 0.5 * l_star, c_opt * 0.5 * m_star, x_init=x_prev)

print("\n[7/7] Avaluant Gran Coalició {delta, lambda, mu}...")
res_all = calibrar_gran_coalicio()
print(f" -> Gran coalició: chi2_red={res_all['chi2_red']:.3f} | ngMeM={res_all['ngmem']:.2f} | L2_t={res_all['l2_temporal']:.2f}")

# ------------------------------------------------------------------------------
# 5.5 Funció característica v(S) per a cadascuna de les 8 coalicions
# ------------------------------------------------------------------------------
# Definim la utilitat de cada objectiu com la fracció normalitzada de reducció del cost:
def utilitat_delta(sol):
    """Fracció de millora del chi2 respecte a l'estat base pla"""
    return max(0.0, min(1.0, (chi2_base - sol["chi2_obs"]) / (chi2_base - chi2_min_dades)))

def utilitat_lambda(sol):
    """Fracció de millora d'entropia ngMeM respecte a l'estat base"""
    denom = s0 - s_star if abs(s0 - s_star) > 1e-12 else 1.0
    return max(0.0, min(1.0, (s0 - sol["ngmem"]) / denom))

def utilitat_mu(sol):
    """Fracció de millora de continuïtat temporal L2 respecte a l'estat base"""
    denom = t0 - t_star if abs(t0 - t_star) > 1e-12 else 1.0
    return max(0.0, min(1.0, (t0 - sol["l2_temporal"]) / denom))

v = {}
v[()] = 0.0
v[("delta",)] = utilitat_delta(res_delta)
v[("lambda",)] = utilitat_lambda(res_l)
v[("mu",)] = utilitat_mu(res_m)

v[("delta", "lambda")] = utilitat_delta(res_dl) + utilitat_lambda(res_dl)
v[("delta", "mu")] = utilitat_delta(res_dm) + utilitat_mu(res_dm)
v[("lambda", "mu")] = utilitat_lambda(res_lm) + utilitat_mu(res_lm)

v[("delta", "lambda", "mu")] = utilitat_delta(res_all) + utilitat_lambda(res_all) + utilitat_mu(res_all)

print("\n--- Valors de la funció característica v(S) ---")
for k, val in v.items():
    nom = "{" + ", ".join(k) + "}" if k else "Ø"
    print(f" v({nom:<20}) = {val:.4f}")

# ------------------------------------------------------------------------------
# 5.6 Fórmula de Shapley per a 3 jugadors
# ------------------------------------------------------------------------------
# Per a un conjunt de jugadors N = {delta, lambda, mu} amb |N| = 3:
#   phi_i = sum_{S subseteq N \ {i}} [ |S|! * (|N| - |S| - 1)! / |N|! ] * [ v(S U {i}) - v(S) ]
#
# Els pesos de cada coalició segons la seva mida |S| són:
#   - Mida |S| = 0 (coalició buida): 0! * 2! / 3! = 2 / 6 = 1 / 3
#   - Mida |S| = 1 (coalicions d'1 jugador): 1! * 1! / 3! = 1 / 6
#   - Mida |S| = 2 (coalicions de 2 jugadors): 2! * 0! / 3! = 2 / 6 = 1 / 3
#
# Per tant:
phi_delta = (
    (1/3) * (v[("delta",)] - v[()]) +
    (1/6) * ((v[("delta", "lambda")] - v[("lambda",)]) + (v[("delta", "mu")] - v[("mu",)])) +
    (1/3) * (v[("delta", "lambda", "mu")] - v[("lambda", "mu")])
)

phi_lambda = (
    (1/3) * (v[("lambda",)] - v[()]) +
    (1/6) * ((v[("delta", "lambda")] - v[("delta",)]) + (v[("lambda", "mu")] - v[("mu",)])) +
    (1/3) * (v[("delta", "lambda", "mu")] - v[("delta", "mu")])
)

phi_mu = (
    (1/3) * (v[("mu",)] - v[()]) +
    (1/6) * ((v[("delta", "mu")] - v[("delta",)]) + (v[("lambda", "mu")] - v[("lambda",)])) +
    (1/3) * (v[("delta", "lambda", "mu")] - v[("delta", "lambda")])
)

print("\n--- Valors de Shapley Calculats ---")
print(f" phi(delta)  = {phi_delta:.4f}")
print(f" phi(lambda) = {phi_lambda:.4f}")
print(f" phi(mu)     = {phi_mu:.4f}")
suma_phi = phi_delta + phi_lambda + phi_mu
print(f" Suma phi    = {suma_phi:.4f} (v(N) = {v[('delta', 'lambda', 'mu')]:.4f})")

# ------------------------------------------------------------------------------
# 5.7 Pesos normalitzats i calibració de l'escala final
# ------------------------------------------------------------------------------
w_delta = max(phi_delta / suma_phi, 0.05)
w_lambda = max(phi_lambda / suma_phi, 0.05)
w_mu = max(phi_mu / suma_phi, 0.05)

# Fixem delta = 1.0 com a referència i ajustem els altres pesos en proporció:
delta_final = 1.0
lambda_base = (w_lambda / w_delta) * l_star
mu_base = (w_mu / w_delta) * m_star

# Calibrem l'escala global c mitjançant bisecció per complir chi2_red ~ 1:
low_c, high_c = 0.1, 5.0
x_prev = res_all["flat"]
for _ in range(12):
    mid_c = np.sqrt(low_c * high_c)
    sol = resoldre_reconstruccio(delta_final, mid_c * lambda_base, mid_c * mu_base, x_init=x_prev)
    x_prev = sol["flat"]
    if sol["chi2_red"] < 1.0:
        low_c = mid_c
    else:
        high_c = mid_c

c_final = np.sqrt(low_c * high_c)
best_delta = float(delta_final)
best_lambda = float(c_final * lambda_base)
best_mu = float(c_final * mu_base)

res_final = resoldre_reconstruccio(best_delta, best_lambda, best_mu, x_init=x_prev)
best_movie = res_final["movie"]

print("\n" + "=" * 70)
print(f" PESOS ÒPTIMS PER SHAPLEY (3 PESOS):")
print(f"   delta_Shapley  = {best_delta:.4f}")
print(f"   lambda_Shapley = {best_lambda:.4f}")
print(f"   mu_Shapley     = {best_mu:.4f}")
print(f"   chi2_red final = {res_final['chi2_red']:.4f}")
print("=" * 70)

# ==============================================================================
# 6. TAULA DE RESULTATS
# ==============================================================================
df_shapley = pd.DataFrame([
    {"coalicio": "Buida Ø", "delta": 0.0, "lambda": 0.0, "mu": 0.0, "v_S": v[()], "chi2_red": chi2_red_base, "ngmem": s0, "l2_temporal": t0},
    {"coalicio": "{delta}", "delta": 1.0, "lambda": 0.0, "mu": 0.0, "v_S": v[("delta",)], "chi2_red": res_delta["chi2_red"], "ngmem": s_delta, "l2_temporal": t_delta},
    {"coalicio": "{delta, lambda}", "delta": 1.0, "lambda": l_star, "mu": 0.0, "v_S": v[("delta", "lambda")], "chi2_red": res_dl["chi2_red"], "ngmem": res_dl["ngmem"], "l2_temporal": res_dl["l2_temporal"]},
    {"coalicio": "{delta, mu}", "delta": 1.0, "lambda": 0.0, "mu": m_star, "v_S": v[("delta", "mu")], "chi2_red": res_dm["chi2_red"], "ngmem": res_dm["ngmem"], "l2_temporal": res_dm["l2_temporal"]},
    {"coalicio": "{lambda, mu}", "delta": 0.0, "lambda": l_star, "mu": m_star, "v_S": v[("lambda", "mu")], "chi2_red": res_lm["chi2_red"], "ngmem": res_lm["ngmem"], "l2_temporal": res_lm["l2_temporal"]},
    {"coalicio": "{delta, lambda, mu} Shapley", "delta": best_delta, "lambda": best_lambda, "mu": best_mu, "v_S": v[("delta", "lambda", "mu")], "chi2_red": res_final["chi2_red"], "ngmem": res_final["ngmem"], "l2_temporal": res_final["l2_temporal"]}
])
df_shapley.to_csv("resultats_shapley_delta_lambda_mu.csv", index=False)
print("\n[OK] Taula guardada correctament a 'resultats_shapley_delta_lambda_mu.csv'.")


# ==============================================================================
# 7. GENERACIÓ DELS GRÀFICS FINALS
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Diagrama de barres dels Valors de Shapley
categories = [r"$\phi_\delta$ (Dades)", r"$\phi_\lambda$ (ngMeM)", r"$\phi_\mu$ (L2 t)"]
shapley_vals = [phi_delta, phi_lambda, phi_mu]
bars = axes[0].bar(categories, shapley_vals, color=["#34a853", "#2b7bba", "#e67300"], width=0.45, edgecolor="black", alpha=0.85)

for bar in bars:
    h = bar.get_height()
    axes[0].annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

axes[0].set_ylim(0, max(shapley_vals) * 1.25)
axes[0].set_title(rf"Valors de Shapley ($\delta$={best_delta:.1f}, $\lambda$={best_lambda:.2f}, $\mu$={best_mu:.2f})")
axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

# (b) Pel·lícula reconstruïda
colors = plt.cm.viridis(np.linspace(0, 1, n_t))
for i, t in enumerate(times):
    axes[1].plot(xi, best_movie[i], color=colors[i], lw=1.8, label=f"t={t:.2f}")

axes[1].set_title("Pel·lícula Reconstruïda (Shapley 3 pesos)")
axes[1].set_xlabel(r"Posició $\xi$")
axes[1].set_ylabel("Intensitat I(xi, t)")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(fontsize=7, ncol=2)

# (c) Cinemàtica del centroide calculada amb bucle explícit
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
plt.savefig("shapley_delta_lambda_mu_pellicula.png", dpi=150)
print("[OK] Gràfic guardat correctament a 'shapley_delta_lambda_mu_pellicula.png'.")
plt.show()
