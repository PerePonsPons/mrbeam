import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==============================================================================
# 1. CÀRREGA I PREPARACIÓ DE LES DADES
# ==============================================================================
# Carreguem els fitxers .npy generats prèviament:
# - x_grid: posicions dels píxels en l'espai 1D entre [-1, 1]
# - times: instants de temps en què s'han pres les observacions (12 fotogrames)
# - noise_std: desviació estàndard del soroll gaussià afegit a les dades (sigma = 0.025)
x_grid = np.load("x.npy")                     # Vector 1D de 64 píxels espacials
times = np.load("times.npy")                  # Vector 1D de 12 fotogrames temporals
noise_std = float(np.load("noise_std.npy"))   # Desviació estàndard del soroll (sigma)

# Freqüències espacials (nu) i visibilitats complexes observades (V)
nu_train = np.load("train_frequencies.npy")   # Forma: (12 temps, 9 freqüències)
nu_val = np.load("validation_frequencies.npy")# Forma: (12 temps, 5 freqüències)
vis_train = np.load("vis_train.npy")          # Forma: (12 temps, 9 freqüències)
vis_val = np.load("vis_validation.npy")       # Forma: (12 temps, 5 freqüències)

# Concatenem les dades d'entrenament i validació per tenir totes les observacions juntes:
# Això ens dóna 14 canals de freqüència observats per a cadascun dels 12 instants de temps.
nu_obs = np.concatenate([nu_train, nu_val], axis=1)    # Forma: (12, 14)
vis_obs = np.concatenate([vis_train, vis_val], axis=1) # Forma: (12, 14)

n_t = len(times)            # Nombre d'instants de temps: 12
n_pix = len(x_grid)         # Nombre de píxels espacials: 64
n_canals = vis_obs.shape[1] # Nombre de canals de freqüència: 14
xi = x_grid                 # Coordenades espacials x_p
m_val = 1.0 / n_pix         # Brillantor de referència plana (prior uniforme)

# Nombre total de mesures reals: cada visibilitat té Part Real i Part Imaginària
# Per tant, N_dades = 2 * (12 * 14) = 336
n_data = 2 * vis_obs.size

print(f"[OK] Dades carregades correctament:")
print(f" - Nombre de temps (n_t): {n_t}")
print(f" - Nombre de píxels (n_pix): {n_pix}")
print(f" - Nombre de canals de freqüència (n_canals): {n_canals}")
print(f" - Nombre total de dades reals (N_dades): {n_data}")

# ==============================================================================
# 2. PRECÀLCUL DE LA MATRIU DE FOURIER
# ==============================================================================
# La Transformada de Fourier discreta avalua l'exponencial complexa:
#   F(t, k, p) = exp(-2 * pi * i * nu[t, k] * x[p])
#
# Per fer el codi eficient i alhora completament comprensible, precalculem
# aquesta matriu 3D de mida (12, 14, 64) mitjançant bucles 'for' clars:
F_matrix = np.zeros((n_t, n_canals, n_pix), dtype=complex)

for t in range(n_t):
    for k in range(n_canals):
        freq = nu_obs[t, k]
        for p in range(n_pix):
            posicio = xi[p]
            # Fórmula d'Euler / Fourier: exp(-2 * pi * i * nu * x)
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
        movie: Matriu 2D de mida (n_t, n_pix) que conté la brillantor I(t, p) de la pel·lícula.

    Retorna:
        v_mod: Matriu 2D complexa de mida (n_t, n_canals) amb les visibilitats predites.
    """
    v_mod = np.zeros((n_t, n_canals), dtype=complex)

    # Bucle 1: recorrem cada instant de temps t
    for t in range(n_t):
        # Bucle 2: recorrem cada canal de freqüència observada k
        for k in range(n_canals):
            suma_fourier = 0.0 + 0.0j
            # Bucle 3: sumem la contribució de cadascun dels píxels p
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
        v_mod: Visibilitats predites pel model, matriu complexa (n_t, n_canals).
        v_obs: Visibilitats observades amb el telescopi, matriu complexa (n_t, n_canals).

    Retorna:
        suma_chi2: Valor escalar del chi2 total.
    """
    suma_chi2 = 0.0
    variancia = noise_std ** 2

    # Bucle 1: sumem sobre tots els instants de temps t
    for t in range(n_t):
        # Bucle 2: sumem sobre tots els canals de mesura k
        for k in range(n_canals):
            dif = v_obs[t, k] - v_mod[t, k]
            # Modul al quadrat del residu: part real al quadrat + part imaginària al quadrat
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

    # Recorrem cada instant de temps t
    for t in range(n_t):
        # Recorrem cada píxel espacial p
        for p in range(n_pix):
            suma_canals = 0.0
            # Sumem sobre tots els canals k
            for k in range(n_canals):
                dif = v_obs[t, k] - v_mod[t, k]
                f_conj = np.conj(F_matrix[t, k, p])
                producte_complex = dif * f_conj
                suma_canals += producte_complex.real

            # Assignem el gradient amb el factor multiplicatiu i el signe negatiu
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
            # Fórmula terme a terme de ngMeM: I * ln(I/m) - I + m
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
# 4. FUNCIÓ OBJECTIU TOTAL (2 PESOS: lambda, mu)
# ==============================================================================

def funcio_objectiu(vector_pla, lmbda, mu):
    """
    Avalua la funció de cost total i el seu vector de gradients per a l'optimitzador.

    Fórmula matemàtica del cost total:
        J(I) = chi2(I) + lambda * S_ngmem(I) + mu * R_temporal(I)

    Fórmula matemàtica del gradient total:
        grad_J(I) = grad_chi2(I) + lambda * grad_ngmem(I) + mu * grad_temporal(I)

    Paràmetres:
        vector_pla: Vector 1D de longitud (n_t * n_pix) amb les variables d'optimització.
        lmbda: Pes del regularitzador espacial d'entropia ngMeM.
        mu: Pes del regularitzador de continuïtat temporal L2.

    Retorna:
        cost_total: Valor numèric escalar del cost.
        grad_total_pla: Vector 1D amb el gradient total (mateixa mida que vector_pla).
    """
    # Reconstruïm la matriu 2D de la pel·lícula (12 fotogrames x 64 píxels)
    movie = vector_pla.reshape((n_t, n_pix))

    # 1. Calculem les visibilitats del model actual
    v_mod = calcular_visibilitats_model(movie)

    # 2. Avaluem cadascun dels tres termes del funcional
    terme_dades = calcular_chi2_total(v_mod, vis_obs)
    terme_espai = calcular_regularitzador_espacial_ngmem(movie)
    terme_temps = calcular_regularitzador_temporal_L2(movie)

    # Suma ponderada del cost
    cost_total = terme_dades + (lmbda * terme_espai) + (mu * terme_temps)

    # 3. Calculem els gradients individuals
    grad_dades = calcular_gradient_chi2(v_mod, vis_obs)
    grad_espai = calcular_gradient_espacial_ngmem(movie)
    grad_temps = calcular_gradient_temporal_L2(movie)

    # Suma ponderada del gradient
    grad_total = grad_dades + (lmbda * grad_espai) + (mu * grad_temps)

    # Retornem el cost escalar i el gradient aplanat a 1D
    return cost_total, grad_total.ravel()


# ==============================================================================
# 5. EXPLORACIÓ PER GRAELLA 2D (GRID SEARCH PER A lambda I mu)
# ==============================================================================

# Valors a explorar per a cada hiperparàmetre:
lambdas = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]
mus = [1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]

results = []      # Llista on guardarem els resultats de cada combinació
movies = {}       # Diccionari per guardar les pel·lícules reconstruïdes

# Punt de partida inicial: imatge completament plana amb intensitat m = 1 / 64
I_inicial = np.full(n_t * n_pix, 1.0 / n_pix)

# Fita inferior per garantir positivitat: I(t, p) >= 1e-8
limits = [(1e-8, None)] * (n_t * n_pix)

total_comb = len(lambdas) * len(mus)
print(f"\nIniciant Grid Search 2D ({len(lambdas)} x {len(mus)} = {total_comb} combinacions)...")

# Recorrem totes les combinacions de lambda i mu amb dos bucles 'for'
for lmbda in lambdas:
    for mu in mus:
        t0 = time.time()

        # Minimitzen la funció objectiu amb l'algorisme L-BFGS-B
        res = minimize(
            funcio_objectiu,
            I_inicial,
            args=(lmbda, mu),
            method="L-BFGS-B",
            jac=True,              # Indiquem que passem el gradient analític exacte
            bounds=limits,         # Assegurem que I(t, p) sigui positiu
            options={"maxiter": 100}
        )

        movie_reco = res.x.reshape((n_t, n_pix))
        elapsed = time.time() - t0

        # Calculem el chi2 reduït del resultat: chi2_red = chi2_total / N_dades
        v_mod = calcular_visibilitats_model(movie_reco)
        chi2_val = calcular_chi2_total(v_mod, vis_obs)
        chi2_red = chi2_val / n_data

        print(f" -> lam={lmbda:8.2g} | mu={mu:8.2g} | chi2_red={chi2_red:6.2f} | temps={elapsed:.2f}s")

        # Guardem les mètriques en un diccionari senzill
        results.append({
            "lambda": lmbda,
            "mu": mu,
            "chi2_total": float(chi2_val),
            "chi2_red": float(chi2_red),
            "time_seconds": round(elapsed, 2),
            "converged": bool(res.success)
        })
        movies[(lmbda, mu)] = movie_reco

        # Warm-start: la solució obtinguda és el punt de partida per a la propera iteració
        I_inicial = res.x.copy()

# Guardem els resultats a un arxiu CSV
df = pd.DataFrame(results)
df.to_csv("resultats_grid_lambda_mu.csv", index=False)
print("\n[OK] Taula guardada correctament a 'resultats_grid_lambda_mu.csv'.")


# ==============================================================================
# 6. SELECCIÓ DEL MILLOR MODEL (PRINCIPI DE DISCREPÀNCIA: chi2_red ~ 1)
# ==============================================================================
# Segons el Principi de Discrepància, el model òptim és aquell el chi2 reduït del qual
# s'aproxima més al valor teòric esperat de 1.0 (ajust correcte sense sobreajust ni subajust).

millor_diferencia = float("inf")
millor_resultat = None

# Recorrem la llista de resultats amb un bucle senzill per trobar el millor
for fila in results:
    diferencia = abs(fila["chi2_red"] - 1.0)
    if diferencia < millor_diferencia:
        millor_diferencia = diferencia
        millor_resultat = fila

best_key = (millor_resultat["lambda"], millor_resultat["mu"])
best_movie = movies[best_key]

print(f"\n=======================================================")
print(f"Millor combinació trobada (chi2_red més proper a 1.0):")
print(f" - lambda:   {millor_resultat['lambda']}")
print(f" - mu:       {millor_resultat['mu']}")
print(f" - chi2_red: {millor_resultat['chi2_red']:.3f}")
print(f"=======================================================")


# ==============================================================================
# 7. GENERACIÓ DELS GRÀFICS I CINEMÀTICA
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Perfils d'intensitat espacial per a cadascun dels 12 instants de temps
colors = plt.cm.viridis(np.linspace(0, 1, n_t))
for t in range(n_t):
    axes[0].plot(xi, best_movie[t, :], color=colors[t], lw=1.8, label=f"t = {times[t]:.2f}")

axes[0].set_title(f"Pel·lícula Òptima (lam={millor_resultat['lambda']}, mu={millor_resultat['mu']})")
axes[0].set_xlabel("Posició espacial xi")
axes[0].set_ylabel("Intensitat I(xi, t)")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(ncol=2, fontsize=8)

# (b) Cinemàtica de la font: Càlcul del centroide per a cada instant de temps
# Fórmula del centre de massa:
#   centroide(t) = sum_{p} [ I(t, p) * x[p] ] / sum_{p} [ I(t, p) ]
centroids = np.zeros(n_t)
for t in range(n_t):
    suma_posicio_ponderada = 0.0
    suma_intensitat_total = 0.0
    for p in range(n_pix):
        suma_posicio_ponderada += best_movie[t, p] * xi[p]
        suma_intensitat_total += best_movie[t, p]
    centroids[t] = suma_posicio_ponderada / suma_intensitat_total

axes[1].plot(centroids, times, "o-", color="crimson", lw=2)
axes[1].set_title("Cinemàtica de la font (Centroide vs Temps)")
axes[1].set_xlabel("Posició del centroide (xi)")
axes[1].set_ylabel("Temps (t)")
axes[1].set_xlim(-1, 1)
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("grid_2pesos_pellicula.png", dpi=150)
print("[OK] Gràfic guardat a 'grid_2pesos_pellicula.png'.")
plt.show()
