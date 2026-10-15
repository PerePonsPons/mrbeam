import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# 1. Carregar dades
# ----------------------------------------------------------------------
x_grid = np.load("x.npy")                     # (64,) punts espacials xi entre [-1, 1]
times = np.load("times.npy")                  # (12,) instants de temps t
noise_std = float(np.load("noise_std.npy"))   # sigma = 0.025

nu_train = np.load("train_frequencies.npy")   # (12, 9)
nu_val = np.load("validation_frequencies.npy")# (12, 5)
vis_train = np.load("vis_train.npy")          # (12, 9)
vis_val = np.load("vis_validation.npy")       # (12, 5)

# Unifiquem totes les mesures per a cada instant
nu_obs = np.concatenate([nu_train, nu_val], axis=1)    # (12, 14)
vis_obs = np.concatenate([vis_train, vis_val], axis=1) # (12, 14)

n_t = len(times)       # 12 fotogrames
n_pix = len(x_grid)    # 64 pixels
xi = x_grid

# Nombre total de dades reals (Re + Im) per al chi2 reduit
n_data = 2 * vis_obs.size  # 2 * (12 * 14) = 336

print(f"[OK] Dades: {n_t} temps, {n_pix} pixels, {vis_obs.shape[1]} canals (N_dades = {n_data}).")

# ----------------------------------------------------------------------
# 2. Transformada Fourier Directe
# ----------------------------------------------------------------------
def calcular_visibilitats_model(movie):
    """
    Per a cada temps t i cada canal k, calcula la suma de Fourier directa:
    V_mod = sum_p I(p) * exp(-2*pi*i * nu * xi_p)
    """
    v_mod = np.zeros_like(vis_obs, dtype=complex)
    
    for t in range(n_t):
        for k in range(vis_obs.shape[1]):
            freq = nu_obs[t, k]
            suma_fourier = 0.0 + 0.0j
            for p in range(n_pix):
                intensitat = movie[t, p]
                posicio = xi[p]
                exponencial = np.exp(-2j * np.pi * freq * posicio)
                suma_fourier += intensitat * exponencial
            v_mod[t, k] = suma_fourier
            
    return v_mod

# ----------------------------------------------------------------------
# 3. Chi-quadrat total (suma de residus sobre la variància)
# ----------------------------------------------------------------------
def calcular_chi2_total(v_mod, v_obs):
    suma = 0.0
    for t in range(n_t):
        for k in range(v_obs.shape[1]):
            diferencia = v_obs[t, k] - v_mod[t, k]
            suma += (np.abs(diferencia) ** 2) / (noise_std ** 2)
    return suma

# ----------------------------------------------------------------------
# 4. Regularitzadors
# ----------------------------------------------------------------------
def calcular_regularitzador_temporal_L2(movie):
    """
    Compara cada pixel p en TOTES les parelles de frames (t, s) amb t < s.
    """
    suma_L2 = 0.0
    for t in range(n_t):
        for s in range(t + 1, n_t):
            for p in range(n_pix):
                salt = movie[t, p] - movie[s, p]
                suma_L2 += salt ** 2
    return suma_L2

def calcular_regularitzador_espacial_ngmem(movie):
    """
    ngMeM: penalitza variacions excessives respecte a una referència plana m = 1/64
    """
    m = 1.0 / n_pix
    suma_ngmem = 0.0
    for t in range(n_t):
        for p in range(n_pix):
            I_val = movie[t, p]
            suma_ngmem += I_val * np.log(I_val / m) - I_val + m
    return suma_ngmem

# ----------------------------------------------------------------------
# 5. Funció Objectiu total a minimitzar
# ----------------------------------------------------------------------
def funcio_objectiu(vector_pla, lmbda, mu): #vecto pla = vector 1D de (n_t * n_pix)= 12*64=768 elements
    movie = vector_pla.reshape((n_t, n_pix))
    
    v_mod = calcular_visibilitats_model(movie)
    terme_dades = calcular_chi2_total(v_mod, vis_obs)
    terme_espai = calcular_regularitzador_espacial_ngmem(movie)
    terme_temps = calcular_regularitzador_temporal_L2(movie)
    
    return terme_dades + (lmbda * terme_espai) + (mu * terme_temps)

# ----------------------------------------------------------------------
# 6. Grid Search 2D (lambda, mu)
# ----------------------------------------------------------------------
lambdas = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]
mus = [1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]

results = []
movies = {}

# Condicions inicials i fites
I_inicial = np.full(n_t * n_pix, 1.0 / n_pix)
limits = [(1e-8, None)] * (n_t * n_pix)

print(f"\nIniciant Grid Search 2D ({len(lambdas)} x {len(mus)} = {len(lambdas)*len(mus)} parelles)...")

for lmbda in lambdas:
    for mu in mus:
        t0 = time.time()
        
        res = minimize(
            funcio_objectiu,
            I_inicial,
            args=(lmbda, mu),
            method="L-BFGS-B",
            bounds=limits,
            options={"maxiter": 100}
        )
        
        movie_reco = res.x.reshape((n_t, n_pix))
        elapsed = time.time() - t0
        
        # Avaluem les mètriques amb la solució trobada
        v_pred = calcular_visibilitats_model(movie_reco)
        chi2_val = calcular_chi2_total(v_pred, vis_obs)
        chi2_red = chi2_val / n_data
        
        print(f" -> lambda={lmbda:8.2g} | mu={mu:8.2g} | chi2_red={chi2_red:6.2f} | {elapsed:.1f}s")
        
        results.append({
            "lambda": lmbda,
            "mu": mu,
            "chi2_total": float(chi2_val),
            "chi2_red": float(chi2_red),
            "time_seconds": round(elapsed, 2),
            "converged": bool(res.success)
        })
        movies[(lmbda, mu)] = movie_reco
        
        # Warm-start: la solució actual serveix de punt de partida per a la següent
        I_inicial = res.x.copy()

# Convertim a taula de resultats
df = pd.DataFrame(results)
df.to_csv("resultats_grid_lambda_mu.csv", index=False)
print("\n[OK] Taula guardada a 'resultats_grid_lambda_mu.csv'.")

# ----------------------------------------------------------------------
# 7. Selecció del millor model pel Principi de Discrepància (chi2_red ~ 1)
# ----------------------------------------------------------------------
best = df.loc[(df["chi2_red"] - 1.0).abs().idxmin()]
best_key = (best["lambda"], best["mu"])
best_movie = movies[best_key]

print(f"\n=======================================================")
print(f"Millor parella trobada (chi2_red mes proper a 1):")
print(f" - lambda: {best['lambda']}")
print(f" - mu:     {best['mu']}")
print(f" - chi2_red: {best['chi2_red']:.3f}")
print(f"=======================================================")

# ----------------------------------------------------------------------
# 8. Gràfics finals
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Perfils reconstruïts al llarg del temps
colors = plt.cm.viridis(np.linspace(0, 1, n_t))
for t in range(n_t):
    axes[0].plot(xi, best_movie[t, :], color=colors[t], lw=1.8, label=f"t = {times[t]:.2f}")
axes[0].set_title(f"Pel·lícula Òptima (lambda={best['lambda']}, mu={best['mu']})")
axes[0].set_xlabel("Posició espacial xi")
axes[0].set_ylabel("Intensitat I(xi, t)")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(ncol=2, fontsize=8)

# (b) Evolució del centroide (centre de massa de brillantor vs temps)
centroids = np.sum(best_movie * xi[None, :], axis=1) / np.sum(best_movie, axis=1)
axes[1].plot(centroids, times, "o-", color="crimson", lw=2)
axes[1].set_title("Cinemàtica de la font (Centroide vs Temps)")
axes[1].set_xlabel("Posició del centroide (xi)")
axes[1].set_ylabel("Temps (t)")
axes[1].set_xlim(-1, 1)
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()