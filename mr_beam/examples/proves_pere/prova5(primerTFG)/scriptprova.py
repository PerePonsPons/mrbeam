import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# 1. Carregar i concatenar dades
# ----------------------------------------------------------------------
x_true = np.load("x.npy")  # (64,)
times = np.load("times.npy")  # (12,)
noise_std = float(np.load("noise_std.npy"))  # 0.025

# Carregar blocs originals
nu_train = np.load("train_frequencies.npy")  # (12, 9)
nu_val = np.load("validation_frequencies.npy")  # (12, 5)
vis_train = np.load("vis_train.npy")  # (12, 9), complex
vis_val = np.load("vis_validation.npy")  # (12, 5), complex

# Concatenar en les variables d'observació completes
nu_obs = np.concatenate([nu_train, nu_val], axis=1)  # (12, 14)
vis_obs = np.concatenate([vis_train, vis_val], axis=1)  # (12, 14), complex

print("[OK] Dades carregades i unificades correctament.")
print(f" - vis_obs shape: {vis_obs.shape}")
print(f" - nu_obs shape:  {nu_obs.shape}")
print(f" - Soroll (sigma): {noise_std}")

# ----------------------------------------------------------------------
# 2. Operador de Fourier 1D (van Cittert-Zernike directe)
# ----------------------------------------------------------------------
n_pix = len(x_true)
xi = np.linspace(-0.5, 0.5, n_pix)


def forward_operator(x_vec, t_arr, nu_arr):
    """Calcula les visibilitats modelades per a cada parell (temps, freqüència)."""
    n_t, n_nu = nu_arr.shape
    vis_model = np.zeros((n_t, n_nu), dtype=np.complex128)

    # Normalització respecte a les freqüències d'observació
    nu_norm = nu_arr / np.max(nu_obs)

    for i in range(n_t):
        for j in range(n_nu):
            u = t_arr[i] * nu_norm[i, j]
            fourier_kernel = np.exp(-2j * np.pi * u * xi)
            vis_model[i, j] = np.dot(fourier_kernel, x_vec)
    return vis_model


# ----------------------------------------------------------------------
# 3. Força Bruta: Provar diferents pesos pel regularitzador gaussià
# ----------------------------------------------------------------------
D = np.diff(np.eye(n_pix), n=2, axis=0)

lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
results = []
x_init = np.zeros(n_pix)

print(f"\nIniciant exploració per força bruta ({len(lambdas)} valors de pes)...")

for lmbda in lambdas:
    t0 = time.time()

    # Funció de pèrdua: Chi2(obs) + lambda * Regularitzador Gaussià
    def loss(x_candidate):
        v_pred = forward_operator(x_candidate, times, nu_obs)
        chi2 = np.sum(np.abs(v_pred - vis_obs) ** 2) / (noise_std**2)
        reg = np.sum((D @ x_candidate) ** 2)
        return chi2 + lmbda * reg

    opt = minimize(loss, x_init, method="L-BFGS-B", options={"maxiter": 100})
    x_reco = opt.x
    elapsed = time.time() - t0

    # Avaluació de l'ajust respecte a les observacions
    v_obs_reco = forward_operator(x_reco, times, nu_obs)
    chi2_obs = np.sum(np.abs(v_obs_reco - vis_obs) ** 2) / (noise_std**2)

    print(
        f" -> Lambda: {lmbda:8.4f} | Chi2 Obs: {chi2_obs:8.2f} | Temps: {elapsed:.2f}s"
    )

    results.append({
        "lambda_weight": lmbda,
        "chi2_obs": float(chi2_obs),
        "time_seconds": round(elapsed, 2),
        "converged": bool(opt.success),
    })

# ----------------------------------------------------------------------
# 4. Desar la taula a CSV
# ----------------------------------------------------------------------
df = pd.DataFrame(results)
output_file = "resultats_forca_bruta.csv"
df.to_csv(output_file, index=False)

print(f"\n[FINALITZAT] Taula guardada amb èxit a '{output_file}'!")
print(df)