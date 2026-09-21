import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# 1. Carregar dades
# ----------------------------------------------------------------------
x_true = np.load("x.npy")  # (64,)
times = np.load("times.npy")  # (12,)
nu_train = np.load("train_frequencies.npy")  # (12, 9)
nu_val = np.load("validation_frequencies.npy")  # (12, 5)
vis_train = np.load("vis_train.npy")  # (12, 9), complex
vis_val = np.load("vis_validation.npy")  # (12, 5), complex
noise_std = float(np.load("noise_std.npy"))  # 0.025

print(f"[OK] Dades carregades correctament.")
print(f" - vis_train shape: {vis_train.shape}")
print(f" - vis_val shape:   {vis_val.shape}")
print(f" - Soroll (sigma):  {noise_std}")

# ----------------------------------------------------------------------
# 2. Operador de Fourier 1D (van Cittert-Zernike directe)
# ----------------------------------------------------------------------
# Graella espacial normalitzada entre -0.5 i 0.5
n_pix = len(x_true)
xi = np.linspace(-0.5, 0.5, n_pix)


def forward_operator(x_vec, t_arr, nu_arr):
  """Calcula les visibilitats modelades per a cada parell (temps, freqüència)."""
  n_t, n_nu = nu_arr.shape
  vis_model = np.zeros((n_t, n_nu), dtype=np.complex128)

  # Freqüències normalitzades per estabilitat numèrica
  nu_norm = nu_arr / np.max(nu_train)

  for i in range(n_t):
    for j in range(n_nu):
      u = times[i] * nu_norm[i, j]
      fourier_kernel = np.exp(-2j * np.pi * u * xi)
      vis_model[i, j] = np.dot(fourier_kernel, x_vec)
  return vis_model


# ----------------------------------------------------------------------
# 3. Força Bruta: Provar diferents pesos pel regularitzador gaussià
# ----------------------------------------------------------------------
# Matriu de segones diferències per al regularitzador de suavitat (gaussià/Tikhonov)
D = np.diff(np.eye(n_pix), n=2, axis=0)

# Graella de pesos lambda a explorar
lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
results = []
x_init = np.zeros(n_pix)

print(f"\nIniciant exploració per força bruta ({len(lambdas)} valors de pes)...")

for lmbda in lambdas:
  t0 = time.time()

  # Funció de pèrdua: Chi2(train) + lambda * Regularitzador Gaussià
  def loss(x_candidate):
    v_pred = forward_operator(x_candidate, times, nu_train)
    # Chi-quadrat respecte a dades d'entrenament ponderat pel soroll
    chi2 = np.sum(np.abs(v_pred - vis_train) ** 2) / (noise_std**2)
    # Terme regularitzador gaussià (suavitat de curvatura)
    reg = np.sum((D @ x_candidate) ** 2)
    return chi2 + lmbda * reg

  # Minimització numèrica (L-BFGS-B)
  opt = minimize(loss, x_init, method="L-BFGS-B", options={"maxiter": 100})
  x_reco = opt.x
  elapsed = time.time() - t0

  # Avaluar l'ajust a Train i la capacitat predictiva a Validation
  v_train_reco = forward_operator(x_reco, times, nu_train)
  chi2_train = np.sum(np.abs(v_train_reco - vis_train) ** 2) / (noise_std**2)

  v_val_reco = forward_operator(x_reco, times, nu_val)
  chi2_val = np.sum(np.abs(v_val_reco - vis_val) ** 2) / (noise_std**2)

  print(
      f" -> Lambda: {lmbda:8.4f} | Chi2 Train: {chi2_train:8.2f} | Chi2 Val:"
      f" {chi2_val:8.2f} | Temps: {elapsed:.2f}s"
  )

  results.append({
      "lambda_weight": lmbda,
      "chi2_train": float(chi2_train),
      "chi2_validation": float(chi2_val),
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