import time
import matplotlib.pyplot as plt
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

# ----------------------------------------------------------------------
# 2. Operador de Fourier 1D (van Cittert-Zernike directe)
# ----------------------------------------------------------------------
n_pix = len(x_true)
xi = np.linspace(-0.5, 0.5, n_pix)


def forward_operator(x_vec, t_arr, nu_arr):
  """Calcula les visibilitats modelades per a cada parell (temps, freqüència)."""
  n_t, n_nu = nu_arr.shape
  vis_model = np.zeros((n_t, n_nu), dtype=np.complex128)

  nu_norm = nu_arr / np.max(nu_obs)

  for i in range(n_t):
    for j in range(n_nu):
      u = t_arr[i] * nu_norm[i, j]
      fourier_kernel = np.exp(-2j * np.pi * u * xi)
      vis_model[i, j] = np.dot(fourier_kernel, x_vec)
  return vis_model


# ----------------------------------------------------------------------
# 3. Força Bruta: Explorar valors de lambda amb regularitzador norma L_2
# ----------------------------------------------------------------------
lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
results = []
models_reconstructed = []
x_init = np.zeros(n_pix)

print(
    f"\nIniciant exploració per força bruta ({len(lambdas)} valors de"
    " pes, regularitzador L2)..."
)

for lmbda in lambdas:
  t0 = time.time()

  def loss(x_candidate):
    v_pred = forward_operator(x_candidate, times, nu_obs)
    chi2 = np.sum(np.abs(v_pred - vis_obs) ** 2) / (noise_std**2)
    # Regularitzador norma L2 pura: ||x||_2^2
    reg = np.sum(x_candidate**2)    
    # Regularitzador ngMeM: ||x - x_true||_2^2
    ngmem_reg = np.sum(np.abs(x_candidate - x_true)**2)
    return chi2 + lmbda * reg

  opt = minimize(loss, x_init, method="L-BFGS-B", options={"maxiter": 100})
  x_reco = opt.x
  elapsed = time.time() - t0

  v_obs_reco = forward_operator(x_reco, times, nu_obs)
  chi2_obs = np.sum(np.abs(v_obs_reco - vis_obs) ** 2) / (noise_std**2)

  print(
      f" -> Lambda: {lmbda:8.4f} | Chi2 Obs: {chi2_obs:8.2f} | Temps:"
      f" {elapsed:.2f}s"
  )

  results.append({
      "lambda_weight": lmbda,
      "chi2_obs": float(chi2_obs),
      "time_seconds": round(elapsed, 2),
      "converged": bool(opt.success),
  })
  models_reconstructed.append(x_reco)

# Desar la taula a CSV
df = pd.DataFrame(results)
output_file = "resultats_forca_bruta_l2.csv"
df.to_csv(output_file, index=False)
print(f"\n[FINALITZAT] Taula guardada a '{output_file}'!")
print(df)


# ----------------------------------------------------------------------
# 4. Ajust d'una Gaussiana pura per a cada pas de temps (t)
# ----------------------------------------------------------------------
def gaussian_profile(xi_grid, amp, mu, sigma):
  """Genera una gaussiana 1D analítica."""
  return amp * np.exp(-0.5 * ((xi_grid - mu) / (sigma + 1e-12)) ** 2)


# Paràmetres que guardarem per a cada t
fit_params = []
gaussians_per_t = []

# Suposició inicial: [amplitud, centre_mu, amplada_sigma]
p0 = [2.0, 0.0, 0.08]
bounds = [(0.0, 20.0), (-0.4, 0.4), (0.01, 0.3)]

for i, t in enumerate(times):
  v_obs_t = vis_obs[i : i + 1, :]
  nu_t = nu_obs[i : i + 1, :]
  t_single = np.array([t])

  def loss_gaussian(params):
    amp, mu, sigma = params
    profile = gaussian_profile(xi, amp, mu, sigma)
    v_pred = forward_operator(profile, t_single, nu_t)
    return np.sum(np.abs(v_pred - v_obs_t) ** 2) / (noise_std**2)

  res = minimize(loss_gaussian, p0, method="L-BFGS-B", bounds=bounds)

  amp_fit, mu_fit, sigma_fit = res.x
  fit_params.append({"t": t, "amp": amp_fit, "mu": mu_fit, "sigma": sigma_fit})

  gauss_curve = gaussian_profile(xi, amp_fit, mu_fit, sigma_fit)
  gaussians_per_t.append(gauss_curve)

  p0 = [amp_fit, mu_fit, sigma_fit]

df_motion = pd.DataFrame(fit_params)

# ----------------------------------------------------------------------
# 5. Gràfics: Gaussianes per temps i Moviment de l'objecte
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
for i, t in enumerate(times):
  axes[0].plot(
      xi,
      gaussians_per_t[i],
      label=f"t = {t:.2f}",
      color=colors[i],
      lw=2.0,
      alpha=0.9,
  )

axes[0].set_title("Evolució temporal del perfil gaussià", fontsize=12)
axes[0].set_xlabel(r"Posició espacial $\xi$", fontsize=11)
axes[0].set_ylabel("Intensitat", fontsize=11)
axes[0].set_xlim(-0.5, 0.5)
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(loc="upper right", fontsize=8, ncol=2)

axes[1].plot(
    df_motion["t"],
    df_motion["mu"],
    marker="o",
    linestyle="-",
    color="crimson",
    lw=2.2,
    markersize=6,
    label=r"Posició del centre $\mu(t)$",
)
axes[1].set_title("Moviment de l'objecte (Posició vs Temps)", fontsize=12)
axes[1].set_xlabel("Temps (t)", fontsize=11)
axes[1].set_ylabel(r"Posició espacial del centre ($\mu$)", fontsize=11)
axes[1].set_ylim(-0.4, 0.4)
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(loc="best", fontsize=10)

plt.tight_layout()
plt.savefig("evolucio_i_moviment_gaussiana.png", dpi=200)
plt.show()