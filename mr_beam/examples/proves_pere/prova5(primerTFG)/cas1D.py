import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# 1. Carregar dades
# ----------------------------------------------------------------------
x_grid = np.load("x.npy")  # (64,) graella espacial [-1, 1] (NO es la imatge)
times = np.load("times.npy")  # (12,)
noise_std = float(np.load("noise_std.npy"))  # 0.025

nu_train = np.load("train_frequencies.npy")  # (12, 9)
nu_val = np.load("validation_frequencies.npy")  # (12, 5)
vis_train = np.load("vis_train.npy")  # (12, 9), complex
vis_val = np.load("vis_validation.npy")  # (12, 5), complex

# El simulador separa train/validation per a ML; aqui no cal: unifiquem tot.
nu_obs = np.concatenate([nu_train, nu_val], axis=1)  # (12, 14)
vis_obs = np.concatenate([vis_train, vis_val], axis=1)  # (12, 14), complex
n_t = len(times)
n_pix = len(x_grid)
xi = x_grid

print("[OK] Dades carregades.")
print(f" - vis_obs shape: {vis_obs.shape}")
print(f" - nu_obs shape:  {nu_obs.shape}")


# ----------------------------------------------------------------------
# 2. Operador de Fourier 1D (una matriu A_t per fotograma)
# ----------------------------------------------------------------------
def build_operator(nu_arr):
  """Retorna A amb shape (n_t, n_nu, n_pix): vis[t] = A[t] @ I[t]."""
  # Nucli de van Cittert-Zernike: exp(-2*pi*i*nu*xi), amb nu = index enter
  # de les dades (sense normalitzar) i xi a [-1, 1].
  # Verificat amb les dades: introduir times (u = t*nu) o normalitzar nu per
  # max(nu) empitjora l'ajust enormement (chi2_red ~ 100-180 vs ~ 0.1), i a
  # t = 0 les visibilitats ja varien amb nu; per tant el temps NOMES enllaca
  # fotogrames (cada t te la seva imatge) i no entra al nucli.
  return np.exp(-2j * np.pi * nu_arr[:, :, None] * xi[None, None, :])


A_obs = build_operator(nu_obs)
n_data = 2 * vis_obs.size  # dades reals (Re + Im), per al chi2 reduit


def chi2_of(movie, A, vis):
  """chi2 sumat sobre tots els fotogrames. movie: (n_t, n_pix)."""
  v_pred = np.einsum("tkp,tp->tk", A, movie)
  return np.sum(np.abs(v_pred - vis) ** 2) / noise_std**2


# ----------------------------------------------------------------------
# 3. Funcional: chi2 + lambda * ngMeM + mu * L2 temporal
# ----------------------------------------------------------------------
# ngMeM (regularitzador espacial per fotograma): entropia respecte a un prior m
#   S = sum_p [ I log(I/m) - I + m ]   (cal I > 0)
# L2 temporal (continuitat): sum_k || I^{t_{k+1}} - I^{t_k} ||_2^2
EPS = 1e-8
prior_m = 1.0 / n_pix  # prior pla (flux total ~1, vis(0) ~ 1)


def objective(flat, lmbda, mu):
  movie = flat.reshape(n_t, n_pix)

  # chi2 + gradient
  resid = np.einsum("tkp,tp->tk", A_obs, movie) - vis_obs
  chi2 = np.sum(np.abs(resid) ** 2) / noise_std**2
  g_chi2 = 2 * np.real(np.einsum("tkp,tk->tp", A_obs.conj(), resid))
  g_chi2 /= noise_std**2

  # ngMeM + gradient
  ngmem = np.sum(movie * np.log(movie / prior_m) - movie + prior_m)
  g_ngmem = np.log(movie / prior_m)

  # L2 temporal + gradient
  diff = np.diff(movie, axis=0)  # (n_t-1, n_pix)
  l2t = np.sum(diff**2)
  g_l2t = np.zeros_like(movie)
  g_l2t[1:] += 2 * diff
  g_l2t[:-1] -= 2 * diff

  f = chi2 + lmbda * ngmem + mu * l2t
  g = g_chi2 + lmbda * g_ngmem + mu * g_l2t
  return f, g.ravel()


# ----------------------------------------------------------------------
# 4. Grid Search 2D (lambda, mu) per forca bruta
# ----------------------------------------------------------------------
lambdas = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]
mus = [1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]

results = []
movies = {}
x_start = np.full(n_t * n_pix, prior_m)  # nomes per a la primera parella
bounds = [(EPS, None)] * (n_t * n_pix)

print(f"\nGrid search 2D: {len(lambdas)} x {len(mus)} = {len(lambdas) * len(mus)} parelles")

for lmbda in lambdas:
  for mu in mus:
    t0 = time.time()
    opt = minimize(
        objective,
        x_start,
        args=(lmbda, mu),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )
    movie = opt.x.reshape(n_t, n_pix)
    x_start = opt.x.copy()  # warm-start: la solucio serveix de punt inicial al seguent
    elapsed = time.time() - t0

    chi2_obs = chi2_of(movie, A_obs, vis_obs)
    ngmem_v = np.sum(movie * np.log(movie / prior_m) - movie + prior_m)
    l2t_v = np.sum(np.diff(movie, axis=0) ** 2)

    print(
        f" -> lambda={lmbda:8.4g} mu={mu:8.4g} | chi2_obs={chi2_obs:9.2f} |"
        f" chi2_red={chi2_obs / n_data:6.2f} | {elapsed:.1f}s"
    )
    results.append({
        "lambda": lmbda,
        "mu": mu,
        "chi2_obs": float(chi2_obs),
        "chi2_red": float(chi2_obs / n_data),
        "ngmem": float(ngmem_v),
        "l2_temporal": float(l2t_v),
        "time_seconds": round(elapsed, 2),
        "converged": bool(opt.success),
    })
    movies[(lmbda, mu)] = movie

df = pd.DataFrame(results)
df.to_csv("resultats_grid_lambda_mu.csv", index=False)
print("\n[FINALITZAT] Taula guardada a 'resultats_grid_lambda_mu.csv'")
print(df)

# Millor parella: principi de discrepancia (chi2 reduit mes proper a 1)
best = df.loc[(df["chi2_red"] - 1).abs().idxmin()]
best_key = (best["lambda"], best["mu"])
best_movie = movies[best_key]
print(f"\nMillor parella (chi2_red ~ 1): lambda={best['lambda']}, mu={best['mu']}")

# ----------------------------------------------------------------------
# 5. Grafics
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Mapa de chi2_red sobre la graella (lambda, mu)
pivot = df.pivot(index="mu", columns="lambda", values="chi2_red")
im = axes[0].imshow(np.log10(pivot.values), origin="lower", cmap="viridis")
axes[0].set_xticks(range(len(lambdas)), [f"{v:g}" for v in pivot.columns])
axes[0].set_yticks(range(len(mus)), [f"{v:g}" for v in pivot.index])
axes[0].set_xlabel(r"$\lambda$ (ngMeM)")
axes[0].set_ylabel(r"$\mu$ (L2 temporal)")
axes[0].set_title(r"$\log_{10}\chi^2_{red}$")
fig.colorbar(im, ax=axes[0])

# (b) Pel.licula reconstruida (perfils per temps)
colors = plt.cm.viridis(np.linspace(0, 1, n_t))
for i, t in enumerate(times):
  axes[1].plot(xi, best_movie[i], color=colors[i], lw=1.8, label=f"t={t:.2f}")
axes[1].set_title(rf"Pel.lícula reconstruïda ($\lambda$={best['lambda']:g}, $\mu$={best['mu']:g})")
axes[1].set_xlabel(r"Posició espacial $\xi$")
axes[1].set_ylabel("Intensitat")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(fontsize=7, ncol=2)

# (c) Moviment: centroide de cada fotograma
centroids = (best_movie * xi[None, :]).sum(axis=1) / best_movie.sum(axis=1)
axes[2].plot(centroids, times, "o-", color="crimson", lw=2)
axes[2].set_title("Moviment (centroide vs temps)")
axes[2].set_xlabel("Posició del centroide")
axes[2].set_ylabel("Temps (t)")
axes[2].set_xlim(-1, 1)
axes[2].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("grid_lambda_mu_pellicula.png", dpi=200)
plt.show()