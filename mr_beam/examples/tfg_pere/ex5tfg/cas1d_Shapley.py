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


def solve_reconstruction(lmbda, mu, x_init=None, maxiter=500):
  """Reconstrueix la pel·lícula per a una parella (lambda, mu) donada."""
  t0 = time.time()
  if x_init is None:
    x0 = np.full(n_t * n_pix, prior_m)
  else:
    x0 = x_init.copy()
  bounds = [(EPS, None)] * (n_t * n_pix)

  opt = minimize(
      objective,
      x0,
      args=(lmbda, mu),
      jac=True,
      method="L-BFGS-B",
      bounds=bounds,
      options={"maxiter": maxiter},
  )
  movie = opt.x.reshape(n_t, n_pix)
  elapsed = time.time() - t0

  chi2_val = chi2_of(movie, A_obs, vis_obs)
  chi2_red = chi2_val / n_data
  ngmem_val = np.sum(movie * np.log(movie / prior_m) - movie + prior_m)
  l2t_val = np.sum(np.diff(movie, axis=0) ** 2)

  return {
      "movie": movie,
      "flat": opt.x,
      "chi2_obs": float(chi2_val),
      "chi2_red": float(chi2_red),
      "ngmem": float(ngmem_val),
      "l2_temporal": float(l2t_val),
      "time_seconds": round(elapsed, 2),
      "converged": bool(opt.success),
  }


# ----------------------------------------------------------------------
# 4. Teoria de jocs cooperatius: Determinació de pesos per valors de Shapley
# ----------------------------------------------------------------------
# Es formula un joc cooperatiu amb 2 jugadors (regularitzadors):
#   - Jugador lambda (ngMeM: regularització espacial d'entropia)
#   - Jugador mu     (L2: regularització temporal de continuïtat)
# Sota la restricció física del principi de discrepància (chi2_red ~ 1).
#
# Coalicions S possibles:
#   - S = buida:         Cap regularització (lambda = 0, mu = 0).
#   - S = {lambda}:      Només regularització espacial activa.
#   - S = {mu}:          Només regularització temporal activa.
#   - S = {lambda, mu}:  Gran coalició amb ambdós regularitzadors cooperant.

print("\n" + "=" * 70)
print(" DETERMINACIÓ DELS PESOS MITJANÇANT ELS VALORS DE SHAPLEY")
print("=" * 70)

# (4.1) Coalició buida S = {} (reconstrucció purament basada en dades)
print("\n[1/4] Avaluant la coalició buida {} (lambda=0, mu=0)...")
res_empty = solve_reconstruction(0.0, 0.0)
s0 = res_empty["ngmem"]
t0 = res_empty["l2_temporal"]
print(f" -> Coalició buida: chi2_red={res_empty['chi2_red']:.3f} | ngMeM={s0:.3f} | L2_t={t0:.3f}")


# (4.2) Cerca del millor lambda* individual (coalició {lambda}, mu=0)
def find_optimal_lambda_alone():
  low, high = 1.0, 500.0
  x_prev = res_empty["flat"]
  for _ in range(12):
    mid = np.sqrt(low * high)
    sol = solve_reconstruction(mid, 0.0, x_init=x_prev)
    x_prev = sol["flat"]
    if sol["chi2_red"] < 1.0:
      low = mid
    else:
      high = mid
  opt_l = np.sqrt(low * high)
  return opt_l, solve_reconstruction(opt_l, 0.0, x_init=x_prev)


print("\n[2/4] Calibrant la coalició {lambda} (ngMeM sol fins a chi2_red ~ 1)...")
l_star, res_l = find_optimal_lambda_alone()
s_l = res_l["ngmem"]
t_l = res_l["l2_temporal"]
print(f" -> lambda* = {l_star:.3f} | chi2_red={res_l['chi2_red']:.3f} | ngMeM={s_l:.3f} | L2_t={t_l:.3f}")


# (4.3) Cerca del millor mu* individual (coalició {mu}, lambda=0)
def find_optimal_mu_alone():
  low, high = 100.0, 20000.0
  x_prev = res_empty["flat"]
  for _ in range(12):
    mid = np.sqrt(low * high)
    sol = solve_reconstruction(0.0, mid, x_init=x_prev)
    x_prev = sol["flat"]
    if sol["chi2_red"] < 1.0:
      low = mid
    else:
      high = mid
  opt_m = np.sqrt(low * high)
  return opt_m, solve_reconstruction(0.0, opt_m, x_init=x_prev)


print("\n[3/4] Calibrant la coalició {mu} (L2 temporal sol fins a chi2_red ~ 1)...")
m_star, res_m = find_optimal_mu_alone()
s_m = res_m["ngmem"]
t_m = res_m["l2_temporal"]
print(f" -> mu* = {m_star:.3f} | chi2_red={res_m['chi2_red']:.3f} | ngMeM={s_m:.3f} | L2_t={t_m:.3f}")


# (4.4) Gran coalició {lambda, mu} (cooperació d'ambdós regularitzadors)
def solve_grand_coalition(l_base, m_base):
  low_c, high_c = 0.2, 3.0
  x_prev = res_empty["flat"]
  for _ in range(10):
    mid_c = np.sqrt(low_c * high_c)
    sol = solve_reconstruction(mid_c * 0.5 * l_base, mid_c * 0.5 * m_base, x_init=x_prev)
    x_prev = sol["flat"]
    if sol["chi2_red"] < 1.0:
      low_c = mid_c
    else:
      high_c = mid_c
  c_opt = np.sqrt(low_c * high_c)
  return solve_reconstruction(c_opt * 0.5 * l_base, c_opt * 0.5 * m_base, x_init=x_prev)


print("\n[4/4] Avaluant la gran coalició {lambda, mu}...")
res_lm = solve_grand_coalition(l_star, m_star)
s_lm = res_lm["ngmem"]
t_lm = res_lm["l2_temporal"]
print(f" -> Gran coalició: chi2_red={res_lm['chi2_red']:.3f} | ngMeM={s_lm:.3f} | L2_t={t_lm:.3f}")

# (4.5) Definició de la funció característica v(S)
# La utilitat normalitzada d'un regularitzador reflecteix la reducció del seu cost
# respecte a la solució base sense regularitzar (normalitzada pel seu òptim individual):
#   u_lambda(I) = (s0 - ngmem(I)) / (s0 - s_l)
#   u_mu(I)     = (t0 - l2t(I))   / (t0 - t_m)
v_empty = 0.0
v_lambda = 1.0  # Assolit al 100% pel jugador lambda en solitari
v_mu = 1.0      # Assolit al 100% pel jugador mu en solitari

u_lambda_coop = (s0 - s_lm) / (s0 - s_l)
u_mu_coop = (t0 - t_lm) / (t0 - t_m)
v_lambda_mu = u_lambda_coop + u_mu_coop

print("\n--- Valors de la funció característica v(S) ---")
print(f" v(Ø)            = {v_empty:.4f}")
print(f" v({{lambda}})     = {v_lambda:.4f}")
print(f" v({{mu}})         = {v_mu:.4f}")
print(f" v({{lambda, mu}}) = {v_lambda_mu:.4f}  (u_lambda={u_lambda_coop:.3f}, u_mu={u_mu_coop:.3f})")

# (4.6) Càlcul dels valors de Shapley:
# Formula clàssica per a 2 jugadors:
#   phi_i = (1/2) * [v({i}) - v(Ø)] + (1/2) * [v({lambda, mu}) - v({altre})]
phi_lambda = 0.5 * (v_lambda - v_empty) + 0.5 * (v_lambda_mu - v_mu)
phi_mu = 0.5 * (v_mu - v_empty) + 0.5 * (v_lambda_mu - v_lambda)

print("\n--- Valors de Shapley calculats ---")
print(f" phi(lambda) = {phi_lambda:.4f}")
print(f" phi(mu)     = {phi_mu:.4f}")
print(f" Suma phi    = {phi_lambda + phi_mu:.4f} (comprovació eficiència: v({{lambda, mu}})={v_lambda_mu:.4f})")

# (4.7) Pesos relatius normalitzats i pesos finals per a la reconstrucció
w_lambda = phi_lambda / (phi_lambda + phi_mu)
w_mu = phi_mu / (phi_lambda + phi_mu)

# Calibrem l'escala global c per assegurar chi2_red ~ 1 amb la proporció Shapley
low_c, high_c = 0.2, 3.0
x_prev = res_lm["flat"]
for _ in range(10):
  mid_c = np.sqrt(low_c * high_c)
  sol = solve_reconstruction(mid_c * w_lambda * l_star, mid_c * w_mu * m_star, x_init=x_prev)
  x_prev = sol["flat"]
  if sol["chi2_red"] < 1.0:
    low_c = mid_c
  else:
    high_c = mid_c

c_final = np.sqrt(low_c * high_c)
best_lambda = float(c_final * w_lambda * l_star)
best_mu = float(c_final * w_mu * m_star)

# Reconstrucció final definitiva amb els pesos de Shapley
res_final = solve_reconstruction(best_lambda, best_mu, x_init=x_prev)
best_movie = res_final["movie"]

print("\n" + "=" * 70)
print(f" MILLORS PESOS OBTINGUTS PER SHAPLEY:")
print(f"   lambda_Shapley = {best_lambda:.4f}")
print(f"   mu_Shapley     = {best_mu:.4f}")
print(f"   chi2_red final = {res_final['chi2_red']:.4f}")
print("=" * 70)

# Guardem la taula de resultats cooperatius
df_shapley = pd.DataFrame([
    {
        "coalicio": "Buida Ø",
        "lambda": 0.0,
        "mu": 0.0,
        "v_S": v_empty,
        "chi2_red": res_empty["chi2_red"],
        "ngmem": s0,
        "l2_temporal": t0,
    },
    {
        "coalicio": "{lambda}",
        "lambda": l_star,
        "mu": 0.0,
        "v_S": v_lambda,
        "chi2_red": res_l["chi2_red"],
        "ngmem": s_l,
        "l2_temporal": t_l,
    },
    {
        "coalicio": "{mu}",
        "lambda": 0.0,
        "mu": m_star,
        "v_S": v_mu,
        "chi2_red": res_m["chi2_red"],
        "ngmem": s_m,
        "l2_temporal": t_m,
    },
    {
        "coalicio": "{lambda, mu} Shapley",
        "lambda": best_lambda,
        "mu": best_mu,
        "v_S": v_lambda_mu,
        "chi2_red": res_final["chi2_red"],
        "ngmem": res_final["ngmem"],
        "l2_temporal": res_final["l2_temporal"],
    },
])
df_shapley.to_csv("resultats_shapley_lambda_mu.csv", index=False)
print("\n[FINALITZAT] Taula guardada a 'resultats_shapley_lambda_mu.csv'")
print(df_shapley)


# ----------------------------------------------------------------------
# 5. Gràfics
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Representació dels valors de Shapley i funció característica
categories = [r"$\phi_\lambda$ (ngMeM)", r"$\phi_\mu$ (L2 temporal)"]
shapley_vals = [phi_lambda, phi_mu]
bar_colors = ["#2b7bba", "#e67300"]

bars = axes[0].bar(categories, shapley_vals, color=bar_colors, width=0.45, edgecolor="black", alpha=0.85)
for bar in bars:
  h = bar.get_height()
  axes[0].annotate(
      f"{h:.4f}",
      xy=(bar.get_x() + bar.get_width() / 2, h),
      xytext=(0, 4),
      textcoords="offset points",
      ha="center",
      va="bottom",
      fontsize=10,
      fontweight="bold",
  )

axes[0].set_ylim(0, max(shapley_vals) * 1.25)
axes[0].set_ylabel("Valor de Shapley")
axes[0].set_title(
    rf"Valors de Shapley per a $\lambda$ i $\mu$" + "\n"
    rf"($\lambda_{{Shapley}}={best_lambda:.2f}$, $\mu_{{Shapley}}={best_mu:.2f}$, $\chi^2_{{red}}={res_final['chi2_red']:.2f}$)",
    fontsize=11,
)
axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

# (b) Pel·lícula reconstruïda (perfils per temps)
colors = plt.cm.viridis(np.linspace(0, 1, n_t))
for i, t in enumerate(times):
  axes[1].plot(xi, best_movie[i], color=colors[i], lw=1.8, label=f"t={t:.2f}")
axes[1].set_title(rf"Pel.lícula reconstruïda (Shapley: $\lambda$={best_lambda:.2f}, $\mu$={best_mu:.2f})")
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
plt.savefig("shapley_lambda_mu_pellicula.png", dpi=200)
print("\n[OK] Gràfic guardat a 'shapley_lambda_mu_pellicula.png'")
plt.show()
