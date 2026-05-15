#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

import ehtim as eh
import pygmo as pg

import GA.solver as solver
from GA.problems import Scattering
from GA.pso import CooperativeGame
from pyswarms.utils.plotters import plot_cost_history

plt.ioff()


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_thread_env(n: int | None = None) -> None:
    if n is None:
        n = min(32, os.cpu_count() or 1)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)


def ensure_odd(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)


def select_band(obs: eh.obsdata.Obsdata,
                nu0_ghz: float,
                halfwidth_ghz: float) -> eh.obsdata.Obsdata:
    """Generic frequency filter if uvfits has per-row freq."""
    nu0 = nu0_ghz * 1e9
    hw = halfwidth_ghz * 1e9

    names = obs.data.dtype.names
    freq_field = None
    for key in ("freq", "nu", "rf"):
        if key in names:
            freq_field = key
            break
    if freq_field is None:
        return obs

    nu = obs.data[freq_field].astype(float)
    mask = (nu >= (nu0 - hw)) & (nu <= (nu0 + hw))
    if not np.any(mask):
        raise ValueError("Band selection removed all rows. Check --nu0-ghz / --halfwidth-ghz.")

    obs2 = obs.copy()
    obs2.data = obs2.data[mask]
    try:
        obs2.rf = float(np.nanmedian(nu[mask]))
    except Exception:
        pass
    return obs2


def ensure_stokesI_obs(obs: eh.obsdata.Obsdata) -> eh.obsdata.Obsdata:
    if getattr(obs, "polrep", None) != "stokes":
        obs = obs.switch_polrep(polrep_out="stokes", allow_singlepol=True)
    return obs


def ensure_stokesI_image(im: eh.image.Image) -> eh.image.Image:
    return im.switch_polrep(polrep_out="stokes", pol_prim_out="I")


def pop_size_simplex_lattice(H: int, m: int) -> int:
    return math.comb(H + m - 1, m - 1)


def choose_representative_from_pareto(fits: np.ndarray) -> int:
    if fits.ndim != 2 or fits.shape[0] == 0:
        return 0
    return int(np.argmin(np.linalg.norm(fits, axis=1)))


def make_gaussian_prior(obs,
                        npix: int,
                        fov_uas: float,
                        flux_jy: float,
                        fwhm_uas: float | None = None,
                        maj_uas: float | None = None,
                        min_uas: float | None = None,
                        pa_deg: float = 0.0):
    """
    Build a Stokes-I Gaussian prior directly in memory.

    FOV is total image width in uas.
    Gaussian sizes are FWHM in uas.
    """
    fov_rad = float(fov_uas) * eh.RADPERUAS

    prior = eh.image.make_square(obs, npix, fov_rad)
    prior = ensure_stokesI_image(prior)

    prior.rf = float(obs.rf)
    prior.ra = obs.ra
    prior.dec = obs.dec

    pa_rad = float(pa_deg) * np.pi / 180.0

    if fwhm_uas is not None:
        major = float(fwhm_uas) * eh.RADPERUAS
        minor = float(fwhm_uas) * eh.RADPERUAS
    else:
        if maj_uas is None or min_uas is None:
            raise ValueError("For elliptical Gaussian prior, provide --prior-maj-uas and --prior-min-uas")
        major = float(maj_uas) * eh.RADPERUAS
        minor = float(min_uas) * eh.RADPERUAS

    prior = prior.add_gauss(
        float(flux_jy),
        (
            major,
            minor,
            pa_rad,
            0.0,
            0.0,
        ),
    )

    prior.imvec = np.maximum(prior.imvec, 1e-12)
    return prior


def load_eps_prior_vector(path: str, npix: int) -> np.ndarray:
    """
    Return epsilon as full vector of length npix^2.
    Supports ndarray 2D, vector n/n-1, OptimizeResult/dict with x.
    """
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        obj = obj.item()

    n = npix * npix

    if hasattr(obj, "screen"):
        scr = np.asarray(obj.screen)
        if scr.ndim != 2 or scr.shape[0] != scr.shape[1]:
            raise ValueError(f"Epsilon .screen is not square: {scr.shape}")
        if scr.shape[0] > npix:
            N = scr.shape[0]
            i0 = (N - npix) // 2
            scr = scr[i0:i0+npix, i0:i0+npix]
        if scr.shape[0] != npix:
            raise ValueError(f"Epsilon .screen npix={scr.shape[0]} but requested npix={npix}")
        return scr.ravel()

    if hasattr(obj, "x"):
        x = np.asarray(obj.x).ravel()
    elif isinstance(obj, dict) and "x" in obj:
        x = np.asarray(obj["x"]).ravel()
    else:
        x = np.asarray(obj).ravel()

    if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[0] == obj.shape[1]:
        scr = np.asarray(obj)
        if scr.shape[0] > npix:
            N = scr.shape[0]
            i0 = (N - npix) // 2
            scr = scr[i0:i0+npix, i0:i0+npix]
        if scr.shape[0] != npix:
            raise ValueError(f"Epsilon 2D npix={scr.shape[0]} but requested npix={npix}")
        return scr.ravel()

    if x.size == (2 * n - 1):
        eps = x[n:]
        return np.append(eps, 0.0)

    if x.size == n:
        return x
    if x.size == (n - 1):
        return np.append(x, 0.0)

    raise ValueError(f"Unexpected epsilon size: {x.size} (n={n})")


def make_phase_screen(sm: eh.scattering.ScatteringModel,
                      eps_vec: np.ndarray,
                      npix: int,
                      ref_img: eh.image.Image) -> eh.image.Image:
    eps_screen = eh.scattering.MakeEpsilonScreenFromList(eps_vec, npix)
    return sm.MakePhaseScreen(eps_screen, ref_img)


def save_panel_figure(outpath: str,
                      img_unscattered_reco: eh.image.Image,
                      img_scattered_sim: eh.image.Image | None,
                      img_scattered_reco: eh.image.Image,
                      screen_sim_phase: eh.image.Image | None,
                      screen_reco_phase: eh.image.Image,
                      title_prefix: str = "") -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax = axes.ravel()

    ax[0].axis("off")
    ax[0].set_title(f"{title_prefix}")

    img_unscattered_reco.display(axis=ax[1], has_cbar=False, show=False)
    ax[1].set_title("Reco (unscattered)")

    if img_scattered_sim is not None:
        img_scattered_sim.display(axis=ax[2], has_cbar=False, show=False)
        ax[2].set_title("Sim (scattered)")
    else:
        ax[2].axis("off")
        ax[2].set_title("Sim (scattered) [N/A]")

    img_scattered_reco.display(axis=ax[3], has_cbar=False, show=False)
    ax[3].set_title("Reco (scattered)")

    if screen_sim_phase is not None:
        screen_sim_phase.display(axis=ax[4], has_cbar=False, show=False)
        ax[4].set_title("Screen (sim)")
    else:
        ax[4].axis("off")
        ax[4].set_title("Screen (sim) [N/A]")

    screen_reco_phase.display(axis=ax[5], has_cbar=False, show=False)
    ax[5].set_title("Screen (reco)")

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlabel("")
        a.set_ylabel("")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Optimizers
# -----------------------------
def run_pso(dictionary: dict,
            udp: pg.problem,
            scatteringfit,
            prior: eh.image.Image,
            obs_sc: eh.obsdata.Obsdata,
            eps_prior_nminus1: np.ndarray,
            outdir: str) -> object:
    scipy_option = {
        "maxiter": 600,
        "ftol": 1e-4,
        "maxcor": 10,
        "gtol": 1e-3,
        "maxls": 100,
        "disp": False,
    }

    dictionary_scatt = solver.read_config_params(dictionary["config_file"], "Scattering")
    dictionary["pop_size"] = dictionary_scatt.get("pop_size", dictionary.get("pop_size", 64))

    dictionary["max_weight"] = float(dictionary.get("max_weight", 1000.0))
    dictionary["snr"] = float(dictionary.get("snr", 1e-4))
    dictionary["minimization_algorithm"] = dictionary.get("minimization_algorithm", "L-BFGS-B")
    dictionary["parallel"] = bool(dictionary.get("parallel", True))
    dictionary["logweights"] = bool(dictionary.get("logweights", False))
    dictionary["logim"] = bool(dictionary.get("logim", False))

    bd = 3.0
    if dictionary["logweights"]:
        dictionary["lower_bounds"] = -bd
        dictionary["upper_bounds"] = bd
    else:
        dictionary["lower_bounds"] = dictionary["snr"]
        dictionary["upper_bounds"] = dictionary["max_weight"]

    dictionary["neighbours"] = int(dictionary.get("neighbours", 1))
    dictionary["generations"] = int(dictionary.get("generations", 100))

    use_gradient = True
    res = obs_sc.res()

    scattering_args = {"prior_screen": eps_prior_nminus1}

    cooperative_game = CooperativeGame(
        dictionary,
        udp,
        scatteringfit,
        "pyswarms",
        scipy_option,
        prior,
        use_gradient,
        mode=dictionary["mode"],
        res=res,
        scattering_args=scattering_args,
    )

    best_fitness, best_position = cooperative_game.cooperative_particles()
    results = cooperative_game._optimize_shapley(best_position, 0)

    try:
        plot_cost_history(cost_history=cooperative_game.optimizer.cost_history)
        plt.savefig(f"{outdir}/cost_history.pdf", bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    return results


def run_moead(dictionary: dict,
              udp: pg.problem,
              x_init: np.ndarray,
              outdir: str,
              H: int,
              neighbours_frac: float,
              jitter: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nf = udp.get_nf()
    if nf < 2:
        raise ValueError(
            f"MOEA/D requires multiobjective (nf>=2). Here nf={nf}. "
            "Check that Scattering returns multiple objectives in mode='pareto'."
        )

    gen = int(dictionary.get("generations", 200))
    seed_init = int(dictionary.get("seed_initial", 0))
    decomp_method = dictionary.get("decomposition_method", "grid")
    decomp_seed = int(dictionary.get("decomposition_seed", 0))

    pop_size = pop_size_simplex_lattice(H=H, m=nf)
    pop = pg.population(udp, size=pop_size, seed=seed_init)

    weights = pg.decomposition_weights(n_f=nf, n_w=len(pop), method=decomp_method, seed=decomp_seed)
    neighbours = max(2, int(neighbours_frac * pop_size))

    algo = pg.algorithm(
        pg.moead(
            gen=gen,
            neighbours=neighbours,
            decomposition="weighted",
            weight_generation=decomp_method,
            seed=decomp_seed,
        )
    )
    algo.set_verbosity(1)

    x0 = np.asarray(x_init, dtype=float)
    rng0 = np.random.default_rng(seed_init)
    for i in range(len(pop)):
        xi = x0.copy()
        if jitter > 0:
            xi = xi + jitter * rng0.standard_normal(size=xi.size)
        pop.set_x(i, xi)

    pop = algo.evolve(pop)

    fits = pop.get_f()
    vectors = pop.get_x()

    np.savetxt(f"{outdir}/moead_fits.txt", fits)
    np.savetxt(f"{outdir}/moead_vectors.txt", vectors)
    np.savetxt(f"{outdir}/moead_weights.txt", np.asarray(weights))

    return fits, vectors, np.asarray(weights)


# -----------------------------
# Paths
# -----------------------------
def stage_paths(outdir: str, npix: int) -> dict:
    return {
        "unscattered_fits": os.path.join(outdir, f"reco_unscattered_{npix}.fits"),
        "unscattered_pdf": os.path.join(outdir, "reco_unscattered.pdf"),
        "screen_fits": os.path.join(outdir, f"screen_reco_phase_{npix}.fits"),
        "screen_pdf": os.path.join(outdir, "screen_reco_phase.pdf"),
        "scattered_fits": os.path.join(outdir, f"reco_scattered_{npix}.fits"),
        "scattered_pdf": os.path.join(outdir, "reco_scattered.pdf"),
    }


# -----------------------------
# Main imaging stage
# -----------------------------
def run_scattering_imaging(args, npix: int) -> None:
    n = npix * npix
    paths = stage_paths(args.outdir, npix)

    dictionary = solver.read_config_params(args.config, "Scattering")
    dictionary["uvf"] = args.uvfits
    dictionary["img"] = args.x0fits
    dictionary["config_file"] = args.config

    if args.moead_gen is not None:
        dictionary["generations"] = int(args.moead_gen)

    # -----------------------------
    # Load data
    # -----------------------------
    obs = eh.obsdata.load_uvfits(dictionary["uvf"], polrep="stokes")
    obs = ensure_stokesI_obs(obs)

    if args.select_band:
        obs_sc = select_band(obs, nu0_ghz=args.nu0_ghz, halfwidth_ghz=args.halfwidth_ghz)
    else:
        obs_sc = obs.copy()

    rf = float(obs_sc.rf)

    # -----------------------------
    # Load/build prior image
    # -----------------------------
    if args.prior_mode == "fits":
        if not dictionary.get("img"):
            raise ValueError("With --prior-mode fits you must provide --x0fits")
        iimg = ensure_stokesI_image(eh.image.load_fits(dictionary["img"]))
        fov = float(iimg.fovx())
        prior = ensure_stokesI_image(iimg.regrid_image(fov, npix))
        fov_uas_used = fov / eh.RADPERUAS
    else:
        if args.prior_fwhm_uas is not None:
            scale_uas = float(args.prior_fwhm_uas)
        else:
            scale_uas = max(float(args.prior_maj_uas or 0.0), float(args.prior_min_uas or 0.0))

        if scale_uas <= 0:
            raise ValueError(
                "For --prior-mode gauss, provide either --prior-fwhm-uas "
                "or both --prior-maj-uas and --prior-min-uas"
            )

        fov_uas_used = float(args.prior_fov_uas) if args.prior_fov_uas is not None else float(args.prior_fov_mult) * scale_uas
        flux_jy = float(args.prior_flux if args.prior_flux is not None else dictionary.get("zbl", 1.0))

        prior = make_gaussian_prior(
            obs=obs_sc,
            npix=npix,
            fov_uas=fov_uas_used,
            flux_jy=flux_jy,
            fwhm_uas=args.prior_fwhm_uas,
            maj_uas=args.prior_maj_uas,
            min_uas=args.prior_min_uas,
            pa_deg=args.prior_pa_deg,
        )

    prior.rf = rf
    prior.ra = obs_sc.ra
    prior.dec = obs_sc.dec
    prior.imvec[prior.imvec < 1e-7] = 1e-7

    prior.display(export_pdf=f"{args.outdir}/prior.pdf", show=False)
    prior.save_fits(f"{args.outdir}/prior_generated_{npix}.fits")

    with open(f"{args.outdir}/prior_info.json", "w") as f:
        json.dump({
            "prior_mode": args.prior_mode,
            "prior_flux_jy": float(args.prior_flux) if args.prior_flux is not None else None,
            "prior_fwhm_uas": float(args.prior_fwhm_uas) if args.prior_fwhm_uas is not None else None,
            "prior_maj_uas": float(args.prior_maj_uas) if args.prior_maj_uas is not None else None,
            "prior_min_uas": float(args.prior_min_uas) if args.prior_min_uas is not None else None,
            "prior_pa_deg": float(args.prior_pa_deg),
            "prior_fov_uas_used": float(fov_uas_used),
            "npix": int(npix),
        }, f, indent=2)

    if args.write_prior_only:
        print(f"[PRIOR] Wrote prior only to {args.outdir}/prior_generated_{npix}.fits")
        return

    # -----------------------------
    # Problem terms
    # -----------------------------
    reg_term = dictionary["reg_term"]
    data_term = dictionary["data_term"]
    reg_term["epsilon"] = 1

    # -----------------------------
    # Scattering model
    # -----------------------------
    sm = eh.scattering.ScatteringModel(r_in=float(args.scattering_rin))

    # -----------------------------
    # Epsilon prior
    # -----------------------------
    if args.eps_prior == "truth":
        if not args.eps_npy:
            raise ValueError("eps-prior=truth requires --eps-npy")
        eps_sim_vec = load_eps_prior_vector(args.eps_npy, npix=npix)
        have_eps_sim = True
    elif args.eps_prior == "random":
        eps_obj = eh.scattering.stochastic_optics.MakeEpsilonScreen(npix, npix, rngseed=args.eps_seed)
        eps_sim_vec = np.asarray(eps_obj.screen if hasattr(eps_obj, "screen") else eps_obj).ravel()
        have_eps_sim = True
    else:
        eps_sim_vec = np.zeros(n, dtype=np.complex128)
        have_eps_sim = False

    screen_sim_phase = None
    if have_eps_sim:
        screen_sim_phase = make_phase_screen(sm, eps_sim_vec, npix=npix, ref_img=prior)
        screen_sim_phase.display(export_pdf=f"{args.outdir}/screen_sim_phase.pdf", show=False)

    rescaling = [1.0, 1.0]
    if screen_sim_phase is not None:
        try:
            rescaling[1] = float(np.max(np.abs(screen_sim_phase.imvec)))
            if rescaling[1] <= 0:
                rescaling[1] = 1.0
        except Exception:
            rescaling[1] = 1.0
    else:
        rescaling[1] = 1.0

    # dimension = image(n) + epsilon(n-1)
    dim = 2 * n - 1
    eps_prior_nminus1 = np.asarray(eps_sim_vec).ravel()[:-1]

    # -----------------------------
    # Build problem + optimize
    # -----------------------------
    if args.optimizer == "pso":
        dictionary["mode"] = "shapley"

        scatteringfit = Scattering.Scattering(
            obs_sc,
            prior,
            data_term,
            reg_term,
            rescaling,
            float(dictionary.get("zbl", prior.total_flux())),
            dim,
            ttype=args.ttype,
            mode=dictionary["mode"],
            epsilon_screen_prior=eps_prior_nminus1,
        )
        scatteringfit.setFit()
        udp = pg.problem(scatteringfit)

        results = run_pso(
            dictionary=dictionary,
            udp=udp,
            scatteringfit=scatteringfit,
            prior=prior,
            obs_sc=obs_sc,
            eps_prior_nminus1=eps_prior_nminus1,
            outdir=args.outdir,
        )

        np.save(f"{args.outdir}/results_{npix}.npy", results)
        x_best = np.asarray(results.x, dtype=float).ravel()

    else:
        dictionary["mode"] = "pareto"

        scatteringfit = Scattering.Scattering(
            obs_sc,
            prior,
            data_term,
            reg_term,
            rescaling,
            float(dictionary.get("zbl", prior.total_flux())),
            dim,
            ttype=args.ttype,
            mode=dictionary["mode"],
            epsilon_screen_prior=eps_prior_nminus1,
        )
        scatteringfit.setFit()
        udp = pg.problem(scatteringfit)

        rescaling0 = float(np.max(prior.imvec)) if np.max(prior.imvec) > 0 else 1.0
        x_init = np.concatenate([
            np.asarray(prior.imvec, dtype=float) / rescaling0,
            np.real(np.asarray(eps_prior_nminus1)).astype(float) / float(rescaling[1]),
        ])

        fits, vectors, weights = run_moead(
            dictionary=dictionary,
            udp=udp,
            x_init=x_init,
            outdir=args.outdir,
            H=int(args.moead_H),
            neighbours_frac=float(args.moead_neigh_frac),
            jitter=float(args.moead_jitter),
        )

        idx = choose_representative_from_pareto(fits)
        x_best = np.asarray(vectors[idx], dtype=float).ravel()
        np.savetxt(f"{args.outdir}/moead_selected_idx.txt", np.array([idx], dtype=int))
                # -------------------------------------------------
        # Save all decoded epsilon/screens for postprocessing
        # -------------------------------------------------
        screens_all = []
        eps_all = []
        imgs_all = []

        for i in range(vectors.shape[0]):
            xi = np.asarray(vectors[i], dtype=float).ravel()

            # decode image
            img_i = ensure_stokesI_image(prior.copy())
            img_i.imvec = np.asarray(xi[:n], dtype=float)
            img_i.imvec = np.maximum(img_i.imvec, 0.0)

            # decode epsilon
            eps_i = np.append(np.asarray(xi[n:], dtype=float), 0.0)

            # build phase screen with EXACT same pipeline logic
            screen_i = make_phase_screen(sm, eps_i, npix=npix, ref_img=img_i)

            imgs_all.append(np.asarray(img_i.imvec, dtype=float))
            eps_all.append(np.asarray(eps_i, dtype=float))
            screens_all.append(np.asarray(screen_i.imvec, dtype=float))

        imgs_all = np.asarray(imgs_all, dtype=float)
        eps_all = np.asarray(eps_all, dtype=float)
        screens_all = np.asarray(screens_all, dtype=float)

        np.save(os.path.join(args.outdir, "moead_imgs_decoded.npy"), imgs_all)
        np.save(os.path.join(args.outdir, "moead_eps_decoded.npy"), eps_all)
        np.save(os.path.join(args.outdir, "moead_phase_screens_decoded.npy"), screens_all)

    # -----------------------------
    # Decode selected solution
    # -----------------------------
    if x_best.size < dim:
        raise ValueError(f"Best vector has size {x_best.size}, expected at least {dim}")

    img_reco = ensure_stokesI_image(prior.copy())
    img_reco.imvec = np.asarray(x_best[:n], dtype=float)
    img_reco.imvec = np.maximum(img_reco.imvec, 0.0)
    img_reco.display(export_pdf=paths["unscattered_pdf"], show=False)
    img_reco.save_fits(paths["unscattered_fits"])

    eps_reco = np.append(np.asarray(x_best[n:], dtype=float), 0.0)
    screen_reco_phase = make_phase_screen(sm, eps_reco, npix=npix, ref_img=img_reco)
    screen_reco_phase.display(export_pdf=paths["screen_pdf"], show=False)
    screen_reco_phase.save_fits(paths["screen_fits"])

    eps_reco_screen = eh.scattering.MakeEpsilonScreenFromList(eps_reco, npix)
    img_scattered_reco = sm.Scatter(img_reco, Epsilon_Screen=eps_reco_screen)
    img_scattered_reco.display(export_pdf=paths["scattered_pdf"], show=False)
    img_scattered_reco.save_fits(paths["scattered_fits"])

    img_scattered_sim = None
    if have_eps_sim:
        eps_sim_screen = eh.scattering.MakeEpsilonScreenFromList(eps_sim_vec, npix)
        img_scattered_sim = sm.Scatter(prior, Epsilon_Screen=eps_sim_screen)
        img_scattered_sim.display(export_pdf=f"{args.outdir}/sim_scattered.pdf", show=False)

    nu_ghz = rf / 1e9
    save_panel_figure(
        outpath=f"{args.outdir}/compare_sources_screens.pdf",
        img_unscattered_reco=img_reco,
        img_scattered_sim=img_scattered_sim,
        img_scattered_reco=img_scattered_reco,
        screen_sim_phase=screen_sim_phase,
        screen_reco_phase=screen_reco_phase,
        title_prefix=f"{args.optimizer.upper()} | {nu_ghz:.3f} GHz | npix={npix}",
    )

    json.dump(dictionary, open(f"{args.outdir}/config_used.json", "w"), indent=2)
    print(f"[OK] Done. Outputs in: {args.outdir}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    epilog = r"""
EXAMPLES

MOEA/D with isotropic Gaussian prior:
  python scattering_gauss_pipeline.py --config imaging.config \
    --uvfits data/scattered.uvfits \
    --outdir results_moead --npix 65 --optimizer moead \
    --prior-mode gauss --prior-fwhm-uas 2255.532 --prior-flux 1.05 \
    --eps-prior random --eps-seed 42

MOEA/D with anisotropic Gaussian prior:
  python scattering_gauss_pipeline.py --config imaging.config \
    --uvfits data/scattered.uvfits \
    --outdir results_moead --npix 65 --optimizer moead \
    --prior-mode gauss --prior-maj-uas 1380 --prior-min-uas 703 \
    --prior-pa-deg 81.9 --prior-flux 1.05 \
    --eps-prior random --eps-seed 42
"""
    p = argparse.ArgumentParser(
        prog="scattering_gauss_pipeline.py",
        description="Scattering imaging with Gaussian or FITS prior, using PSO or MOEA/D.",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("--config", required=True, help="Config file (Scattering section)")
    p.add_argument("--uvfits", required=True, help="Scattered UVFITS")
    p.add_argument("--x0fits", default=None, help="FITS prior/initial image")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--npix", type=int, default=65, help="Reconstruction npix (forced odd)")
    p.add_argument("--threads", type=int, default=None, help="BLAS threads")

    p.add_argument("--optimizer", choices=["pso", "moead"], default="moead")

    # Prior
    p.add_argument("--prior-mode", choices=["fits", "gauss"], default="gauss",
                   help="Use an input FITS prior or build a Gaussian prior on the fly")
    p.add_argument("--prior-fwhm-uas", type=float, default=None,
                   help="Isotropic Gaussian FWHM in microarcseconds")
    p.add_argument("--prior-maj-uas", type=float, default=None,
                   help="Elliptical Gaussian major-axis FWHM in microarcseconds")
    p.add_argument("--prior-min-uas", type=float, default=None,
                   help="Elliptical Gaussian minor-axis FWHM in microarcseconds")
    p.add_argument("--prior-pa-deg", type=float, default=0.0,
                   help="Elliptical Gaussian position angle in degrees")
    p.add_argument("--prior-flux", type=float, default=None,
                   help="Total flux in Jy; if omitted, use zbl or 1.0")
    p.add_argument("--prior-fov-mult", type=float, default=8.0,
                   help="Total image FOV = prior_fov_mult * characteristic Gaussian size")
    p.add_argument("--prior-fov-uas", type=float, default=None,
                   help="Override total image FOV directly in microarcseconds")
    p.add_argument("--write-prior-only", action="store_true",
                   help="Only generate/save the prior FITS and exit")

    # Band selection
    p.add_argument("--select-band", action="store_true", help="Filter rows around center frequency")
    p.add_argument("--nu0-ghz", type=float, default=230.0)
    p.add_argument("--halfwidth-ghz", type=float, default=2.0)

    # Scattering controls
    p.add_argument("--eps-prior", choices=["truth", "random", "zero"], default="random")
    p.add_argument("--eps-seed", type=int, default=42)
    p.add_argument("--eps-npy", default=None, help="Truth epsilon .npy if eps-prior=truth")
    p.add_argument("--scattering-rin", type=float, default=8.0e7,
                   help="Inner scale r_in passed to eh.scattering.ScatteringModel")

    # objective / fft
    p.add_argument("--ttype", choices=["fast", "nfft", "direct"], default="fast")

    # MOEA/D
    p.add_argument("--moead-H", type=int, default=3)
    p.add_argument("--moead-neigh-frac", type=float, default=0.15)
    p.add_argument("--moead-jitter", type=float, default=0.0)
    p.add_argument("--moead-gen", type=int, default=None,
                   help="MOEA/D generations override")

    return p.parse_args()


def main():
    args = parse_args()
    set_thread_env(args.threads)
    ensure_dir(args.outdir)

    npix = ensure_odd(int(args.npix))
    run_scattering_imaging(args, npix)


if __name__ == "__main__":
    main()
