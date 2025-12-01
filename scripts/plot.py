#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified script for MACE model inference, analysis, and plotting.
VERSION: FINAL ROBUST (Jitter KDE + NaN Fix + Complete Info Flush)

Updates:
1. ROBUST KDE: Added jitter and NaN filtering to prevent Gaussian_KDE failures on singular data.
2. VISIBILITY FIX: Ensures scatter plot is visible even if KDE fails (fallback to solid color).
3. LAYOUT: GridSpec layout adjusted for histograms, colorbar, and legend.
4. OUTPUT: Forces stdout flush to ensure all info (E0s, Metrics) is printed.
"""

import numpy as np
import matplotlib
# Use 'Agg' backend to prevent errors on headless servers
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from ase.io import iread
from ase import Atoms
from ase.data import chemical_symbols
import torch
from tqdm import tqdm
import argparse
import glob
import os
import sys
from typing import List, Generator
from concurrent.futures import ProcessPoolExecutor
import functools

# --- ML & Stats Imports ---
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# --- Imports from MACE ---
from torch.serialization import add_safe_globals
from mace.modules.models import ScaleShiftMACE
from e3nn import o3
from mace import data
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, utils

# --- PyTorch Safe Loading ---
add_safe_globals([ScaleShiftMACE])
add_safe_globals([slice])

# --- Torch Scatter Import with Fallback ---
try:
    from torch_scatter import scatter_mean
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False
    print("[Warning] 'torch_scatter' not found. Using NumPy fallback (slower).")


# =============================================================================
# --- Helper Functions ---
# =============================================================================

def read_chunks(filename: str, chunk_size: int) -> Generator[List[Atoms], None, None]:
    """Reads ASE file in chunks."""
    chunk = []
    iterator = iread(filename)
    while True:
        try:
            for _ in range(chunk_size):
                chunk.append(next(iterator))
            yield chunk
            chunk = []
        except StopIteration:
            if chunk:
                yield chunk
            break
        except Exception as e:
            print(f"Error reading file: {e}")
            break

def _process_single_atom_config(image: Atoms, head_name: str):
    """Worker: Atoms -> Config."""
    try:
        # --- 修改开始 ---
        # 1. 定义映射关系： 'Model_Expected_Key': 'Atoms_Info_Key'
        # 如果你的 xyz 文件里 charge 叫 'Q', 这里就写 'charge': 'Q'
        info_map = {"total_charge": "charge", "total_spin": "spin"}
        
        # 2. 传入 info_keys
        keyspec = data.KeySpecification(
            info_keys=info_map, 
            arrays_keys={"charges": "Qs"} # 保持原有的原子电荷映射
        )
        # --- 修改结束 ---
        
        config = data.config_from_atoms(image, key_specification=keyspec, head_name=[head_name])
        return config, None
    except Exception as e:
        return None, str(e)

def create_batches_parallel(images: List[Atoms], max_edges: int, model, head_name="Default", workers=4):
    """Parallel preprocessing of Atoms into AtomicData batches."""
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cutoff = float(model.r_max.cpu())
    
    configs_list = []
    if workers > 1 and len(images) > 50:
        func = functools.partial(_process_single_atom_config, head_name=head_name)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(func, images))
        for (conf, err), img in zip(results, images):
            if conf: configs_list.append((conf, img))
    else:
        for img in images:
            conf, err = _process_single_atom_config(img, head_name)
            if conf: configs_list.append((conf, img))

    data_list, img_list, edge_counts = [], [], []
    for conf, img in configs_list:
        try:
            ad = data.AtomicData.from_config(conf, z_table=z_table, cutoff=cutoff, heads=[head_name])
            data_list.append(ad); img_list.append(img); edge_counts.append(ad.edge_index.shape[1])
        except: pass

    batches_data, batches_img = [], []
    curr_d, curr_i, curr_e = [], [], 0
    for ad, img, e in zip(data_list, img_list, edge_counts):
        if curr_e + e <= max_edges:
            curr_d.append(ad); curr_i.append(img); curr_e += e
        else:
            if curr_d: batches_data.append(curr_d); batches_img.append(curr_i)
            curr_d = [ad]; curr_i = [img]; curr_e = e
    if curr_d: batches_data.append(curr_d); batches_img.append(curr_i)
    return batches_data, batches_img

def calculate_metrics(true, pred):
    # Filter NaNs for robustness
    mask = np.isfinite(true) & np.isfinite(pred)
    if not np.any(mask):
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    
    t, p = true[mask], pred[mask]
    mae = np.mean(np.abs(t - p))
    rmse = np.sqrt(np.mean((t - p) ** 2))
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else float('nan')
    return {"mae": mae, "rmse": rmse, "r2": r2}

def fit_e0_and_get_binding_energy(total_energies, atom_counts):
    """Fits E = Sum(N_i * E0_i). Returns binding energies and E0 coefs."""
    reg = LinearRegression(fit_intercept=False)
    reg.fit(atom_counts, total_energies)
    e0_values = reg.coef_
    e_ref = reg.predict(atom_counts)
    binding_energies = total_energies - e_ref
    return binding_energies, e0_values

# =============================================================================
# --- Main Script ---
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MACE Analysis Script (Final Robust)")
    parser.add_argument("--mode", required=True, choices=['run', 'plot', 'collate'])
    parser.add_argument("--input", type=str)
    parser.add_argument("--output-prefix", type=str, required=True)
    
    parser.add_argument('--model', type=str, default="mace.model")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--dtype', type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument('--num-workers', type=int, default=4)
    
    parser.add_argument('--max-edges', type=int, default=15000)
    parser.add_argument('--chunk-size', type=int, default=50000)
    
    parser.add_argument('--skip-forces', action='store_true', help='Skip force calculation.')
    parser.add_argument('--compute-stress', action='store_true')
    parser.add_argument('--layers-to-keep', type=int, default=-1)
    
    parser.add_argument('--compute-binding-energy', action='store_true', help='Fit E0 and plot Binding Energy.')
    parser.add_argument('--plot-pca', action='store_true', help='Plot PCA maps.')
    parser.add_argument('--plot-components', action='store_true', help='Plot separate force/stress components.')
    parser.add_argument('--use-quantiles', action='store_true', help='Filter outliers (0.1%%-99.9%%).')

    args = parser.parse_args()

    # =========================================================================
    # --- MODE: RUN ---
    # =========================================================================
    if args.mode == 'run':
        if not args.input: parser.error("Run mode needs --input")
        
        existing_chunks = glob.glob(f"{args.output_prefix}_chunk_*_data.npz")
        if existing_chunks:
            print(f"[Warning] Found {len(existing_chunks)} existing chunk files. Ensure no mixing!")
        
        print(f"Loading model: {args.model}")
        model = torch.load(args.model, map_location=args.device)
        model = model.float() if args.dtype == "float32" else model.double()
        model.to(args.device).eval()
        
        model_z_table = [int(z) for z in model.atomic_numbers]
        print(f"Model Elements (Z): {model_z_table}")

        try:
            irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
            l_max = irreps_out.lmax
            num_inv_feats = irreps_out.dim // (l_max + 1) ** 2
            num_layers = int(model.num_interactions)
            keep = args.layers_to_keep if args.layers_to_keep != -1 else num_layers
            layer_dims = [irreps_out.dim for _ in range(num_layers)]
            layer_dims[-1] = num_inv_feats 
            feat_dim = np.sum(layer_dims[:keep])
            print(f"Descriptor: Keeping {keep}/{num_layers} layers. Dim: {feat_dim}")
        except:
            print("Warning: Auto-descriptor sizing failed. Extracting all.")
            feat_dim = None
            l_max = 0 
            num_layers = int(model.num_interactions)
            num_inv_feats = 0

        chunks = read_chunks(args.input, args.chunk_size)
        for i, chunk in enumerate(chunks):
            print(f"Processing Chunk {i} ({len(chunk)} atoms in memory)...")
            
            batches_d, batches_i = create_batches_parallel(chunk, args.max_edges, model, "Default", args.num_workers)
            
            res = {k: [] for k in ["dft_e", "mlp_e", "dft_f", "mlp_f", "dft_s", "mlp_s", "desc", "counts", "natoms"]}
            comp_force = not args.skip_forces
            comp_stress = args.compute_stress

            for bd, bi in tqdm(zip(batches_d, batches_i), desc=f"Infer Chunk {i}", total=len(batches_d)):
                natoms = [len(a) for a in bi]
                counts = [[sum(a.numbers == z) for z in model_z_table] for a in bi]
                
                res["natoms"].extend(natoms)
                res["counts"].extend(counts)
                res["dft_e"].extend([a.get_potential_energy()/n for a, n in zip(bi, natoms)])
                
                if comp_force: res["dft_f"].extend([a.get_forces() for a in bi])
                if comp_stress: 
                    try: res["dft_s"].extend([a.get_stress(voigt=False) for a in bi])
                    except: comp_stress = False

                loader = torch_geometric.dataloader.DataLoader(dataset=bd, batch_size=len(bd), shuffle=False)
                batch = next(iter(loader)).to(args.device)
                
                grad_needed = (comp_force or comp_stress)
                with torch.set_grad_enabled(grad_needed):
                    out = model(batch.to_dict(), compute_force=comp_force, compute_stress=comp_stress)
                    
                    energies = out['energy'].detach().cpu().numpy()
                    res["mlp_e"].extend([e/n for e, n in zip(energies, natoms)])
                    
                    if comp_force:
                        f_pred = out['forces'].detach().cpu().numpy()
                        ptr = batch.ptr.cpu().numpy()
                        for j in range(len(ptr)-1): res["mlp_f"].append(f_pred[ptr[j]:ptr[j+1]])
                    
                    if comp_stress: 
                        res["mlp_s"].extend(out['stress'].detach().cpu().numpy())

                    node_feats = out['node_feats'].detach()
                    invs = extract_invariant(node_feats, num_layers=num_layers, num_features=num_inv_feats, l_max=l_max)
                    if feat_dim: invs = invs[:, :feat_dim]
                    
                    if HAS_TORCH_SCATTER:
                        pooled = scatter_mean(invs, batch.batch, dim=0).cpu().numpy()
                    else:
                        grp = batch.batch.cpu().numpy()
                        invs_np = invs.cpu().numpy()
                        pooled = np.array([invs_np[grp==k].mean(0) for k in range(len(bi))])
                    res["desc"].extend(pooled)

            np_save = {
                "dft_energies": np.array(res["dft_e"]),
                "mlp_energies": np.array(res["mlp_e"]),
                "mlp_descriptors": np.array(res["desc"]),
                "atom_counts": np.array(res["counts"], dtype=int),
                "num_atoms": np.array(res["natoms"], dtype=int),
                "z_numbers": np.array(model_z_table, dtype=int) 
            }
            if comp_force and res["mlp_f"]:
                np_save["dft_forces"] = np.concatenate(res["dft_f"])
                np_save["mlp_forces"] = np.concatenate(res["mlp_f"])
            if comp_stress and res["mlp_s"]:
                np_save["dft_stresses"] = np.array(res["dft_s"])
                np_save["mlp_stresses"] = np.array(res["mlp_s"])

            np.savez_compressed(f"{args.output_prefix}_chunk_{i}_data.npz", **np_save)
        print("Run complete. Data saved.")

    # =========================================================================
    # --- MODE: PLOT ---
    # =========================================================================
    elif args.mode == 'plot':
        files = sorted(glob.glob(f"{args.output_prefix}_chunk_*_data.npz"))
        if not files: print("No data found."); return

        print(f"Loading {len(files)} chunks...")
        d_e, m_e, descs, counts, natoms = [], [], [], [], []
        d_f, m_f, d_s, m_s = [], [], [], []
        z_numbers = None
        has_f, has_s = True, True

        for f in tqdm(files):
            data = np.load(f)
            d_e.append(data['dft_energies']); m_e.append(data['mlp_energies'])
            descs.append(data['mlp_descriptors'])
            if 'atom_counts' in data: counts.append(data['atom_counts'])
            if 'num_atoms' in data: natoms.append(data['num_atoms'])
            if 'z_numbers' in data and z_numbers is None: z_numbers = data['z_numbers']

            if 'dft_forces' in data and data['dft_forces'].size > 0: 
                d_f.append(data['dft_forces']); m_f.append(data['mlp_forces'])
            else: has_f = False
            
            if 'dft_stresses' in data and data['dft_stresses'].size > 0:
                d_s.append(data['dft_stresses']); m_s.append(data['mlp_stresses'])
            else: has_s = False

        dft_e_pa = np.concatenate(d_e)
        mlp_e_pa = np.concatenate(m_e)
        descriptors = np.concatenate(descs)
        
        print("-" * 50)
        print("DATA LOADED. STARTING ANALYSIS...")
        
        # --- Binding Energy Logic ---
        if args.compute_binding_energy and counts:
            print("\n>>> Computing Binding Energies (Fitting E0)...")
            all_counts = np.concatenate(counts)
            all_natoms = np.concatenate(natoms)
            dft_e_total = dft_e_pa * all_natoms
            
            try:
                dft_bind_total, e0s = fit_e0_and_get_binding_energy(dft_e_total, all_counts)
                dft_bind_pa = dft_bind_total / all_natoms
                
                mlp_e_total = mlp_e_pa * all_natoms
                e_ref = np.dot(all_counts, e0s)
                mlp_bind_total = mlp_e_total - e_ref
                mlp_bind_pa = mlp_bind_total / all_natoms
                
                if z_numbers is not None:
                    counts_sum = np.sum(all_counts, axis=0)
                    present_indices = np.where(counts_sum > 0)[0]
                    e0_str_list = []
                    for idx in present_indices:
                        z = z_numbers[idx]
                        e = e0s[idx]
                        e0_str_list.append(f"{chemical_symbols[z]}: {e:.3f} eV")
                    print(f"Fitted E0s (Dataset present elements):\n  {', '.join(e0_str_list)}")
                else:
                    print(f"Fitted E0s (All): {e0s}")

                plot_dft_e, plot_mlp_e = dft_bind_pa, mlp_bind_pa
                e_label = "Binding Energy (eV/atom)"
            except Exception as e:
                print(f"[Error] E0 fitting failed: {e}. Reverting to Total Energy.")
                plot_dft_e, plot_mlp_e = dft_e_pa, mlp_e_pa
                e_label = "Total Energy (eV/atom)"
                dft_bind_pa = dft_e_pa
        else:
            plot_dft_e, plot_mlp_e = dft_e_pa, mlp_e_pa
            e_label = "Total Energy (eV/atom)"
            dft_bind_pa = dft_e_pa

        # --- Parity Plots ---
        print("\n>>> Generating Parity Plots...")
        met_e = calculate_metrics(plot_dft_e, plot_mlp_e)
        print(f"  [Energy] MAE={met_e['mae']*1e3:.2f} meV/atom, R2={met_e['r2']:.4f}")
        make_parity_plot(plot_dft_e, plot_mlp_e, f"{e_label}", f"MACE {e_label}", f"DFT {e_label}",
                         met_e, f"{args.output_prefix}_energy.png", use_quantiles=args.use_quantiles)

        if has_f and d_f:
            d_f_v, m_f_v = np.concatenate(d_f), np.concatenate(m_f)
            met_f = calculate_metrics(d_f_v.flatten(), m_f_v.flatten())
            print(f"  [Force]  MAE={met_f['mae']*1e3:.2f} meV/A")
            if args.plot_components:
                idx = np.random.choice(len(d_f_v), min(50000, len(d_f_v)), replace=False)
                make_parity_plot(d_f_v[idx], m_f_v[idx], "Force Components", "MACE F", "DFT F", met_f, 
                                 f"{args.output_prefix}_force.png", labels=["x","y","z"], colors=['r','g','b'], use_quantiles=args.use_quantiles)
            else:
                make_parity_plot(np.linalg.norm(d_f_v, axis=1), np.linalg.norm(m_f_v, axis=1), 
                                 "Force Norm", "MACE |F|", "DFT |F|", met_f, f"{args.output_prefix}_force.png", use_quantiles=args.use_quantiles)

        if has_s and d_s:
            d_s_t, m_s_t = np.concatenate(d_s), np.concatenate(m_s)
            met_s = calculate_metrics(d_s_t.flatten(), m_s_t.flatten())
            print(f"  [Stress] MAE={met_s['mae']*1e3:.2f} meV/A^3")
            
            if args.plot_components:
                N = d_s_t.shape[0]
                comp_indices = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
                labels = ["$\sigma_{xx}$", "$\sigma_{yy}$", "$\sigma_{zz}$", "$\sigma_{yz}$", "$\sigma_{xz}$", "$\sigma_{xy}$"]
                colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
                
                d_s_comp = np.zeros((N, 6))
                m_s_comp = np.zeros((N, 6))
                for i, (r, c) in enumerate(comp_indices):
                    d_s_comp[:, i] = d_s_t[:, r, c]
                    m_s_comp[:, i] = m_s_t[:, r, c]
                
                if N > 100000:
                    idx = np.random.choice(N, 100000, replace=False)
                    d_plot, m_plot = d_s_comp[idx], m_s_comp[idx]
                else:
                    d_plot, m_plot = d_s_comp, m_s_comp
                
                make_parity_plot(d_plot, m_plot, "Stress Components", "MACE Stress", "DFT Stress", 
                                 met_s, f"{args.output_prefix}_stress.png", 
                                 labels=labels, colors=colors, use_quantiles=args.use_quantiles)
            else:
                make_parity_plot(d_s_t.flatten(), m_s_t.flatten(), "Stress", "MACE", "DFT", 
                                 met_s, f"{args.output_prefix}_stress.png", use_quantiles=args.use_quantiles)

        # --- PCA Analysis ---
        if args.plot_pca:
            print("\n>>> Running PCA Analysis...")
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(descriptors)
            print(f"  PCA Explained Variance: {pca.explained_variance_ratio_}")
            
            # Plot 1: Density
            fig, ax = plt.subplots(figsize=(7, 6))
            if len(pca_coords) > 100000:
                idx = np.random.choice(len(pca_coords), 100000, replace=False)
                xy = pca_coords[idx]
            else:
                xy = pca_coords
            try:
                # Jitter for PCA Density too
                xy_jitter = xy + np.random.normal(0, 1e-6, xy.shape)
                z = gaussian_kde(xy_jitter.T)(xy_jitter.T)
                sc = ax.scatter(xy[:,0], xy[:,1], c=z, s=5, cmap='viridis', rasterized=True)
                plt.colorbar(sc, label='Density')
            except:
                ax.scatter(xy[:,0], xy[:,1], s=1, alpha=0.5)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
            ax.set_title("Descriptor Space (Density)")
            plt.savefig(f"{args.output_prefix}_pca_density.png", dpi=300); plt.close()
            
            # Plot 2: Energy
            fig, ax = plt.subplots(figsize=(7, 6))
            if args.use_quantiles:
                v_min, v_max = np.quantile(dft_bind_pa, [0.01, 0.99])
            else:
                v_min, v_max = dft_bind_pa.min(), dft_bind_pa.max()
            
            sc = ax.scatter(pca_coords[:,0], pca_coords[:,1], c=dft_bind_pa, 
                            s=5, cmap='plasma', vmin=v_min, vmax=v_max, alpha=0.8, rasterized=True)
            plt.colorbar(sc, label=e_label)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})"); ax.set_ylabel(f"PC2")
            ax.set_title("Descriptor Space (Energy)")
            plt.savefig(f"{args.output_prefix}_pca_energy.png", dpi=300); plt.close()
            print("  PCA plots saved.")

        print("-" * 50)
        print("ALL TASKS COMPLETED.")
        sys.stdout.flush() # Force print buffer flush

    # =========================================================================
    # --- MODE: COLLATE ---
    # =========================================================================
    elif args.mode == 'collate':
        files = sorted(glob.glob(f"{args.output_prefix}_chunk_*_data.npz"))
        collated = {}
        keys = ["dft_energies", "mlp_energies", "mlp_descriptors", "atom_counts", "num_atoms", "dft_forces", "mlp_forces", "dft_stresses"]
        for f in tqdm(files):
            d = np.load(f)
            for k in keys:
                if k in d:
                    if k not in collated: collated[k] = []
                    collated[k].append(d[k])
        for k in collated: collated[k] = np.concatenate(collated[k], axis=0)
        np.savez_compressed(f"{args.output_prefix}_collated.npz", **collated)
        print("Collation complete.")

def make_parity_plot(x, y, title, xlabel, ylabel, metrics, filename, labels=None, colors=None, use_quantiles=False):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], 
                           left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # Sanitize Data (Remove NaNs)
    x_flat, y_flat = x.flatten(), y.flatten()
    mask = np.isfinite(x_flat) & np.isfinite(y_flat)
    x_safe, y_safe = x_flat[mask], y_flat[mask]
    
    if len(x_safe) == 0:
        print(f"  [Error] No valid data for {title}. Skipping.")
        return

    # Limits Calculation
    if use_quantiles:
        vmin, vmax = np.quantile(np.concatenate([x_safe, y_safe]), [0.005, 0.995])
    else:
        vmin, vmax = np.min([x_safe.min(), y_safe.min()]), np.max([x_safe.max(), y_safe.max()])
    
    # Prevent Singular Limits
    if np.abs(vmax - vmin) < 1e-6:
        vmin -= 0.1; vmax += 0.1
    pad = (vmax - vmin) * 0.05
    lim_min, lim_max = vmin - pad, vmax + pad
    
    # --- PLOTTING ---
    if labels:
        # Component Plot
        for i in range(x.shape[1]):
            # Filter per component if needed, here assume alignment matches
            ax_main.scatter(y[:,i], x[:,i], s=5, alpha=0.6, label=labels[i], c=colors[i])
        
        lgnd = ax_main.legend(loc='lower right', markerscale=2, scatterpoints=1, fontsize=10)
        for lh in lgnd.legend_handles: lh.set_alpha(1)
        
        # Simple Histograms
        ax_histx.hist(y_safe, bins=50, density=True, alpha=0.5, color='gray')
        ax_histy.hist(x_safe, bins=50, density=True, alpha=0.5, orientation='horizontal', color='gray')
        
    else:
        # Density Plot
        if len(x_safe) > 100000:
            idx = np.random.choice(len(x_safe), 100000, replace=False)
            sx, sy = y_safe[idx], x_safe[idx]
        else:
            sx, sy = y_safe, x_safe
        
        try:
            # JITTER: Add tiny noise to prevent Singular Matrix in KDE
            xy = np.vstack([sx, sy])
            jitter = np.random.normal(0, 1e-6, xy.shape)
            z = gaussian_kde(xy + jitter)(xy)
            
            # Check for NaNs in KDE output
            if not np.all(np.isfinite(z)): raise ValueError("KDE returned NaNs")

            idx = z.argsort()
            sc = ax_main.scatter(sx[idx], sy[idx], c=z[idx], s=5, cmap='viridis', rasterized=True)
            
            # Explicit Colorbar Axis
            cbar_ax = fig.add_axes([0.91, 0.11, 0.02, 0.65]) 
            fig.colorbar(sc, cax=cbar_ax, label='Density')
        except Exception as e:
            print(f"  [Warning] KDE failed ({e}). Fallback to plain scatter.")
            ax_main.scatter(sx, sy, s=5, alpha=0.5, c='steelblue')
            ax_main.text(0.05, 0.85, "Density Failed", transform=ax_main.transAxes, color='red')

        ax_histx.hist(y_safe, bins=50, density=True, alpha=0.6, color='steelblue')
        ax_histy.hist(x_safe, bins=50, density=True, alpha=0.6, orientation='horizontal', color='steelblue')

    ax_main.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1)
    ax_main.set_xlim(lim_min, lim_max)
    ax_main.set_ylim(lim_min, lim_max)
    ax_main.set_xlabel(xlabel); ax_main.set_ylabel(ylabel)
    ax_histx.set_title(title, fontweight='bold')

    txt = f"MAE: {metrics['mae']:.4f}\nRMSE: {metrics['rmse']:.4f}\n$R^2$: {metrics['r2']:.4f}"
    ax_main.text(0.05, 0.95, txt, transform=ax_main.transAxes, va='top', ha='left',
                 bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    
    plt.savefig(filename, dpi=300)
    print(f"  Saved plot: {filename}")
    plt.close()

if __name__ == "__main__":
    main()
