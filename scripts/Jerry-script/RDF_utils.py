import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import neighbor_list
from numba import njit
from joblib import Parallel, delayed
import multiprocessing
import os
import time
import argparse
import yaml
import json
import sys

# =========================================================
# Part A: Numba 核心算法 (与之前一致)
# =========================================================
@njit(fastmath=True, nogil=True)
def compute_hist_triclinic(pos1, pos2, cell, inv_cell, rmax_sq, nbins, rmax, is_same_elem):
    n1 = len(pos1)
    n2 = len(pos2)
    dr_step = rmax / nbins
    hist = np.zeros(nbins, dtype=np.int64)
    
    for i in range(n1):
        for j in range(n2):
            if is_same_elem and i == j:
                continue
            dx = pos2[j, 0] - pos1[i, 0]
            dy = pos2[j, 1] - pos1[i, 1]
            dz = pos2[j, 2] - pos1[i, 2]
            sx = dx*inv_cell[0,0] + dy*inv_cell[1,0] + dz*inv_cell[2,0]
            sy = dx*inv_cell[0,1] + dy*inv_cell[1,1] + dz*inv_cell[2,1]
            sz = dx*inv_cell[0,2] + dy*inv_cell[1,2] + dz*inv_cell[2,2]
            sx -= round(sx)
            sy -= round(sy)
            sz -= round(sz)
            rx = sx*cell[0,0] + sy*cell[1,0] + sz*cell[2,0]
            ry = sx*cell[0,1] + sy*cell[1,1] + sz*cell[2,1]
            rz = sx*cell[0,2] + sy*cell[1,2] + sz*cell[2,2]
            d2 = rx*rx + ry*ry + rz*rz
            if d2 < rmax_sq:
                d = np.sqrt(d2)
                bin_idx = int(d / dr_step)
                if bin_idx < nbins:
                    hist[bin_idx] += 1
    return hist

def get_cell_widths(cell):
    vol = np.abs(np.linalg.det(cell))
    if vol < 1e-8: return np.array([0.0, 0.0, 0.0])
    a, b, c = cell[0], cell[1], cell[2]
    wa = vol / np.linalg.norm(np.cross(b, c))
    wb = vol / np.linalg.norm(np.cross(a, c))
    wc = vol / np.linalg.norm(np.cross(a, b))
    return np.array([wa, wb, wc])

# =========================================================
# Part B: 智能混合处理核
# =========================================================
def process_chunk_smart(frames_chunk, elem1, elem2, rmax, nbins):
    local_hist = np.zeros(nbins, dtype=np.int64)
    local_rho_sum = 0.0
    local_n_centers_sum = 0.0
    n_frames = len(frames_chunk)
    is_same_elem = (elem1 == elem2)
    rmax_sq = rmax * rmax
    ATOM_THRESHOLD = 2000 
    
    for atoms in frames_chunk:
        cell = atoms.get_cell()
        cell_array = np.array(cell)
        widths = get_cell_widths(cell_array)
        repeats = np.ceil((2.01 * rmax) / widths).astype(int)
        
        if np.any(repeats > 1):
            atoms = atoms.repeat(repeats)
            cell_array = np.array(atoms.get_cell())
            
        vol = atoms.get_volume()
        positions = atoms.get_positions()
        symbols = np.array(atoms.get_chemical_symbols())
        idx1 = np.where(symbols == elem1)[0]
        idx2 = np.where(symbols == elem2)[0]
        n_centers = len(idx1)
        n_neighbors = len(idx2)
        total_atoms = len(atoms)
        
        if n_centers == 0 or n_neighbors == 0:
            continue
            
        local_rho_sum += n_neighbors / vol
        local_n_centers_sum += n_centers
        
        if total_atoms < ATOM_THRESHOLD:
            inv_cell = np.linalg.inv(cell_array)
            pos1 = positions[idx1]
            pos2 = positions[idx2]
            hist = compute_hist_triclinic(
                np.ascontiguousarray(pos1), np.ascontiguousarray(pos2), 
                np.ascontiguousarray(cell_array), np.ascontiguousarray(inv_cell), 
                rmax_sq, nbins, rmax, is_same_elem
            )
            local_hist += hist
        else:
            i_idx, j_idx, d_vals = neighbor_list('ijd', atoms, cutoff=rmax)
            mask_i = np.isin(i_idx, idx1)
            mask_j = np.isin(j_idx, idx2)
            mask = mask_i & mask_j
            if is_same_elem: mask = mask & (i_idx != j_idx)
            valid_d = d_vals[mask]
            hist, _ = np.histogram(valid_d, bins=nbins, range=(0, rmax))
            local_hist += hist
            
    return local_hist, local_rho_sum, local_n_centers_sum, n_frames

# =========================================================
# Part C: 并行驱动与绘图
# =========================================================
def compute_rdf_hybrid_parallel(traj_path, elements, rmax, nbins=120, n_jobs=-1):
    elem1, elem2 = elements
    print(f"  -> Reading: {os.path.basename(traj_path)} ...")
    try:
        traj = read(traj_path, index=':', format='traj')
    except Exception as e:
        print(f"  !! Failed to read {traj_path}: {e}")
        return None
        
    total_frames = len(traj)
    if n_jobs == -1: n_jobs = multiprocessing.cpu_count()
    chunk_size = max(1, total_frames // n_jobs)
    chunks = [traj[i:i + chunk_size] for i in range(0, total_frames, chunk_size)]
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk_smart)(chunk, elem1, elem2, rmax, nbins) 
        for chunk in chunks
    )
    
    total_hist = np.zeros(nbins, dtype=np.int64)
    total_rho = 0.0
    total_n_centers = 0.0
    total_processed_frames = 0
    
    for res in results:
        h, rho, nc, nf = res
        total_hist += h
        total_rho += rho
        total_n_centers += nc
        total_processed_frames += nf
        
    dr = rmax / nbins
    r = np.arange(dr/2, rmax, dr)
    
    if total_processed_frames == 0 or total_n_centers == 0:
        return r, np.zeros_like(r), np.zeros_like(r)

    avg_n_centers = total_n_centers / total_processed_frames
    cn = np.cumsum(total_hist) / (total_processed_frames * avg_n_centers)
    avg_rho = total_rho / total_processed_frames
    shell_volumes = 4 * np.pi * r**2 * dr
    normalization = shell_volumes * avg_rho * avg_n_centers * total_processed_frames
    
    with np.errstate(divide='ignore', invalid='ignore'):
        g_r = total_hist / normalization
        
    return r, np.nan_to_num(g_r), cn

def plot_final_results(all_results, pair_name, output_dir, cn_cutoff):
    fig, ax1 = plt.subplots(figsize=(9, 6), dpi=150)
    ax2 = ax1.twinx()
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
    
    print(f"\n  >>> Stats for {pair_name} (CN Cutoff = {cn_cutoff} A) <<<")
    
    for i, res in enumerate(all_results):
        r = res['r']
        g_r = res['g_r']
        cn = res['cn']
        label = res['label']
        c = colors[i % len(colors)]
        
        ax1.plot(r, g_r, color=c, lw=2, alpha=0.9, label=f"{label}")
        ax2.plot(r, cn, color=c, ls='--', lw=1.5, alpha=0.6)
        
        if cn_cutoff is not None:
            idx = (np.abs(r - cn_cutoff)).argmin()
            print(f"    Trait: {label:<20} | CN = {cn[idx]:.4f}")

    ax1.set_xlabel(r'Distance $r$ ($\AA$)', fontsize=12)
    ax1.set_ylabel(r'$g(r)$ (Solid Line)', fontsize=12)
    ax2.set_ylabel(r'Coordination Number (Dashed Line)', fontsize=12, rotation=270, labelpad=20)
    ax1.set_title(f'RDF & CN Analysis: {pair_name}', fontsize=14)
    ax1.set_xlim(0, all_results[0]['r'][-1])
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    
    filename = os.path.join(output_dir, f"RDF_{pair_name}.png")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"  -> Saved figure: {filename}")
    plt.close()

# =========================================================
# Part D: 配置解析与脚本入口
# =========================================================
def load_config(config_path):
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError("Unsupported config format. Use .yaml or .json")

def main():
    parser = argparse.ArgumentParser(description="High-Performance RDF/CN Analysis Tool")
    parser.add_argument("config", help="Path to configuration file (YAML/JSON)")
    args = parser.parse_args()
    
    # 1. 加载配置
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)
        
    config = load_config(args.config)
    
    # 2. 解析全局设置
    output_dir = config.get("output_dir", "./results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    n_jobs = config.get("n_jobs", -1)
    nbins = config.get("nbins", 120)
    
    trajectories = config.get("trajectories", [])
    tasks = config.get("tasks", [])
    
    if not trajectories:
        print("Error: No trajectories defined in config.")
        sys.exit(1)
        
    print(f"=== Starting Analysis ===")
    print(f"Output Directory: {output_dir}")
    print(f"CPU Cores: {multiprocessing.cpu_count()} (Using {n_jobs if n_jobs != -1 else 'ALL'})")
    
    # 3. 循环处理任务
    for task in tasks:
        pair = task.get("pair")
        rmax = task.get("rmax", 5.0)
        cn_cutoff = task.get("cn_cutoff", None)
        
        if not pair or len(pair) != 2:
            print(f"Skipping invalid task: {task}")
            continue
            
        pair_name = f"{pair[0]}-{pair[1]}"
        print(f"\n=== Processing Task: {pair_name} ===")
        
        results_container = []
        
        # 4. 循环处理轨迹
        for traj_conf in trajectories:
            path = traj_conf.get("path")
            label = traj_conf.get("label", os.path.basename(path))
            
            # 处理相对路径 (相对于配置文件所在目录)
            if not os.path.isabs(path):
                config_dir = os.path.dirname(os.path.abspath(args.config))
                path = os.path.join(config_dir, path)
            
            if not os.path.exists(path):
                print(f"  !! File not found: {path}")
                continue
                
            r, g_r, cn = compute_rdf_hybrid_parallel(
                path, (pair[0], pair[1]), rmax, nbins, n_jobs
            )
            
            if r is not None:
                results_container.append({
                    'r': r, 'g_r': g_r, 'cn': cn, 'label': label
                })
        
        # 5. 绘图
        if results_container:
            plot_final_results(results_container, pair_name, output_dir, cn_cutoff)
            
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()