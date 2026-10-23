"""Render a compact phase-space GIF from completed 1D1V PIC HDF5 output."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('preview'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    with h5py.File(args.input, 'r') as h:
        if not h.attrs.get('complete', False):
            raise ValueError('Input run is incomplete')
        times = h['distribution/time'][:]
        indices = np.unique(np.linspace(0, len(times)-1, min(51,len(times)), dtype=int))
        x, v = h['distribution/x_edges'][:], h['distribution/v_edges'][:]
        fig, ax = plt.subplots(figsize=(6.4,4), dpi=100)
        fig.subplots_adjust(left=.11,right=.85,bottom=.14,top=.87)
        ax.set(xlabel='Position x (normalized)', ylabel='Velocity v (normalized)',ylim=(v[0],v[-1]))
        # Fixed logarithmic color scale across all frames, no smoothing.
        from matplotlib.colors import LogNorm
        peak=max(float(h['distribution/f'][int(i)].sum(axis=0).max()) for i in indices)
        mesh=ax.pcolormesh(x,v,h['distribution/f'][0].sum(axis=0).T,cmap='viridis',norm=LogNorm(vmin=peak/1000,vmax=peak),rasterized=True)
        fig.colorbar(mesh, ax=ax, label='Number distribution f', fraction=.05, pad=.03)
        title=ax.set_title('')
        for i in indices:
            mesh.set_array(h['distribution/f'][int(i)].sum(axis=0).T.ravel())
            title.set_text(f'Two-stream phase space | t = {times[i]:.1f}')
            fig.canvas.draw()
            frame=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]).convert('P',palette=Image.Palette.ADAPTIVE,colors=64)
            frames.append(frame)
        frames[-1].convert('RGB').save(args.output/'poster.png',optimize=True)
        frames[0].save(args.output/'two-stream.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
        plt.close(fig)
    print(f'{len(frames)} frames, t={times[0]} to {times[-1]}')

if __name__ == '__main__':
    main()
