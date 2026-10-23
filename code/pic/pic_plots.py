"""Standalone HDF5 postprocessor; does not import or run the PIC solver.

python pic_plots.py results/my_run/run.h5 --animate
python pic_plots.py results/my_run/run.h5 --population electrons --distribution-animation
"""
import argparse
from pathlib import Path
import h5py
import numpy as np


def render(path,animate=False,population=None,distribution_animation=False,output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm,LogNorm
    from PIL import Image
    path=Path(path);out=Path(output) if output else path.parent/('plots' if population is None else 'plots_population_'+str(population))
    out.mkdir(parents=True,exist_ok=True)

    def save_gif(fig,update,values,destination):
        images=[]
        for value in values:
            update(value);fig.canvas.draw()
            image=Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy())
            images.append(image.quantize(colors=128,method=Image.Quantize.FASTOCTREE))
        images[0].save(destination,save_all=True,append_images=images[1:],duration=80,loop=0)

    with h5py.File(path,'r') as data:
        if data.attrs.get('format')!='electrostatic_1d1v_pic' or not data.attrs.get('complete',False):
            raise ValueError('Expected a completed electrostatic_1d1v_pic HDF5 file')
        names=list(data['populations/name'].asstr()[:])
        if population is not None and population not in names:raise ValueError(f'Unknown population; choose from {names}')
        selected=list(range(len(names))) if population is None else [names.index(population)]
        label='all populations' if population is None else population
        fg=data['fields'];t=fg['time'][:];x=fg['x'][:];field=fg['electric'][:];energy=fg['electric_energy'][:]
        dg=data['distribution'];times=dg['time'][:];xe=dg['x_edges'][:];ve=dg['v_edges'][:]
        length=xe[-1]-xe[0]
        periodic=x[-1]<xe[-1]-.5*(x[1]-x[0])
        mx=np.append(x,xe[-1]) if periodic else x
        me=np.column_stack((field,field[:,0])) if periodic else field
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
        limit=max(float(np.max(abs(field))),1e-15)
        im=axes[0].pcolormesh(mx,t,me,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='auto')
        axes[0].set(title='Electric field E(x,t)',xlabel='x [v0/omega_p0]',ylabel='Time [1/omega_p0]',xlim=(xe[0],xe[-1]),ylim=(t[0],t[-1]))
        fig.colorbar(im,ax=axes[0],label='E [m_e v0 omega_p0/e]')
        axes[1].semilogy(t,np.maximum(np.sqrt(2*energy/length),1e-30))
        axes[1].set(title='RMS field amplitude',xlabel='Time',ylabel='sqrt(integral E² dx / L)')
        axes[2].semilogy(t,np.maximum(energy,1e-30))
        axes[2].set(title='Stored field energy',xlabel='Time',ylabel='0.5 integral E² dx')
        fig.suptitle(path.parent.name)
        fig.savefig(out/'electric_field_evolution.png',dpi=150);plt.close(fig)

        def distribution(j):return np.sum(dg['f'][j,selected,:,:],axis=0)
        maximum=max(max(float(distribution(j).max()) for j in range(len(times))),1e-15)
        norm=LogNorm(vmin=maximum*1e-5,vmax=maximum)
        outside=(dg['number_below_range'][:,selected]+dg['number_above_range'][:,selected]).sum(axis=1)
        total=dg['number'][:,selected].sum(axis=1)
        fraction=np.divide(outside,total,out=np.zeros_like(outside),where=total>0)
        fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        chosen=np.rint(np.linspace(0,len(times)-1,5)).astype(int)
        for ax,j in zip(axes.flat[:5],chosen):
            im=ax.pcolormesh(xe,ve,np.ma.masked_less_equal(distribution(j).T,0),norm=norm,cmap='magma',shading='flat')
            ax.set(title=f't={times[j]:g}',xlabel='x',ylabel='v')
        fig.colorbar(im,ax=list(axes.flat[:5]),label='f [n0/v0]')
        for j in (0,len(times)-1):
            marginal=(distribution(j)*np.diff(xe)[:,None]).sum(axis=0)/length
            axes.flat[5].plot((ve[:-1]+ve[1:])/2,marginal,label=f't={times[j]:g}')
        axes.flat[5].set(xlabel='v',ylabel='Spatial mean f(v)');axes.flat[5].legend()
        fig.suptitle(f'{label}: f(x,v,t), all solver particles | max outside velocity range: {fraction.max():.3%}')
        fig.savefig(out/'distribution.png',dpi=150);plt.close(fig)
        if distribution_animation:
            fig,(ax,trace)=plt.subplots(2,1,figsize=(9,7),layout='constrained',gridspec_kw={'height_ratios':[3,1]})
            mesh=ax.pcolormesh(xe,ve,np.ma.masked_less_equal(distribution(0).T,0),norm=norm,cmap='magma',shading='flat')
            fig.colorbar(mesh,ax=ax,label='f [n0/v0]');ax.set(xlabel='x',ylabel='v')
            trace.semilogy(t,np.maximum(energy,1e-30));trace.set(xlabel='Time',ylabel='Field energy');cursor=trace.axvline(0,color='red')
            def update_f(j):
                mesh.set_array(np.ma.masked_less_equal(distribution(j).T,0).ravel())
                ax.set_title(f'{label}: t={times[j]:g}, outside range={fraction[j]:.3%}')
                cursor.set_xdata([times[j],times[j]])
            save_gif(fig,update_f,range(len(times)),out/'distribution.gif');plt.close(fig)

        # Particle frames have their own time axis, independent of field/f cadence.
        groups=list(data['particles'].values())
        if groups:
            def particles(g):
                arrays=[(g[f'p{k:03d}/x'][:],g[f'p{k:03d}/v'][:],k) for k in selected]
                return (np.concatenate([a[0] for a in arrays]),np.concatenate([a[1] for a in arrays]),
                        np.concatenate([np.full(len(a[0]),a[2]) for a in arrays]))
            bounds=[particles(g)[1] for g in groups]
            all_v=np.concatenate(bounds);low,high=(all_v.min(),all_v.max()) if len(all_v) else (-1,1)
            pad=.05*max(high-low,1);limits=(low-pad,high+pad)
            cmap=plt.get_cmap('tab10' if len(names)<=10 else 'turbo',len(names))
            colors=BoundaryNorm(np.arange(len(names)+1)-.5,len(names))
            fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
            for ax,j in zip(axes.flat[:5],np.rint(np.linspace(0,len(groups)-1,5)).astype(int)):
                px,pv,pc=particles(groups[j]);ax.scatter(px,pv,c=pc,cmap=cmap,norm=colors,s=.7)
                ax.set(title=f't={groups[j].attrs["time"]:g}',xlabel='x',ylabel='v',xlim=(0,length),ylim=limits)
            axes.flat[5].semilogy(t,np.maximum(energy,1e-30));axes.flat[5].set(xlabel='Time',ylabel='Field energy')
            fig.savefig(out/'phase_space.png',dpi=150);plt.close(fig)
            if animate:
                fig,(ax,trace)=plt.subplots(2,1,figsize=(9,7),layout='constrained',gridspec_kw={'height_ratios':[3,1]})
                dots=ax.scatter([],[],s=1,c=[],cmap=cmap,norm=colors)
                cb=fig.colorbar(dots,ax=ax,ticks=np.arange(len(names)));cb.ax.set_yticklabels(names);cb.set_label('Population')
                ax.set(xlim=(0,length),ylim=limits,xlabel='x',ylabel='v')
                trace.semilogy(t,np.maximum(energy,1e-30));trace.set(xlabel='Time',ylabel='Field energy');cursor=trace.axvline(0,color='red')
                def update_p(j):
                    px,pv,pc=particles(groups[j]);dots.set_offsets(np.column_stack((px,pv)));dots.set_array(pc)
                    time=groups[j].attrs['time'];ax.set_title(f'{label}: t={time:g}')
                    cursor.set_xdata([time,time])
                save_gif(fig,update_p,range(len(groups)),out/'phase_space.gif');plt.close(fig)
        elif animate:
            print('No particle snapshots were requested. Use --distribution-animation to animate f instead.')
        print(f'Plots: {out}; max fraction outside velocity grid: {fraction.max():.6%}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path);p.add_argument('--animate',action='store_true')
    p.add_argument('--distribution-animation',action='store_true');p.add_argument('--population')
    p.add_argument('--output',type=Path)
    a=p.parse_args();render(a.data,a.animate,a.population,a.distribution_animation,a.output)
