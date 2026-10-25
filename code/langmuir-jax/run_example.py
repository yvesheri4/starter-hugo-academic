"""Run validation, synthetic density inference, plots, and animation.

python run_example.py --output results
python run_example.py --check-only
"""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize, minimize_scalar
import jax
import jax.numpy as jnp
import matplotlib
if __name__ == '__main__': matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from solver import (simulate, probe, field, potential, energy, analytical_probe,
                    N_REF, LENGTH, OMEGA_REF, E_REF)

Q_TRUE = 1.35
AMP = .02
DT = .025
STEPS = 720
SEED = 20261025


def validation():
    """Fail loudly if physical, convergence, or differentiation checks fail."""
    t = np.arange(STEPS+1)*DT
    spatial = []
    for n in (16, 32, 64, 128):
        e = np.asarray(probe(Q_TRUE, n=n))
        exact = np.asarray(analytical_probe(Q_TRUE, t))
        err = float(np.sqrt(np.mean((e-exact)**2))/(Q_TRUE*AMP))
        # Fit oscillation frequency with the discrete mode's known amplitude.
        dx = 2*np.pi/n
        amplitude = Q_TRUE*AMP*(dx/2)/np.tan(dx/2)
        fit = minimize_scalar(lambda w: np.mean((e+amplitude*np.cos(w*t))**2),
                              bounds=(1.05, 1.25), method='bounded',
                              options={'xatol': 1e-13})
        spatial.append(dict(n=n,relative_rms_field_error=err,
                            omega_over_omega_ref=float(fit.x),
                            relative_frequency_error=float(abs(fit.x/np.sqrt(Q_TRUE)-1))))
    spatial_orders = [float(np.log2(a['relative_rms_field_error']/b['relative_rms_field_error']))
                      for a,b in zip(spatial, spatial[1:])]
    assert min(spatial_orders)>1.9, spatial_orders
    temporal=[]
    n=64; dx=2*np.pi/n
    for dt in (.4,.2,.1,.05):
        steps=round(18/dt); times=np.arange(steps+1)*dt
        e=np.asarray(probe(Q_TRUE,n=n,steps=steps,dt=dt))
        # Exact semi-discrete solution isolates RK4 error from spatial error.
        expected=-Q_TRUE*AMP*(dx/2)/np.tan(dx/2)*np.cos(np.sqrt(Q_TRUE)*np.cos(dx/2)*times)
        temporal.append(dict(dt=dt,relative_rms_field_error=float(np.sqrt(np.mean((e-expected)**2))/(Q_TRUE*AMP))))
    temporal_orders=[float(np.log2(a['relative_rms_field_error']/b['relative_rms_field_error']))
                     for a,b in zip(temporal,temporal[1:])]
    assert min(temporal_orders)>3.8, temporal_orders
    states=simulate(Q_TRUE)
    energies=np.asarray(energy(states,Q_TRUE))
    energy_error=float(np.max(abs(energies/energies[0]-1)))
    mass_error=float(np.max(abs(np.asarray(states[:,0]).mean(axis=-1))))
    equilibrium=float(np.max(abs(np.asarray(simulate(Q_TRUE,amplitude=0.)))))
    phi=np.asarray(potential(states[:,0],Q_TRUE))
    lap=(np.roll(phi,-1,axis=-1)-2*phi+np.roll(phi,1,axis=-1))/dx**2
    poisson_error=float(np.max(abs(lap-Q_TRUE*np.asarray(states[:,0]))))
    assert np.isfinite(states).all()
    assert energy_error<1e-7 and mass_error<1e-12 and equilibrium==0 and poisson_error<1e-10
    data=analytical_probe(Q_TRUE,jnp.asarray(t))[::4]
    loss=jax.jit(lambda q: jnp.mean(((probe(q)[::4]-data)/(Q_TRUE*AMP))**2))
    grad=jax.jit(jax.grad(loss))
    gradients=[]
    for q in (.9,1.1,1.5):
        ad=float(grad(q))
        for h in (1e-3,1e-4,1e-5):
            fd=float((loss(q+h)-loss(q-h))/(2*h))
            gradients.append(dict(q=q,h=h,autodiff=ad,finite_difference=fd,
                                  relative_error=abs(ad-fd)/max(abs(ad),1e-12)))
    assert max(a['relative_error'] for a in gradients if a['h']==1e-5)<1e-6
    report=dict(spatial=spatial,spatial_orders=spatial_orders,temporal=temporal,
                temporal_orders=temporal_orders,gradients=gradients,
                max_relative_energy_drift=energy_error,max_mean_density_perturbation=mass_error,
                equilibrium_error=equilibrium,max_poisson_residual=poisson_error,
                all_checks_passed=True)
    print('Validation passed:',json.dumps(report,indent=2),flush=True)
    return report


def infer(data, n=64):
    """Fit log density with bounded L-BFGS-B and JAX reverse-mode gradients.

    Times, initial relative perturbation, probe position, and calibration are
    known. The bounds and initial guess are explicit; global recovery is not
    guaranteed for an oscillatory loss.
    """
    data=jnp.asarray(data)
    loss=jax.jit(lambda logq:jnp.mean(((probe(jnp.exp(logq),n=n)[::4]-data)/AMP)**2))
    vg=jax.jit(jax.value_and_grad(loss))
    history=[]
    def objective(p):
        v,g=vg(p[0]); return float(v),np.array([float(g)])
    def record(p): history.append([float(np.exp(p[0])),float(loss(p[0]))])
    record(np.array([np.log(1.0)]))
    fit=minimize(objective,[np.log(1.0)],jac=True,method='L-BFGS-B',
                 bounds=[(np.log(.7),np.log(1.8))],callback=record,
                 options={'gtol':1e-10,'ftol':1e-14,'maxiter':100,'maxls':40})
    if not np.isclose(history[-1][0],np.exp(fit.x[0]),rtol=0,atol=1e-14):record(fit.x)
    assert fit.success, fit.message
    q=float(np.exp(fit.x[0]))
    return dict(q=q,n_m3=q*N_REF,relative_error=(q/Q_TRUE-1),
                loss=float(fit.fun),iterations=int(fit.nit),success=bool(fit.success)),np.array(history)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    report=validation()
    if args.check_only:return
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    t=np.arange(STEPS+1)*DT; times_ns=t/OMEGA_REF*1e9
    exact=np.asarray(analytical_probe(Q_TRUE,jnp.asarray(t)))
    rng=np.random.default_rng(SEED)
    observations=exact[::4]+rng.normal(0,.03*Q_TRUE*AMP,len(t[::4]))
    clean,hclean=infer(exact[::4]); noisy,hnoisy=infer(observations)
    fine,_=infer(exact[::4],n=128)
    assert abs(clean['relative_error'])<.005
    assert abs(noisy['relative_error'])<.02
    assert abs(fine['relative_error'])<abs(clean['relative_error'])/3
    traces={k:np.asarray(probe(v)) for k,v in [('initial',1.),('clean_fit',clean['q']),('noisy_fit',noisy['q'])]}
    states=np.asarray(simulate(Q_TRUE)); fields=np.asarray(field(jnp.asarray(states[:,0]),Q_TRUE))
    eunit=float(E_REF)
    np.savetxt(out/'traces.csv',np.column_stack([times_ns,exact*eunit,*[v*eunit for v in traces.values()]]),delimiter=',',header='time_ns,analytical_E_V_m,initial_E_V_m,clean_fit_E_V_m,noisy_fit_E_V_m',comments='')
    np.savetxt(out/'observations.csv',np.column_stack([times_ns[::4],observations*eunit]),delimiter=',',header='time_ns,noisy_E_V_m',comments='')
    for name,h in [('clean',hclean),('noisy',hnoisy)]:
        np.savetxt(out/(name+'_optimization.csv'),np.column_stack([np.arange(len(h)),h]),delimiter=',',header='iteration,n_over_nref,scaled_mean_square_loss',comments='')
    np.savez_compressed(out/'fields.npz',time_ns=times_ns,x_mm=np.arange(64)/64*LENGTH*1e3,eta=states[:,0],u=states[:,1],E_V_m=fields*eunit)
    summary=dict(model='Linear cold-electron fluid; fixed ions; periodic 1D',n_ref_m3=N_REF,n_true_m3=Q_TRUE*N_REF,
                 length_m=LENGTH,amplitude=AMP,n_grid=64,dt_over_omega_ref=DT,steps=STEPS,
                 omega_ref_rad_s=OMEGA_REF,noise_std_fraction_of_peak=.03,seed=SEED,
                 observation_count=len(observations),initial_density_m3=N_REF,fit_bounds_m3=[.7*N_REF,1.8*N_REF],
                 clean_fit=clean,noisy_fit=noisy,clean_fit_128=fine,validation=report,
                 environment=dict(python=platform.python_version(),jax=jax.__version__,numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,backend=jax.default_backend(),devices=[str(d) for d in jax.devices()]))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf9f6','axes.facecolor':'#faf9f6','axes.labelcolor':'#183740','text.color':'#183740','axes.titleweight':'bold'})
    teal='#158a8a'; orange='#c75f30'; navy='#183740'
    fig,axes=plt.subplots(2,1,figsize=(9,7),layout='constrained')
    a=axes[0];a.plot(times_ns,exact*eunit,color=navy,lw=2,label='Analytical truth');a.scatter(times_ns[::4],observations*eunit,s=10,alpha=.45,color=orange,label='Synthetic observations (3% noise)');a.plot(times_ns,traces['noisy_fit']*eunit,color=teal,lw=1.8,ls='--',label='JAX fit');a.plot(times_ns,traces['initial']*eunit,color='gray',alpha=.6,lw=1,label='Initial guess');a.set(xlabel='Time (ns)',ylabel='Probe electric field (V/m)',title='Recovering density from a plasma oscillation');a.legend(fontsize=9,ncol=2)
    a=axes[1]
    for h,label,color in [(hclean,'Clean data',teal),(hnoisy,'3% noise',orange)]:a.plot(np.arange(len(h)),h[:,0],'-o',label=label,color=color,ms=4)
    a.axhline(Q_TRUE,color=navy,ls=':',label='True density');a.set(xlabel='Optimizer iteration (including initial guess)',ylabel=r'Density / $10^{14}$ m$^{-3}$',title='A wrong initial guess converges toward the true density');a.legend(fontsize=9)
    fig.savefig(out/'inference.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for a,data,xkey,title in [(axes[0],report['spatial'],'n','Second-order spatial convergence'),(axes[1],report['temporal'],'dt','Fourth-order time integration')]:
        xs=[d[xkey] for d in data]
        a.loglog(xs,[d['relative_rms_field_error'] for d in data],'-o',color=teal);a.set(xlabel='Grid points' if xkey=='n' else r'Timestep ($\omega_{ref}^{-1}$)',ylabel='RMS field error / initial peak',title=title);a.grid(alpha=.2,which='both')
        a.set_xticks(xs,[str(x) for x in xs]);a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.savefig(out/'validation.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,6),layout='constrained',sharex=True)
    for a,z,label in [(axes[0],100*states[:,0],r'Density perturbation (%)'),(axes[1],fields*eunit,'Electric field (V/m)')]:
        im=a.imshow(z.T,origin='lower',aspect='auto',extent=[times_ns[0],times_ns[-1],0,LENGTH*1e3],cmap='RdBu_r',vmin=-abs(z).max(),vmax=abs(z).max());a.set(ylabel='Position (mm)',title=label);fig.colorbar(im,ax=a,pad=.02)
    axes[-1].set_xlabel('Time (ns)');fig.savefig(out/'evolution.png',dpi=150);plt.close(fig)
    # A finite one-period clip; the website provides user-controlled playback.
    fig,axes=plt.subplots(2,1,figsize=(7,4.5),layout='constrained',sharex=True)
    x=np.arange(65)/64*LENGTH*1e3
    lines=[]
    for a,z,color,label in [(axes[0],100*states[:,0],teal,'Density perturbation (%)'),(axes[1],fields*eunit,orange,'Electric field (V/m)')]:
        line,=a.plot(x,np.r_[z[0],z[0,0]],color=color,lw=2);lines.append(line);a.set(xlim=(0,LENGTH*1e3),ylim=(-1.1*abs(z).max(),1.1*abs(z).max()),ylabel=label);a.axhline(0,color='gray',lw=.5)
    axes[-1].set_xlabel('Position (mm)');title=axes[0].set_title('Langmuir oscillation')
    indices=np.linspace(0,round(2*np.pi/(np.sqrt(Q_TRUE)*np.cos(np.pi/64))/DT),48).astype(int)
    def update(i):
        lines[0].set_ydata(np.r_[100*states[i,0],100*states[i,0,0]]);lines[1].set_ydata(np.r_[fields[i]*eunit,fields[i,0]*eunit]);title.set_text(f'Langmuir oscillation | t = {times_ns[i]:.2f} ns');return lines+[title]
    fig.savefig(out/'poster.png',dpi=110)
    FuncAnimation(fig,update,frames=indices,interval=100).save(out/'oscillation.gif',writer=PillowWriter(fps=10),dpi=90);plt.close(fig)
    print('Inference:',json.dumps({'clean':clean,'noisy':noisy,'fine':fine},indent=2),flush=True)


if __name__=='__main__':main()
