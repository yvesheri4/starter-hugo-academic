"""General normalized electrostatic 1D1V PIC. Usage: python pic_solver.py input.json.

One numerical engine for all inputs: CIC, Poisson, kick-drift-kick.
No problem names or example-dependent numerical branches.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import NormalDist
import numpy as np


def keys(obj, allowed, required=(), context='input'):
    if not isinstance(obj, dict):
        raise ValueError(f'{context}: expected an object')
    missing, extra = set(required)-set(obj), set(obj)-set(allowed)
    if missing or extra:
        raise ValueError(f'{context}: missing keys {sorted(missing)}, unknown keys {sorted(extra)}')


def real(value, name, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value):
        raise ValueError(f'{name}: expected a finite number')
    if minimum is not None and (value < minimum or (strict and value == minimum)):
        raise ValueError(f'{name}: must be {">" if strict else ">="} {minimum}')
    return value


def integer(value, name, minimum=0):
    real(value,name,minimum)
    if int(value) != value:
        raise ValueError(f'{name}: expected an integer')
    return int(value)


def validate(config):
    """Strict schema validation; returned config includes all default values."""
    c=deepcopy(config)
    keys(c,('domain','time','background_charge_density','populations','output','phase_space','seed'),
         ('domain','time','background_charge_density','populations','phase_space'))
    c['seed']=integer(c.get('seed',42),'seed')
    real(c['background_charge_density'],'background_charge_density')
    d=c['domain']
    keys(d,('length','cells','field_boundary','particle_boundary','potential_left','potential_right'),
         ('length','cells','field_boundary','particle_boundary'),'domain')
    real(d['length'],'length',0,True)
    d['cells']=integer(d['cells'],'cells',4)
    if d['field_boundary'] not in ('periodic','dirichlet'):
        raise ValueError('field_boundary must be periodic or dirichlet')
    b=d['particle_boundary']
    if d['field_boundary']=='periodic':
        if b != 'periodic' or d['cells']%2:
            raise ValueError('Periodic field requires periodic particles and even cells')
        if 'potential_left' in d or 'potential_right' in d:
            raise ValueError('Electrode potentials are not used with periodic boundaries')
    else:
        keys(b,('left','right'),('left','right'),'particle_boundary')
        if any(v not in ('absorbing','reflecting') for v in b.values()):
            raise ValueError('Each particle boundary must be absorbing or reflecting')
        for k in ('potential_left','potential_right'):
            d[k]=real(d.get(k,0.),k)
    t=c['time']
    keys(t,('dt','end','diagnostic_interval'),('dt','end'),'time')
    for k in ('dt','end'): real(t[k],k,0,True)
    t['diagnostic_interval']=real(t.get('diagnostic_interval',t['dt']),'diagnostic_interval',0,True)
    for k in ('end','diagnostic_interval'):
        ratio=t[k]/t['dt']
        if ratio<1 or not np.isclose(ratio,round(ratio),rtol=0,atol=1e-8):
            raise ValueError(f'{k} must be a positive integer multiple of dt')
    o=c.setdefault('output',{})
    keys(o,('directory','save_interval','field_interval','distribution_interval','particle_interval','particle_dump','display_particles'),context='output')
    o.setdefault('directory','results/run')
    if not isinstance(o['directory'],str) or not o['directory']:
        raise ValueError('output.directory must be a nonempty path')
    o['display_particles']=integer(o.get('display_particles',10000),'display_particles',1)
    o['save_interval']=real(o.get('save_interval',t['diagnostic_interval']),'save_interval',0,True)
    for k in ('save_interval','field_interval','distribution_interval','particle_interval'):
        o[k]=real(o.get(k,o['save_interval']),k,0,True)
        ratio=o[k]/t['dt']
        if ratio<1 or not np.isclose(ratio,round(ratio),rtol=0,atol=1e-8):
            raise ValueError(f'{k} must be a positive integer multiple of dt')
    o.setdefault('particle_dump','sample')
    if o['particle_dump'] not in ('none','sample','all'):
        raise ValueError('particle_dump must be none, sample or all')
    ps=c['phase_space']
    keys(ps,('position_bins','velocity_bins','velocity_min','velocity_max'),
         ('position_bins','velocity_bins','velocity_min','velocity_max'),'phase_space')
    for k in ('position_bins','velocity_bins'):ps[k]=integer(ps[k],k,1)
    for k in ('velocity_min','velocity_max'):real(ps[k],k)
    if ps['velocity_max']<=ps['velocity_min']:raise ValueError('velocity_max must exceed velocity_min')
    if not isinstance(c['populations'],list) or not c['populations']:
        raise ValueError('At least one population is required')
    names=[]
    for p in c['populations']:
        keys(p,('name','charge','mass','density','particles','particle_weight','initial','sources'),
             ('name','charge','mass','density','particles'),'population')
        if not isinstance(p['name'],str) or not p['name'] or p['name'] in names:
            raise ValueError('Population names must be nonempty and unique')
        names.append(p['name'])
        real(p['charge'],'charge'); real(p['mass'],'mass',0,True)
        real(p['density'],'density',0)
        p['particles']=integer(p['particles'],'particles')
        if p['particles']:
            real(p['density'],'density',0,True)
            weight=p['density']*d['length']/p['particles']
            if 'particle_weight' in p and not np.isclose(real(p['particle_weight'],'particle_weight',0,True),weight,rtol=1e-12,atol=0):
                raise ValueError('particle_weight disagrees with density*length/particles')
            p['particle_weight']=weight
        elif p['density'] != 0 or 'particle_weight' not in p:
            raise ValueError('Empty initial populations require density=0 and particle_weight>0')
        real(p['particle_weight'],'particle_weight',0,True)
        init=p.setdefault('initial',{})
        keys(init,('file','group','position','velocity'),context='initial')
        if 'file' in init:
            if set(init)-{'file','group'} or not isinstance(init['file'],str):
                raise ValueError('initial.file must be a path and cannot be combined with position/velocity')
            if 'group' in init and not isinstance(init['group'],str):raise ValueError('initial.group must be a string')
        else:
            if 'group' in init:raise ValueError('initial.group requires initial.file')
            pos=init.setdefault('position',{})
            keys(pos,('sampling','spatial_points','density_mode'),context='position')
            pos.setdefault('sampling','quiet')
            if pos['sampling'] not in ('quiet','random','lattice'):
                raise ValueError('position sampling must be quiet, random or lattice')
            if pos['sampling']=='lattice':
                pos['spatial_points']=integer(pos.get('spatial_points',128),'spatial_points',1)
                if p['particles']%pos['spatial_points']:
                    raise ValueError('particles must be divisible by spatial_points')
            elif 'spatial_points' in pos:
                raise ValueError('spatial_points only applies to lattice sampling')
            if 'density_mode' in pos:
                mode=pos['density_mode']
                keys(mode,('amplitude','mode','phase'),('amplitude',),'density_mode')
                if abs(real(mode['amplitude'],'density amplitude'))>=1:
                    raise ValueError('Density amplitude must have magnitude below 1')
                mode['mode']=integer(mode.get('mode',1),'density mode',1)
                mode['phase']=real(mode.get('phase',0.),'density phase')
            vel=init.setdefault('velocity',{})
            keys(vel,('distribution','drift','thermal_speed','sampling','perturbation'),context='velocity')
            vel.setdefault('distribution','cold');vel.setdefault('sampling','random')
            vel['drift']=real(vel.get('drift',0.),'drift')
            vel['thermal_speed']=real(vel.get('thermal_speed',0.),'thermal_speed',0)
            if vel['distribution'] not in ('cold','maxwellian','waterbag') or vel['sampling'] not in ('random','quiet'):
                raise ValueError('Unsupported velocity distribution or sampling')
            if vel['distribution']=='cold' and vel['thermal_speed']!=0:
                raise ValueError('Cold distributions require thermal_speed=0')
            if 'perturbation' in vel:
                mode=vel['perturbation']
                keys(mode,('amplitude','mode','phase'),('amplitude',),'velocity perturbation')
                real(mode['amplitude'],'velocity amplitude')
                mode['mode']=real(mode.get('mode',1.),'velocity mode',0,True)
                mode['phase']=real(mode.get('phase',0.),'velocity phase')
                if d['field_boundary']=='periodic' and mode['mode']!=int(mode['mode']):
                    raise ValueError('Periodic velocity modes must be integers')
        p.setdefault('sources',[])
        if not isinstance(p['sources'],list): raise ValueError('sources must be a list')
        if p['sources'] and d['field_boundary']=='periodic':
            raise ValueError('Sources require nonperiodic boundaries')
        for s in p['sources']:
            keys(s,('boundary','number_flux','velocity'),('boundary','number_flux','velocity'),'source')
            if s['boundary'] not in ('left','right'): raise ValueError('Source boundary must be left or right')
            if b[s['boundary']]!='absorbing': raise ValueError('Sources require an absorbing inlet')
            real(s['number_flux'],'number_flux',0,True);real(s['velocity'],'source velocity')
            if (s['boundary']=='left' and s['velocity']<=0) or (s['boundary']=='right' and s['velocity']>=0):
                raise ValueError('Source velocity must point into the domain')
    if d['field_boundary']=='periodic':
        charge=c['background_charge_density']+sum(p['charge']*p['density'] for p in c['populations'])
        scale=max(1.,abs(c['background_charge_density']),sum(abs(p['charge']*p['density']) for p in c['populations']))
        if abs(charge)>1e-12*scale:
            raise ValueError('Periodic Poisson requires zero net charge: background + sum(charge*density)=0')
    return c


class PIC:
    def __init__(self, config, input_directory='.'):
        self.config=validate(config)
        self.base=Path(input_directory)
        d=self.config['domain']
        self.length,self.cells=d['length'],d['cells']
        self.dx=self.length/self.cells
        self.periodic=d['field_boundary']=='periodic'
        self.nodes=np.arange(self.cells+(not self.periodic))*self.dx
        self.dt=self.config['time']['dt'];self.index=0
        self.populations=[]
        for k,spec in enumerate(self.config['populations']):
            rng=np.random.default_rng(np.random.SeedSequence([self.config['seed'],k]))
            x,v=self.initialize(spec,rng)
            self.populations.append(dict(spec=spec,x=x,v=v,injected=0,absorbed_left=0,absorbed_right=0))

    @property
    def time(self): return self.index*self.dt

    def initialize(self,p,rng):
        n=p['particles'];init=p['initial']
        if 'file' in init:
            path=self.base/init['file']
            if path.suffix.lower() in ('.h5','.hdf5'):
                import h5py
                with h5py.File(path,'r') as data:
                    group=data[init.get('group','/')]
                    x=np.asarray(group['x'],dtype=float);v=np.asarray(group['v'],dtype=float)
            else:
                if 'group' in init:raise ValueError('initial.group only applies to HDF5 input')
                with np.load(path,allow_pickle=False) as data:
                    x=np.asarray(data['x'],dtype=float).copy();v=np.asarray(data['v'],dtype=float).copy()
            if x.shape!=(n,) or v.shape!=(n,) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)) or np.any(x<0) or np.any(x>=self.length):
                raise ValueError('Initial particle file needs finite x,v arrays of length particles with 0 <= x < L')
            return x,v
        if n==0:return np.empty(0),np.empty(0)
        pos,vel=init['position'],init['velocity']
        nx=pos.get('spatial_points',n)
        base=((np.arange(nx)+.5)/nx if pos['sampling']!='random' else rng.random(nx))*self.length
        x=base.copy()
        if 'density_mode' in pos:
            m=pos['density_mode'];a=m['amplitude'];k=2*np.pi*m['mode']/self.length;phase=m['phase']
            # Invert the monotone density CDF by bisection (also robust for |a| near 1).
            low=np.zeros(nx);high=np.full(nx,self.length)
            for _ in range(55):
                x=(low+high)/2
                residual=x+a*(np.sin(k*x+phase)-np.sin(phase))/k-base
                high=np.where(residual>0,x,high);low=np.where(residual<=0,x,low)
            x=(low+high)/2
        nv=n//nx
        if pos['sampling']=='lattice': x=np.repeat(x,nv)
        if vel['distribution']=='cold': v=np.full(n,vel['drift'],dtype=float)
        else:
            count=nv if pos['sampling']=='lattice' and vel['sampling']=='quiet' else n
            u=(np.arange(count)+.5)/count if vel['sampling']=='quiet' else rng.random(count)
            if vel['distribution']=='maxwellian':
                z=np.array([NormalDist().inv_cdf(float(t)) for t in u]) if vel['sampling']=='quiet' else rng.standard_normal(count)
            else:z=np.sqrt(3)*(2*u-1)  # thermal_speed denotes standard deviation
            if pos['sampling']=='lattice' and vel['sampling']=='quiet':z=np.tile(z,nx)
            elif vel['sampling']=='quiet':rng.shuffle(z)  # avoid artificial x-v correlation
            v=vel['drift']+vel['thermal_speed']*z
        if 'perturbation' in vel:
            m=vel['perturbation']
            v+=m['amplitude']*np.sin(2*np.pi*m['mode']*x/self.length+m['phase'])
        return x,v

    def weights(self,x):
        u=(x%self.length if self.periodic else np.clip(x,0,self.length))/self.dx
        i=np.floor(u).astype(int)
        if self.periodic:return i%self.cells,(i+1)%self.cells,u-i
        i=np.minimum(i,self.cells-1)
        return i,i+1,u-i

    def charge_density(self):
        rho=np.full(len(self.nodes),float(self.config['background_charge_density']))
        for p in self.populations:
            i,j,a=self.weights(p['x'])
            number=(np.bincount(i,weights=1-a,minlength=len(rho))+np.bincount(j,weights=a,minlength=len(rho)))*p['spec']['particle_weight']/self.dx
            if not self.periodic:number[[0,-1]]*=2
            rho+=p['spec']['charge']*number
        return rho

    def poisson(self,rho):
        if self.periodic:
            k=2*np.pi*np.fft.rfftfreq(self.cells,d=self.dx)
            r=np.fft.rfft(rho);e=np.zeros_like(r);phi=np.zeros_like(r)
            e[1:-1]=r[1:-1]/(1j*k[1:-1]);phi[1:]=r[1:]/k[1:]**2
            return np.fft.irfft(phi,n=self.cells),np.fft.irfft(e,n=self.cells)
        d=self.config['domain']
        increments=np.concatenate(([0.],np.cumsum(-rho[1:-1]*self.dx**2)))
        increments+=(d['potential_right']-d['potential_left'])/self.cells-increments.mean()
        phi=np.concatenate(([d['potential_left']],d['potential_left']+np.cumsum(increments)))
        phi[-1]=d['potential_right']
        return phi,-np.gradient(phi,self.dx,edge_order=2)

    def acceleration(self,x,e,spec):
        i,j,a=self.weights(x)
        return spec['charge']/spec['mass']*((1-a)*e[i]+a*e[j])

    def boundaries(self,x,v):
        if self.periodic:return x%self.length,v,np.ones(len(x),dtype=bool),0,0
        b=self.config['domain']['particle_boundary']
        if b['left']==b['right']=='reflecting':
            u=x%(2*self.length);flip=u>=self.length
            x=np.where(flip,2*self.length-u,u);v=np.where(flip,-v,v)
        else:
            if b['left']=='reflecting':
                hit=x<0;x=np.where(hit,-x,x);v=np.where(hit,-v,v)
            if b['right']=='reflecting':
                hit=x>=self.length;x=np.where(hit,2*self.length-x,x);v=np.where(hit,-v,v)
        left=x<0;right=x>=self.length
        if b['right']=='reflecting':
            # x=L after reflection belongs just inside the domain.
            x=np.where(x==self.length,np.nextafter(self.length,0),x);right=x>=self.length
        keep=~(left|right)
        return x[keep],v[keep],keep,int(left.sum()),int(right.sum())

    def step(self):
        _,e=self.poisson(self.charge_density())
        staged=[]
        for p in self.populations:
            s=p['spec']
            half=p['v']+.5*self.dt*self.acceleration(p['x'],e,s)
            x,half,_,nl,nr=self.boundaries(p['x']+self.dt*half,half)
            p['absorbed_left']+=nl;p['absorbed_right']+=nr
            lengths=np.full(len(x),self.dt)
            xs=[x];vs=[half];hs=[lengths]
            for source in s['sources']:
                rate=source['number_flux']/s['particle_weight']
                first=int(np.floor(self.time*rate+.5+1e-9))
                stop=int(np.floor((self.time+self.dt)*rate+.5+1e-9))
                tau=np.clip(self.time+self.dt-(np.arange(first,stop)+.5)/rate,0,self.dt)
                left=source['boundary']=='left'
                a=s['charge']/s['mass']*e[0 if left else -1]
                x0=0. if left else self.length
                new_x=x0+source['velocity']*tau+.5*a*tau**2
                new_half=np.full(len(tau),source['velocity'])+.5*a*tau
                new_x,new_half,keep,nl,nr=self.boundaries(new_x,new_half)
                p['injected']+=len(tau);p['absorbed_left']+=nl;p['absorbed_right']+=nr
                xs.append(new_x);vs.append(new_half);hs.append(tau[keep])
            p['x']=np.concatenate(xs)
            staged.append((np.concatenate(vs),np.concatenate(hs)))
        _,end_e=self.poisson(self.charge_density())
        for p,(half,h) in zip(self.populations,staged):
            p['v']=half+.5*h*self.acceleration(p['x'],end_e,p['spec'])
        self.index+=1

    def integrate(self,values):
        return self.dx*(values.sum() if self.periodic else values.sum()-.5*(values[0]+values[-1]))

    def diagnostics(self):
        rho=self.charge_density();phi,e=self.poisson(rho)
        electric=.5*self.integrate(e*e)
        row=dict(time=self.time,electric=electric,field_rms=np.sqrt(2*electric/self.length),
                 net_charge=self.integrate(rho),kinetic=0.,momentum=0.,particles=0)
        for k,p in enumerate(self.populations):
            s=p['spec'];w=s['particle_weight'];n=len(p['x'])
            kinetic=.5*s['mass']*w*np.sum(p['v']**2)
            row['kinetic']+=kinetic;row['momentum']+=s['mass']*w*p['v'].sum();row['particles']+=n
            row[f'p{k}_particles']=n;row[f'p{k}_number']=n*w;row[f'p{k}_kinetic']=kinetic
            for key in ('injected','absorbed_left','absorbed_right'):row[f'p{k}_{key}']=p[key]
            row[f'p{k}_balance_error']=n-s['particles']-p['injected']+p['absorbed_left']+p['absorbed_right']
        row['total']=row['kinetic']+row['electric']
        return row,rho,phi,e


def run(config,input_directory='.',output_override=None):
    from pic_io import HDF5Writer
    sim=PIC(config,input_directory)
    c=sim.config
    output=Path(output_override) if output_override else Path(input_directory)/c['output']['directory']
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    steps=round(c['time']['end']/sim.dt)
    with HDF5Writer(output/'run.h5',sim) as writer:
        for i in range(steps+1):
            writer.record(sim)
            if i%max(1,steps//4)==0:print(f't={sim.time:g}/{c["time"]["end"]:g}',flush=True)
            if i<steps:sim.step()
        writer.finish(sim)
        writer.write_csv(output/'diagnostics.csv')
    print(f'Saved {output}/run.h5 (plot separately with pic_plots.py)',flush=True)
    return sim,output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,help='JSON simulation input')
    parser.add_argument('--check',action='store_true',help='Validate and initialize without time stepping or writing output')
    parser.add_argument('--output',type=Path,help='Override output directory (relative to current terminal directory)')
    args=parser.parse_args()
    try:
        c=json.loads(args.input.read_text(encoding='utf-8-sig'))
        if args.check:
            sim=PIC(c,args.input.resolve().parent)
            print(f'Valid input: {len(sim.populations)} populations, {sim.cells} cells, dt={sim.dt:g}')
        else:run(c,args.input.resolve().parent,args.output)
    except (ValueError,KeyError,OSError) as error:parser.exit(2,f'Input/run error: {error}\n')


if __name__=='__main__':main()
