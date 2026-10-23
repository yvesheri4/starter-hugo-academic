"""HDF5 output schema 1.0. Distributions are computed from ALL particles.

f is number density per unit velocity, not a probability density:
sum(f * dx_bin * dv_bin) = represented particle number inside velocity bounds.
"""
import csv
import json
import platform
import numpy as np
import h5py


def schedule(end,dt,interval):
    steps=round(end/dt);stride=round(interval/dt)
    return sorted(set(range(0,steps+1,stride))|{steps})


class HDF5Writer:
    def __init__(self,path,sim):
        self.file=h5py.File(path,'w')
        f=self.file;c=sim.config;o=c['output'];ps=c['phase_space']
        f.attrs.update(format='electrostatic_1d1v_pic',schema_version='1.0',complete=False,
            normalization='omega_p0=1, epsilon0=1, electron_mass=1, elementary_charge=1; reference speed v0',
            time_unit='1/omega_p0',length_unit='v0/omega_p0',velocity_unit='v0',
            number_density_unit='n0',potential_unit='m_e*v0^2/e',
            electric_field_unit='m_e*v0*omega_p0/e',distribution_unit='n0/v0',
            energy_unit='n0*m_e*v0^3/omega_p0 (per unit transverse area)',
            python=platform.python_version(),numpy=np.__version__,h5py=h5py.__version__)
        f.create_dataset('configuration_json',data=json.dumps(c),dtype=h5py.string_dtype())
        pop=f.create_group('populations')
        pop.create_dataset('name',data=[p['spec']['name'] for p in sim.populations],dtype=h5py.string_dtype())
        for key in ('charge','mass','particle_weight'):
            pop.create_dataset(key,data=[p['spec'][key] for p in sim.populations])
        self.schedules={key:schedule(c['time']['end'],sim.dt,o[key+'_interval']) for key in ('field','distribution','particle')}
        self.lookup={k:{step:j for j,step in enumerate(v)} for k,v in self.schedules.items()}
        self.diag_steps=set(schedule(c['time']['end'],sim.dt,c['time']['diagnostic_interval']))
        self.rows=[];self.rng=np.random.default_rng(c['seed']+100000)
        fg=f.create_group('fields')
        ft=fg.create_dataset('time',data=np.array(self.schedules['field'])*sim.dt);ft.make_scale('time')
        x=fg.create_dataset('x',data=sim.nodes);x.make_scale('x')
        for key,unit in (('electric','m_e*v0*omega_p0/e'),('potential','m_e*v0^2/e'),('charge_density','e*n0')):
            ds=fg.create_dataset(key,shape=(len(ft),len(x)),dtype='f8',chunks=(1,len(x)),compression='gzip',shuffle=True)
            ds.attrs['unit']=unit;ds.dims[0].attach_scale(ft);ds.dims[1].attach_scale(x)
        fg.create_dataset('electric_energy',shape=(len(ft),),dtype='f8')
        dg=f.create_group('distribution')
        dt=dg.create_dataset('time',data=np.array(self.schedules['distribution'])*sim.dt);dt.make_scale('time')
        self.x_edges=np.linspace(0,sim.length,ps['position_bins']+1)
        self.v_edges=np.linspace(ps['velocity_min'],ps['velocity_max'],ps['velocity_bins']+1)
        self.bin_volume=(self.x_edges[1]-self.x_edges[0])*(self.v_edges[1]-self.v_edges[0])
        dg.create_dataset('x_edges',data=self.x_edges);dg.create_dataset('v_edges',data=self.v_edges)
        xc=dg.create_dataset('x',data=(self.x_edges[:-1]+self.x_edges[1:])/2);xc.make_scale('x')
        vc=dg.create_dataset('v',data=(self.v_edges[:-1]+self.v_edges[1:])/2);vc.make_scale('v')
        pc=dg.create_dataset('population',data=np.arange(len(sim.populations)));pc.make_scale('population')
        ds=dg.create_dataset('f',shape=(len(dt),len(pc),len(xc),len(vc)),dtype='f8',
                            chunks=(1,1,len(xc),len(vc)),compression='gzip',shuffle=True)
        ds.attrs.update(axis_order='time,population,x,v',unit='n0/v0',
            normalization='sum(f*dx_bin*dv_bin) = number in velocity range; no smoothing',
            sampling='all solver particles; independent of particle_dump and display_particles')
        for i,scale in enumerate((dt,pc,xc,vc)):ds.dims[i].attach_scale(scale)
        for key in ('number','number_in_range','number_below_range','number_above_range'):
            dg.create_dataset(key,shape=(len(dt),len(pc)),dtype='f8')
        f.create_group('particles').attrs.update(mode=o['particle_dump'],
            warning='Sampled particle dumps are for display only. Use distribution/f for quantitative distributions.')
        self.write_particles(f.create_group('initial_particles'),sim,full=True)
        f.flush()

    def __enter__(self):return self
    def __exit__(self,*args):self.file.close()

    def write_particles(self,group,sim,full=False):
        group.attrs['time']=sim.time
        group.attrs['sampled']=not full
        counts=[len(p['x']) for p in sim.populations]
        total=sum(counts)
        chosen=None if full else self.rng.choice(total,min(total,sim.config['output']['display_particles']),replace=False)
        offset=0
        for k,p in enumerate(sim.populations):
            g=group.create_group(f'p{k:03d}')
            index=slice(None) if full else chosen[(chosen>=offset)&(chosen<offset+counts[k])]-offset
            offset+=counts[k]
            g.attrs.update(name=p['spec']['name'],total_particles=counts[k],particle_weight=p['spec']['particle_weight'])
            for key in ('x','v'):g.create_dataset(key,data=p[key][index],compression='gzip',shuffle=True)

    def record(self,sim):
        i=sim.index;f=self.file
        if i in self.lookup['field'] or i in self.diag_steps:
            row,rho,phi,e=sim.diagnostics()
            if not all(np.isfinite(v) for v in row.values()):raise RuntimeError('Nonfinite diagnostics; revise input resolution')
            if i in self.diag_steps:self.rows.append(row)
            if i in self.lookup['field']:
                j=self.lookup['field'][i]
                for key,value in (('electric',e),('potential',phi),('charge_density',rho),('electric_energy',row['electric'])):
                    f['fields'][key][j]=value
        if i in self.lookup['distribution']:
            j=self.lookup['distribution'][i];dg=f['distribution']
            for k,p in enumerate(sim.populations):
                w=p['spec']['particle_weight']
                h=np.histogram2d(p['x'],p['v'],bins=(self.x_edges,self.v_edges))[0]
                dg['f'][j,k]=h*w/self.bin_volume
                dg['number'][j,k]=len(p['x'])*w
                dg['number_in_range'][j,k]=h.sum()*w
                dg['number_below_range'][j,k]=np.count_nonzero(p['v']<self.v_edges[0])*w
                dg['number_above_range'][j,k]=np.count_nonzero(p['v']>self.v_edges[-1])*w
        mode=sim.config['output']['particle_dump']
        if i in self.lookup['particle'] and mode!='none':
            self.write_particles(f['particles'].create_group(f'step_{i:09d}'),sim,full=mode=='all')
        f.attrs['last_completed_step']=i
        if i in self.lookup['field'] or i in self.lookup['distribution']:f.flush()

    def finish(self,sim):
        self.write_particles(self.file.create_group('final_particles'),sim,full=True)
        dg=self.file.create_group('diagnostics')
        for key in self.rows[0]:dg.create_dataset(key,data=[r[key] for r in self.rows])
        self.file.attrs['complete']=True
        self.file.flush()

    def write_csv(self,path):
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(self.rows)
