"""Run with: python -m unittest test_pic_solver.py"""
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import h5py
from pic_solver import PIC,validate,run

HERE=Path(__file__).resolve().parent


def example(name):return json.loads((HERE/'inputs'/f'{name}.json').read_text())


class NumericalTests(unittest.TestCase):
    def test_all_inputs(self):
        for path in (HERE/'inputs').glob('*.json'):
            s=PIC(json.loads(path.read_text()),path.parent)
            s.step()
            self.assertTrue(np.isfinite(s.diagnostics()[0]['total']))

    def test_cic_charge_integral(self):
        for name in ('two_stream','pierce_single'):
            s=PIC(example(name))
            for p in s.populations:p['x']=np.array([0.,.001,s.length*.3,np.nextafter(s.length,0)])
            expected=s.config['background_charge_density']*s.length+sum(p['spec']['charge']*p['spec']['particle_weight']*len(p['x']) for p in s.populations)
            self.assertAlmostEqual(s.integrate(s.charge_density()),expected,places=12)

    def test_periodic_poisson(self):
        s=PIC(example('two_stream'));k=2*np.pi/s.length
        rho=np.cos(k*s.nodes)
        phi,e=s.poisson(rho)
        np.testing.assert_allclose(e,np.sin(k*s.nodes)/k,atol=1e-13)
        np.testing.assert_allclose(phi,np.cos(k*s.nodes)/k**2,atol=1e-13)

    def test_biased_dirichlet_poisson(self):
        c=example('pierce_single');c['domain'].update(potential_left=2.,potential_right=5.)
        s=PIC(c);phi,e=s.poisson(np.ones(len(s.nodes)))
        x=s.nodes;L=s.length
        np.testing.assert_allclose(phi,2+3*x/L+.5*x*(L-x),atol=1e-11)
        np.testing.assert_allclose(e,-3/L+x-L/2,atol=1e-11)

    def test_reflection_multiple_crossings_and_mixed_wall(self):
        c=example('reflecting_plasma');s=PIC(c)
        x,v,keep,nl,nr=s.boundaries(np.array([-1.,11.,21.,-21.]),np.ones(4))
        np.testing.assert_allclose(x,[1,9,1,1]);np.testing.assert_allclose(v,[-1,-1,1,-1])
        self.assertTrue(keep.all());self.assertEqual(nl+nr,0)
        s.config['domain']['particle_boundary']['right']='absorbing'
        x,v,keep,nl,nr=s.boundaries(np.array([-21.,-1.,11.]),np.ones(3))
        np.testing.assert_allclose(x,[1]);self.assertEqual(nr,2)

    def test_mass_and_charge_affect_force(self):
        s=PIC(example('reflecting_plasma'));x=np.array([1.,4.,8.]);e=np.full(len(s.nodes),2.)
        np.testing.assert_allclose(s.acceleration(x,e,dict(charge=-1,mass=1)),-2)
        np.testing.assert_allclose(s.acceleration(x,e,dict(charge=2,mass=4)),1)

    def test_injected_equilibrium_and_balance(self):
        for name in ('pierce_single','pierce_two_stream'):
            c=example(name)
            c['domain']['cells']=128  # commensurate quiet load: particles/cells is integral
            for p in c['populations']:
                p['particles']=2048
                p['initial']['velocity'].pop('perturbation')
            s=PIC(c)
            for _ in range(100):s.step()
            self.assertLess(np.max(abs(s.poisson(s.charge_density())[1])),1e-10)
            for k,p in enumerate(s.populations):
                self.assertEqual(s.diagnostics()[0][f'p{k}_balance_error'],0)
                np.testing.assert_allclose(p['v'],p['spec']['initial']['velocity']['drift'],atol=1e-10)

    def test_landau_quiet_loading(self):
        s=PIC(example('landau'));v=s.populations[0]['v']
        self.assertAlmostEqual(v.mean(),0.,places=12)
        self.assertLess(abs(np.mean(v*v)-1),.002)
        e=s.poisson(s.charge_density())[1]
        self.assertLess(abs(2*abs(np.fft.rfft(e)[1])/s.cells-.02),1e-4)

    def test_zero_initial_population_and_custom_file(self):
        c=example('pierce_single');p=c['populations'][0]
        p.update(particles=0,density=0,particle_weight=.001)
        s=PIC(c);s.step()
        self.assertGreater(len(s.populations[0]['x']),0)
        self.assertEqual(s.diagnostics()[0]['p0_balance_error'],0)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            assert Path(directory).resolve().parent == HERE
            p.update(particles=4,density=1);p.pop('particle_weight')
            p['initial']={'file':'custom.npz'}
            np.savez(Path(directory)/'custom.npz',x=[1.,2.,3.,4.],v=[-1.,0.,1.,2.])
            s=PIC(c,directory)
            np.testing.assert_array_equal(s.populations[0]['v'],[-1,0,1,2])

    def test_invalid_input_is_rejected(self):
        c=example('two_stream');c['domain']['cellz']=64
        with self.assertRaises(ValueError):validate(c)
        c=example('two_stream');c['background_charge_density']=0
        with self.assertRaises(ValueError):validate(c)
        c=example('pierce_single');c['populations'][0]['sources'][0]['velocity']=-1
        with self.assertRaises(ValueError):validate(c)
        c=example('landau');c['populations'][0]['particles']=100
        with self.assertRaises(ValueError):validate(c)

    def test_output_and_empty_domain(self):
        c=example('pierce_single');p=c['populations'][0]
        p.update(particles=0,density=0,particle_weight=.001,sources=[])
        c['time']=dict(dt=.005,end=.01,diagnostic_interval=.005)
        c['output'].update(save_interval=.005,particle_dump='none')
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            assert Path(directory).resolve().parent == HERE
            s,out=run(c,output_override=directory)
            with h5py.File(out/'run.h5','r') as f:
                self.assertTrue(f.attrs['complete'])
                self.assertEqual(f['fields/electric'].shape,(3,385))
                self.assertEqual(len(f['final_particles/p000/x']),0)
            self.assertEqual(s.diagnostics()[0]['particles'],0)

    def test_output_cadences_and_distribution_normalization(self):
        c=example('two_stream')
        c['time']=dict(dt=.005,end=.03,diagnostic_interval=.01)
        c['output'].update(save_interval=.01,field_interval=.005,
            distribution_interval=.015,particle_interval=.02,display_particles=2)
        c['phase_space'].update(velocity_min=-.5,velocity_max=.5)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            assert Path(directory).resolve().parent == HERE
            _,out=run(c,output_override=directory)
            with h5py.File(out/'run.h5','r') as f:
                np.testing.assert_allclose(f['fields/time'][:],np.arange(7)*.005)
                np.testing.assert_allclose(f['distribution/time'][:],[0,.015,.03])
                np.testing.assert_allclose(f['diagnostics/time'][:],[0,.01,.02,.03])
                groups=list(f['particles'].values())
                np.testing.assert_allclose([g.attrs['time'] for g in groups],[0,.02,.03])
                self.assertEqual(sum(len(g['x']) for g in groups[0].values()),2)
                d=f['distribution']
                volume=np.diff(d['x_edges'][:])[:,None]*np.diff(d['v_edges'][:])[None,:]
                represented=(d['f'][:]*volume).sum(axis=(-2,-1))
                np.testing.assert_allclose(represented,d['number_in_range'][:])
                np.testing.assert_allclose(represented+d['number_below_range'][:]+d['number_above_range'][:],d['number'][:])
                self.assertGreater(d['number_above_range'][:].sum(),0)

    def test_phase_histogram_uses_all_particles(self):
        c=example('landau')
        c['time']=dict(dt=.005,end=.01,diagnostic_interval=.005)
        c['output'].update(save_interval=.005,display_particles=1,particle_dump='all')
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            assert Path(directory).resolve().parent == HERE
            s,out=run(c,output_override=directory)
            with h5py.File(out/'run.h5','r') as f:
                p=s.populations[0];d=f['distribution'];x=d['x_edges'][:];v=d['v_edges'][:]
                h=np.histogram2d(p['x'],p['v'],bins=(x,v))[0]*p['spec']['particle_weight']
                np.testing.assert_allclose(d['f'][-1,0]*np.diff(x)[:,None]*np.diff(v)[None,:],h,atol=1e-15)
                self.assertEqual(len(f['particles/step_000000002/p000/x']),p['spec']['particles'])
            # HDF5 can also supply arbitrary initial particles to a new run.
            c['populations'][0]['initial']={'file':str(out/'run.h5'),'group':'initial_particles/p000'}
            loaded=PIC(c)
            self.assertEqual(len(loaded.populations[0]['x']),262144)

    def test_time_refinement_in_harmonic_field(self):
        errors=[]
        for dt in (.1,.05,.025):
            c=example('pierce_single');c['domain'].update(length=10.,cells=64)
            c['time']=dict(dt=dt,end=1.,diagnostic_interval=dt)
            c['output']['save_interval']=dt
            p=c['populations'][0];p.update(density=1e-12,particles=1,sources=[])
            s=PIC(c);s.populations[0]['x']=np.array([6.]);s.populations[0]['v']=np.array([0.])
            for _ in range(round(1/dt)):s.step()
            errors.append(abs(s.populations[0]['x'][0]-(5+np.cos(1))))
        self.assertGreater(errors[0]/errors[1],3.9)
        self.assertGreater(errors[1]/errors[2],3.9)

    def test_output_frequency_does_not_change_dynamics(self):
        c=example('two_stream')
        c['time']=dict(dt=.005,end=.025,diagnostic_interval=.005)
        c['output'].update(save_interval=.005,particle_dump='sample',display_particles=1)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            assert Path(directory).resolve().parent == HERE
            first,_=run(c,output_override=Path(directory)/'frequent')
            c['output'].update(save_interval=.02,particle_dump='none')
            c['time']['diagnostic_interval']=.01
            second,_=run(c,output_override=Path(directory)/'sparse')
            for a,b in zip(first.populations,second.populations):
                np.testing.assert_array_equal(a['x'],b['x'])
                np.testing.assert_array_equal(a['v'],b['v'])


if __name__=='__main__':unittest.main()
