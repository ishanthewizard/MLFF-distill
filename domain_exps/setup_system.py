import numpy as np

import ase.io
import ase.units as units
from ase.build import bulk
from ase.constraints import ExpCellFilter
from ase.optimize.precon import PreconLBFGS

from nglview import show_ase

from matscipy.fracture_mechanics.crack import CubicCrystalCrack
from matscipy.fracture_mechanics.clusters import diamond, set_groups
from matscipy.calculators.manybody.explicit_forms.stillinger_weber import StillingerWeber, Holland_Marder_PRL_80_746_Si
from matscipy.calculators.manybody import Manybody
from matscipy.elasticity import fit_elastic_constants

calc = Manybody(**StillingerWeber(Holland_Marder_PRL_80_746_Si))

el = 'Si' # chemical element
si = bulk(el, cubic=True)
si.calc = calc
ecf = ExpCellFilter(si)
opt = PreconLBFGS(ecf)
opt.run(fmax=1e-6, smax=1e-6)
a0 = si.cell[0, 0]

si = bulk(el, a=a0)
si.calc = calc
C, C_err = fit_elastic_constants(si, symmetry='cubic')

C_11 = C[0, 0] / units.GPa
C_12 = C[0, 1] / units.GPa
C_44 = C[3, 3] / units.GPa
(C_11, C_12, C_44)

# UPDATE to import from here once `adapticecont` branch is merged.
# from matscipy.fracture_mechanics.clusters import find_surface_energy

def find_surface_energy(symbol,calc,a0,surface,size=(8,1,1),vacuum=10,fmax=0.0001,unit='0.1J/m^2'):
    # Import required lattice builder
    if surface.startswith('bcc'):
        from ase.lattice.cubic import BodyCenteredCubic as lattice_builder
    elif surface.startswith('fcc'):
        from ase.lattice.cubic import FaceCenteredCubic as lattice_builder #untested
    elif surface.startswith('diamond'):
        from ase.lattice.cubic import Diamond as lattice_builder #untested
    ## Append other lattice builders here
    else:
        print('Error: Unsupported lattice ordering.')

    # Set orthogonal directions for cell axes
    if surface.endswith('100'):
        directions=[[1,0,0], [0,1,0], [0,0,1]] #tested for bcc
    elif surface.endswith('110'):
        directions=[[1,1,0], [-1,1,0], [0,0,1]] #tested for bcc
    elif surface.endswith('111'):
        directions=[[1,1,1], [-2,1,1],[0,-1,1]] #tested for bcc
    ## Append other cell axis options here
    else:
        print('Error: Unsupported surface orientation.')
    
    # Make bulk and slab with same number of atoms (size)
    bulk = lattice_builder(directions=directions, size=size, symbol=symbol, latticeconstant=a0, pbc=(1,1,1))
    cell = bulk.get_cell() ; cell[0,:] *=2 # vacuum along x axis (surface normal)
    slab = bulk.copy() ; slab.set_cell(cell)
    
    # Optimize the geometries
    from ase.optimize import LBFGSLineSearch
    bulk.calc = calc ; opt_bulk = LBFGSLineSearch(bulk) ; opt_bulk.run(fmax=fmax)
    slab.calc = calc ; opt_slab = LBFGSLineSearch(slab) ; opt_slab.run(fmax=fmax)

    # Find surface energy
    import numpy as np
    Ebulk = bulk.get_potential_energy() ; Eslab = slab.get_potential_energy()
    area = np.linalg.norm(np.cross(slab.get_cell()[1,:],slab.get_cell()[2,:]))
    gamma_ase = (Eslab - Ebulk)/(2*area)

    # Convert to required units
    if unit == 'ASE':
        return [gamma_ase,'ase_units']
    else:
        from ase import units
        gamma_SI = (gamma_ase / units.J ) * (units.m)**2
        if unit =='J/m^2':
            return [gamma_SI,'J/m^2']
        elif unit == '0.1J/m^2':
            return [10*gamma_SI,'0.1J/m^2'] # units required for the fracture code
        else:
            print('Error: Unsupported unit of surface energy.')

surface_energy, unit = find_surface_energy(el, calc, a0, 'diamond111')

n               = [ 4, 4, 1 ]
crack_surface   = [ 1, 1, 1 ]
crack_front     = [ 1, -1, 0 ]
crack_tip       = [ 41, 56 ]
skin_x, skin_y = 1, 1

vacuum          = 6.0
fmax            = 0.05

# Setup crack system
cryst = diamond(el, a0, n, crack_surface, crack_front)
set_groups(cryst, n, skin_x, skin_y)

ase.io.write('cryst.cfg', cryst)

# Compute crack tip position
r0 = np.sum(cryst.get_positions()[crack_tip,:], axis=0)/len(crack_tip)
tip_x0, tip_y0, tip_z0 = r0

crk = CubicCrystalCrack(crack_surface, crack_front,
                        C_11, C_12, C_44)
k1g = crk.k1g(surface_energy)
k1g

tip_x = cryst.cell.diagonal()[0]/2
tip_y = cryst.cell.diagonal()[1]/2

crack = cryst.copy()
crack.set_pbc([False, False, True])

k1 = 1.0

ux, uy = crk.displacements(cryst.positions[:,0], cryst.positions[:,1],
                            tip_x, tip_y, k1*k1g)
crack.positions[:, 0] += ux
crack.positions[:, 1] += uy

# Center notched configuration in simulation cell and ensure enough vacuum.
oldr = crack[0].position.copy()
crack.center(vacuum=vacuum, axis=0)
crack.center(vacuum=vacuum, axis=1)
tip_x += crack[0].x - oldr[0]
tip_y += crack[0].y - oldr[1]

show_ase(crack)