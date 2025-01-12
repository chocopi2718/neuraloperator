# from .burgers import load_burgers
from .wave1d_2 import load_wave1d
from .darcy import load_darcy_pt, load_darcy_flow_small
from .spherical_swe import load_spherical_swe
from .navier_stokes import load_navier_stokes_pt 
#, load_navier_stokes_zarr
# from .navier_stokes import load_navier_stokes_hdf5

# from .positional_encoding import append_2d_grid_positional_encoding, get_grid_positional_encoding
from .pt_dataset import load_pt_traintestsplit
