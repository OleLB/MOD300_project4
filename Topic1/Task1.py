import numpy as np
from astropy import units as u
from mw_plot import MWFaceOn

def gen_milkyway():
    mw1 = MWFaceOn(
        radius=20 * u.kpc,
        unit=u.kpc,
        coord="galactocentric",
        annotation=True,
        figsize=(10, 8),
    )

    mw1.title = "Bird's Eyes View"

    mw1.scatter(8 * u.kpc, 0 * u.kpc, c="r", s=2)