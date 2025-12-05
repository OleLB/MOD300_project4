"""Generate a view of the Milky Way Galaxy"""

from astropy import units as u
from mw_plot import MWFaceOn

def gen_milkyway():
    """
    Source: https://milkyway-plot.readthedocs.io/en/stable/
    """
    mw1 = MWFaceOn(
        radius=20 * u.kpc, # pylint: disable=no-member
        unit=u.kpc, # pylint: disable=no-member
        coord="galactocentric",
        annotation=True,
        figsize=(10, 8),
    )

    mw1.title = "Milky Way Galaxy"
    mw1.scatter(8 * u.kpc, 0 * u.kpc, c="r", s=2) # pylint: disable=no-member
