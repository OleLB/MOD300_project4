"""Task 2: Generate visualizations of sector in the milky way starting in different centers"""
import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWSkyMap


def gen_milkyway_sector(
        sector_center: str,
        radius: int,
        background: str = "Mellinger color optical survey",
        save: bool = True
    ):
    """
    Create a view of a sector in the milky way galaxy.
    
    Parameters
    ----------
    sector_center : str
    radius : int
    background : str (default="Mellinger color optical survey")
    save : bool (default=True), whether to save the generated figure as a PNG file.

    Source: https://milkyway-plot.readthedocs.io/en/stable/
    """

    sector = MWSkyMap(
        center=sector_center,
        radius=(radius, radius) * u.arcsec, # pylint: disable=no-member
        background=background,
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    sector.transform(ax)
    if save:
        sector.savefig(f'images/{sector_center}_{radius}.png')
    return fig


if __name__ == "__main__":
    gen_milkyway_sector("M8", 4000)
