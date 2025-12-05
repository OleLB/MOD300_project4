"""Task 2: Generate visualizations of sector in the milky way starting in different centers"""
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from mw_plot import MWSkyMap


def gen_milkyway_sector(sector_center: str, radius: int, background: str = "Mellinger color optical survey", save: bool = True):
    """
    Generate a sector view of the milky way galaxy.
    
    Parameters
    ----------
    sector_center : str
    radius : int
    background : str (default="Mellinger color optical survey")
    """

    # Check if centnter is a valid known object
    center_info = SkyCoord.from_name(sector_center)
    if not center_info:
        raise Exception(f"Could not resolve '{sector_center}' to a known object.")

    sector = MWSkyMap(
        center=sector_center,
        radius=(radius, radius) * u.arcsec,
        background=background,
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    sector.transform(ax)
    if save:
        sector.savefig(f'images/{sector_center}_{radius}.png')
    return fig


if __name__ == "__main__":
    gen_milkyway_sector("M8", 4000)