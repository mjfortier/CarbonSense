import numpy as np
import json
from shapely.geometry import Polygon
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


class NpEncoder(json.JSONEncoder):
    '''JSON encoder used for metadata files'''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

def unformat_timestamp(ts, time=1200):
  # ts - string representation 'YYYY_MM_DD'
  # assume output is solar noon
  return int(f"{''.join(ts.split('-'))}{time}")


def get_polygon_wkt(lat, lon, offset_m) -> Polygon:
    """Generate a WKT polygon based on the provided lat, lon, and offset in meters.
    
    Args:
        lat (float): Latitude of the site in WGS84.
        lon (float): Longitude of the site in WGS84.
        offset_m (int): Number of meters to offset in each direction from the site.
        
    Returns:
        Polygon: A Shapely polygon representing the box around the site.
    """
    utm_crs_info = query_utm_crs_info(
        area_of_interest=AreaOfInterest(west_lon_degree=lon, south_lat_degree=lat, east_lon_degree=lon, north_lat_degree=lat),
        datum_name="WGS 84"
    )[0]
    utm_crs = CRS.from_epsg(utm_crs_info.code)
    transformer_utm_to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    transformer_wgs84_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    # Calculate box bounds
    x, y = transformer_wgs84_to_utm.transform(lon, lat)
    bottom_left = (x - offset_m, y - offset_m)
    bottom_right = (x + offset_m, y - offset_m)
    top_right = (x + offset_m, y + offset_m)
    top_left = (x - offset_m, y + offset_m)

    # Create the polygon in WGS84
    box = Polygon([bottom_left, bottom_right, top_right, top_left, bottom_left])
    lon_lat_coords = [transformer_utm_to_wgs84.transform(xx, yy) for xx, yy in zip(*box.exterior.xy)]
    geo_polygon = Polygon(lon_lat_coords)
    return geo_polygon