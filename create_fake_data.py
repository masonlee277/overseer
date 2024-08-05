import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from datetime import datetime

def create_mock_data():
    base_dir = os.path.join('data', 'mock')
    input_dir = os.path.join(base_dir, 'inputs')
    output_dir = os.path.join(base_dir, 'outputs')

    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Define common parameters for all GeoTIFF files
    width, height = 100, 100
    transform = from_origin(0, 0, 30, 30)  # 30m resolution
    crs = rasterio.crs.CRS.from_epsg(32610)  # EPSG:32610 - UTM Zone 10N

    # Create input files
    input_files = ['asp', 'cbd', 'cbh', 'cc', 'ch', 'dem', 'fbfm40', 'slp', 'adj', 'new_phi', 'ws', 'wd', 'm1', 'm10', 'm100']
    for filename in input_files:
        data = np.ones((height, width))
        output_path = os.path.join(input_dir, f'{filename}.tif')
        with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=data.dtype, crs=crs, transform=transform) as dst:
            dst.write(data, 1)
        print(f"Created mock input file: {output_path}")

    # Create output files
    output_files = ['flin', 'spread_rate', 'time_of_arrival']
    for filename in output_files:
        data = np.ones((height, width))
        output_path = os.path.join(output_dir, f'{filename}.tif')
        with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=data.dtype, crs=crs, transform=transform) as dst:
            dst.write(data, 1)
        print(f"Created mock output file: {output_path}")

    # Create elmfire.data.in file
    elmfire_data_in_path = os.path.join(base_dir, 'elmfire.data.in')
    with open(elmfire_data_in_path, 'w') as f:
        f.write("&INPUTS\n")
        f.write(f"  FUELS_AND_TOPOGRAPHY_DIRECTORY = '{input_dir}'\n")
        for filename in input_files:
            f.write(f"  {filename.upper()}_FILENAME = '{filename}.tif'\n")
        f.write(f"  DT_METEOROLOGY = 3600.0\n")
        f.write(f"  WEATHER_DIRECTORY = '{input_dir}'\n")
        f.write(f"  LH_MOISTURE_CONTENT = 30.0\n")
        f.write(f"  LW_MOISTURE_CONTENT = 60.0\n")
        f.write("/\n\n")
        f.write("&OUTPUTS\n")
        f.write(f"  OUTPUTS_DIRECTORY = '{output_dir}'\n")
        f.write("  DTDUMP = 3600.\n")
        f.write("  DUMP_FLIN = .TRUE.\n")
        f.write("  DUMP_SPREAD_RATE = .TRUE.\n")
        f.write("  DUMP_TIME_OF_ARRIVAL = .TRUE.\n")
        f.write("  CONVERT_TO_GEOTIFF = .TRUE.\n")
        f.write("  DUMP_SPOTTING_OUTPUTS = .TRUE.\n")
        f.write("/\n")
    print(f"Created mock elmfire.data.in file: {elmfire_data_in_path}")

if __name__ == "__main__":
    create_mock_data()