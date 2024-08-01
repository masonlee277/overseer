import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
from typing import Dict, List, Tuple

def create_fake_data():
    # Create fake elevation data (DEM)
    dem = np.random.rand(100, 100) * 1000  # 100x100 grid with elevations 0-1000m
    
    # Create fake aspect data (ASP)
    asp = np.random.rand(100, 100) * 360  # 0-360 degrees
    
    # Create fake canopy bulk density data (CBD)
    cbd = np.random.rand(100, 100) * 0.5  # 0-0.5 kg/m^3
    
    # Create fake canopy base height data (CBH)
    cbh = np.random.rand(100, 100) * 10  # 0-10 meters
    
    # Create fake canopy cover data (CC)
    cc = np.random.rand(100, 100) * 100  # 0-100 percent
    
    # Create fake canopy height data (CH)
    ch = np.random.rand(100, 100) * 30  # 0-30 meters
    
    # Create fake fuel model data (FBFM)
    fbfm = np.random.randint(1, 41, (100, 100))  # 40 standard fuel models
    
    # Create fake slope data (SLP)
    slp = np.random.rand(100, 100) * 45  # 0-45 degrees
    
    # Create fake wind speed data (WS)
    ws = np.random.rand(100, 100) * 20  # 0-20 m/s
    
    # Create fake wind direction data (WD)
    wd = np.random.rand(100, 100) * 360  # 0-360 degrees
    
    # Create fake fuel moisture data (M1, M10, M100)
    m1 = np.random.rand(100, 100) * 30  # 0-30 percent
    m10 = np.random.rand(100, 100) * 30  # 0-30 percent
    m100 = np.random.rand(100, 100) * 30  # 0-30 percent
    
    # Create fake road network
    roads = gpd.GeoDataFrame(
        geometry=[LineString([(np.random.rand()*100, np.random.rand()*100) for _ in range(2)]) for _ in range(10)],
        crs="EPSG:32610"
    )
    
    # Create fake population centers
    population_centers = gpd.GeoDataFrame(
        geometry=[Point(np.random.rand()*100, np.random.rand()*100) for _ in range(5)],
        crs="EPSG:32610"
    )

    fire_size = (100, 100)
    fire_center = (50, 50)
    fire_radius = 10
    fire = create_circular_fire(fire_size, fire_center, fire_radius)

    return {
        'dem': dem, 'asp': asp, 'cbd': cbd, 'cbh': cbh, 'cc': cc, 'ch': ch,
        'fbfm': fbfm, 'slp': slp, 'ws': ws, 'wd': wd, 'm1': m1, 'm10': m10, 'm100': m100,
        'roads': roads, 'population_centers': population_centers,
        'fire': fire
    }

def create_circular_fire(size: Tuple[int, int], center: Tuple[int, int], radius: int) -> np.ndarray:
    y, x = np.ogrid[:size[0], :size[1]]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    fire = (dist_from_center <= radius).astype(int)
    return fire

def visualize_multiple(matrices: Dict[str, np.ndarray], num_cols: int = 2):
    num_plots = len(matrices)
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten() if num_plots > 1 else [axes]

    for ax, (title, matrix) in zip(axes, matrices.items()):
        if title == "Sampled Firelines":
            ax.imshow(matrices["Initial Fire"], cmap='YlOrRd', alpha=0.7)
            ax.imshow(matrices["Action Mask"], cmap='Blues', alpha=0.3)
            for x, y, angle, length in matrix:
                dx = length * np.cos(angle)
                dy = length * np.sin(angle)
                ax.plot([x, x + dx], [y, y + dy], 'g-', linewidth=2)
        elif matrix.ndim == 2:
            im = ax.imshow(matrix, cmap='viridis')
        elif matrix.ndim == 3 and matrix.shape[0] == 2:  # For gradient visualization
            magnitude = np.sqrt(matrix[0]**2 + matrix[1]**2)
            im = ax.imshow(magnitude, cmap='viridis')
            ax.quiver(np.arange(0, matrix.shape[2], 5), np.arange(0, matrix.shape[1], 5),
                      matrix[1, ::5, ::5], matrix[0, ::5, ::5],
                      color='white', scale=20, headwidth=4, headlength=6)
        
        ax.set_title(title)
        if title != "Sampled Firelines":
            plt.colorbar(im, ax=ax)

    # Hide any unused subplots
    for ax in axes[num_plots:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()