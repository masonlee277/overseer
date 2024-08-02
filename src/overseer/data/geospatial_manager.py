# src/overseer/geospatial/geospatial_manager.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import rasterio
import numpy as np
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape, Polygon, Point, LineString 

from scipy import ndimage
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from overseer.config.config import OverseerConfig
from overseer.utils.logging import OverseerLogger
from overseer.data.utils import create_fake_data, visualize_multiple

class GeoSpatialManager:
    """
    Manages geospatial data operations and calculations for the ELMFIRE simulation system.

    This class is responsible for:
    1. Loading and saving geospatial data (e.g., GeoTIFF files)
    2. Performing geospatial calculations and analysis
    3. Providing utility functions for geospatial operations
    4. Converting between different geospatial data formats

    It does NOT:
    1. Manage overall data flow or storage (handled by DataManager)
    2. Make decisions about when to perform calculations or save data
    3. Handle non-geospatial data
    4. Manage simulation state or episode data
    5. Interact directly with the ELMFIRE simulator
    """

    def __init__(self, config: OverseerConfig):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.crs = self.config.get('coordinate_reference_system', 'EPSG:4326')
        self.resolution = self.config.get('spatial_resolution', 30)  # in meters

    def load_tiff(self, filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a GeoTIFF file and return the data and metadata."""
        try:
            with rasterio.open(filepath) as src:
                data = src.read(1)  # Assuming single band
                metadata = src.meta
            self.logger.info(f"Loaded GeoTIFF from {filepath}")
            return data, metadata
        except Exception as e:
            self.logger.error(f"Failed to load GeoTIFF from {filepath}: {str(e)}")
            raise

    def save_tiff(self, filepath: str, data: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Save data as a GeoTIFF file."""
        try:
            with rasterio.open(filepath, 'w', **metadata) as dst:
                dst.write(data, 1)
            self.logger.info(f"Saved GeoTIFF to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save GeoTIFF to {filepath}: {str(e)}")
            raise

    def calculate_burned_area(self, fire_intensity: np.ndarray, threshold: float) -> float:
        """Calculate the total burned area based on fire intensity."""
        burned_pixels = np.sum(fire_intensity > threshold)
        area = burned_pixels * (self.resolution ** 2) / 10000  # Convert to hectares
        return area

    def calculate_fire_perimeter(self, fire_intensity: np.ndarray, threshold: float = 0) -> np.ndarray:
        """Calculate the fire perimeter based on fire intensity."""
        binary_fire = (fire_intensity > threshold).astype(np.uint8)
        perimeter = ndimage.binary_dilation(binary_fire) ^ binary_fire
        return perimeter

    def compute_fire_spread_direction(self, fire_intensity: np.ndarray) -> np.ndarray:
        """Compute the predominant fire spread direction."""
        grad_y, grad_x = np.gradient(fire_intensity)
        direction = np.arctan2(grad_y, grad_x)
        return direction

    def calculate_terrain_effects(self, elevation: np.ndarray, fire_intensity: np.ndarray) -> np.ndarray:
        """Calculate terrain effects on fire spread."""
        slope, aspect = self.calculate_slope_aspect(elevation)
        terrain_effect = np.sin(slope) * np.cos(aspect - self.compute_fire_spread_direction(fire_intensity))
        return terrain_effect

    def calculate_slope_aspect(self, elevation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate slope and aspect from elevation data."""
        dy, dx = np.gradient(elevation, self.resolution, self.resolution)
        slope = np.arctan(np.sqrt(dx*dx + dy*dy))
        aspect = np.arctan2(-dx, dy)
        return slope, aspect

    def compute_cumulative_burn_map(self, fire_intensities: List[np.ndarray]) -> np.ndarray:
        """Compute a cumulative burn map from a list of fire intensity arrays."""
        return np.maximum.reduce(fire_intensities)

    def calculate_fire_shape_complexity(self, fire_perimeter: np.ndarray) -> float:
        """Calculate the complexity of the fire shape using fractal dimension."""
        perimeter_length = np.sum(fire_perimeter)
        area = np.sum(ndimage.binary_fill_holes(fire_perimeter))
        fractal_dimension = 2 * np.log(perimeter_length / 4) / np.log(area)
        return fractal_dimension

    def identify_high_risk_areas(self, fire_intensity: np.ndarray, elevation: np.ndarray, fuel_type: np.ndarray) -> np.ndarray:
        """Identify high-risk areas based on fire intensity, elevation, and fuel type."""
        terrain_effect = self.calculate_terrain_effects(elevation, fire_intensity)
        risk = fire_intensity * terrain_effect * fuel_type
        return risk

    def calculate_fire_containment(self, fire_perimeter: np.ndarray, containment_lines: np.ndarray) -> float:
        """Calculate the percentage of fire perimeter that is contained."""
        contained_perimeter = np.sum(fire_perimeter & containment_lines)
        total_perimeter = np.sum(fire_perimeter)
        containment_percentage = (contained_perimeter / total_perimeter) * 100 if total_perimeter > 0 else 0
        return containment_percentage

    def interpolate_weather_data(self, weather_stations: gpd.GeoDataFrame, grid: np.ndarray) -> np.ndarray:
        """Interpolate weather data from point observations to a grid."""
        # This is a placeholder for a more complex interpolation method
        # In a real implementation, you might use methods like Kriging or IDW
        interpolated_data = np.zeros_like(grid)
        # Implement interpolation logic here
        return interpolated_data

    def calculate_fire_intensity_change_rate(self, fire_intensity_t1: np.ndarray, fire_intensity_t2: np.ndarray, time_step: float) -> np.ndarray:
        """Calculate the rate of change of fire intensity."""
        return (fire_intensity_t2 - fire_intensity_t1) / time_step

    def generate_fire_spread_probability_map(self, fire_intensity: np.ndarray, wind_speed: np.ndarray, wind_direction: np.ndarray, fuel_moisture: np.ndarray) -> np.ndarray:
        """Generate a probability map for fire spread based on current conditions."""
        # This is a simplified placeholder. In reality, this would involve a more complex fire spread model.
        spread_probability = fire_intensity * wind_speed * (1 - fuel_moisture)
        # Adjust for wind direction
        # Implement wind direction adjustment logic here
        return spread_probability

    def calculate_resource_allocation_efficiency(self, fire_intensity: np.ndarray, resources_deployed: np.ndarray) -> float:
        """Calculate the efficiency of resource allocation based on fire intensity and deployed resources."""
        total_intensity = np.sum(fire_intensity)
        total_resources = np.sum(resources_deployed)
        if total_intensity == 0 or total_resources == 0:
            return 0
        return np.sum(fire_intensity * resources_deployed) / (total_intensity * total_resources)

    def identify_natural_barriers(self, elevation: np.ndarray, water_bodies: np.ndarray, threshold: float) -> np.ndarray:
        """Identify natural barriers that could slow or stop fire spread."""
        slope = self.calculate_slope_aspect(elevation)[0]
        barriers = (slope > threshold) | water_bodies
        return barriers

    def calculate_evacuation_routes(self, road_network: gpd.GeoDataFrame, fire_intensity: np.ndarray, population_centers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate optimal evacuation routes based on road network, fire intensity, and population centers."""
        # This is a placeholder for a more complex routing algorithm
        # In a real implementation, you might use networkx or a specialized routing library
        evacuation_routes = road_network.copy()
        # Implement route calculation logic here
        return evacuation_routes

    def raster_to_vector(self, raster_data: np.ndarray, metadata: Dict[str, Any]) -> gpd.GeoDataFrame:
        """Convert raster data to vector format."""
        shapes = features.shapes(raster_data, transform=metadata['transform'])
        geometries = [shape(s) for s, v in shapes if v == 1]
        return gpd.GeoDataFrame({'geometry': geometries}, crs=self.crs)

    def vector_to_raster(self, vector_data: gpd.GeoDataFrame, like_raster: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Convert vector data to raster format."""
        rasterized = features.rasterize(vector_data.geometry, out_shape=like_raster.shape, transform=metadata['transform'])
        return rasterized

    def calculate_wind_effect(self, wind_speed: np.ndarray, wind_direction: np.ndarray, terrain: np.ndarray) -> np.ndarray:
        """Calculate the effect of wind on fire spread considering terrain."""
        slope, aspect = self.calculate_slope_aspect(terrain)
        wind_effect = wind_speed * np.cos(wind_direction - aspect) * np.exp(slope)
        return wind_effect

    def identify_spot_fire_probability(self, fire_intensity: np.ndarray, wind_speed: np.ndarray, wind_direction: np.ndarray) -> np.ndarray:
        """Identify areas with high probability of spot fires."""
        ember_production = fire_intensity * wind_speed
        # This is a simplified model. A real implementation would consider more factors and use a more complex probability model.
        spot_fire_prob = ndimage.gaussian_filter(ember_production, sigma=wind_speed)
        return spot_fire_prob

    def calculate_fire_age(self, fire_intensity_history: List[np.ndarray]) -> np.ndarray:
        """Calculate the age of the fire at each pixel."""
        fire_age = np.zeros_like(fire_intensity_history[0])
        for intensity in fire_intensity_history:
            fire_age[intensity > 0] += 1
        return fire_age

    def calculate_burn_severity(self, pre_fire_ndvi: np.ndarray, post_fire_ndvi: np.ndarray) -> np.ndarray:
        """Calculate burn severity using pre and post-fire NDVI."""
        return (pre_fire_ndvi - post_fire_ndvi) / (pre_fire_ndvi + post_fire_ndvi + 1e-6)  # Avoid division by zero

    def generate_fire_progression_map(self, fire_intensity_history: List[np.ndarray]) -> np.ndarray:
        """Generate a map showing the progression of the fire over time."""
        progression_map = np.zeros_like(fire_intensity_history[0], dtype=int)
        for i, intensity in enumerate(fire_intensity_history, 1):
            progression_map[intensity > 0] = i
        return progression_map

    def calculate_fire_return_interval(self, historical_fires: List[np.ndarray]) -> np.ndarray:
        """Calculate the fire return interval for each pixel based on historical fire data."""
        fire_count = np.sum([fire > 0 for fire in historical_fires], axis=0)
        return len(historical_fires) / (fire_count + 1e-6)  # Avoid division by zero

    def identify_fire_breaks(self, land_cover: np.ndarray, roads: np.ndarray, water_bodies: np.ndarray) -> np.ndarray:
        """Identify potential fire breaks in the landscape."""
        return np.logical_or.reduce((land_cover == 0, roads > 0, water_bodies > 0))  # Assuming 0 is non-burnable in land_cover

    def calculate_landscape_diversity(self, land_cover: np.ndarray) -> float:
        """Calculate landscape diversity using Shannon's diversity index."""
        unique, counts = np.unique(land_cover, return_counts=True)
        proportions = counts / np.sum(counts)
        return -np.sum(proportions * np.log(proportions))

    def generate_viewshed(self, elevation: np.ndarray, observer_points: List[Tuple[int, int]]) -> np.ndarray:
        """Generate a viewshed from given observer points."""
        viewshed = np.zeros_like(elevation, dtype=bool)
        for point in observer_points:
            view
    
    def calculate_distance_to_fire(self, fire_intensity: np.ndarray) -> np.ndarray:
        """Calculate the distance to the nearest fire for each cell."""
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(fire_intensity == 0)

    def load_state_layers(self, state: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Load multiple layers from the state dictionary."""
        layers = {}
        for key, path in state.items():
            if key.endswith('_path'):
                layer_name = key[:-5]  # Remove '_path' suffix
                layers[layer_name] = self.load_tiff(path)[0]
        return layers
    
    def update_raster_with_fireline(self, filepath: str, fireline_coords: List[Tuple[int, int]]) -> None:
        """
        Update a raster file by setting values to zero along the fireline coordinates.

        Args:
            filepath (str): Path to the raster file.
            fireline_coords (List[Tuple[int, int]]): List of (x, y) coordinates representing the fireline.
        """
        data, metadata = self.load_tiff(filepath)
        for x, y in fireline_coords:
            data[y, x] = 0  # Set the value to zero along the fireline
        self.save_tiff(filepath, data, metadata)
        self.logger.info(f"Updated raster file with fireline: {filepath}")


    def create_circular_fire(self, size: Tuple[int, int], center: Tuple[int, int], radius: int) -> np.ndarray:
        """
        Create a circular fire in the center of a grid.

        Args:
            size (Tuple[int, int]): Size of the grid (rows, cols).
            center (Tuple[int, int]): Center of the fire (row, col).
            radius (int): Radius of the fire.

        Returns:
            np.ndarray: Binary grid with fire (1) and no fire (0).
        """
        y, x = np.ogrid[:size[0], :size[1]]
        dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        fire = (dist_from_center <= radius).astype(int)
        return fire

    def visualize_matrix(self, matrix: np.ndarray, title: str, cmap: str = 'viridis'):
        """
        Visualize a matrix using matplotlib.

        Args:
            matrix (np.ndarray): The matrix to visualize.
            title (str): Title of the plot.
            cmap (str): Colormap to use for visualization.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap=cmap)
        plt.colorbar(label='Value')
        plt.title(f"{title}\nShape: {matrix.shape}")
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.show()

    def visualize_fire(self, fire: np.ndarray, title: str):
        """
        Visualize a fire matrix using a custom colormap.

        Args:
            fire (np.ndarray): The fire matrix to visualize.
            title (str): Title of the plot.
        """
        cmap = ListedColormap(['lightgreen', 'red'])
        plt.figure(figsize=(10, 8))
        plt.imshow(fire, cmap=cmap)
        plt.title(f"{title}\nShape: {fire.shape}")
        plt.xlabel('Column')
        plt.ylabel('Row')
        legend_elements = [plt.Rectangle((0, 0), 1, 1, fc='lightgreen', label='No Fire'),
                           plt.Rectangle((0, 0), 1, 1, fc='red', label='Fire')]
        plt.legend(handles=legend_elements, loc='upper right')
        plt.show()

    def expand_fire(self, fire: np.ndarray, expansion_factor: float = 0.1) -> np.ndarray:
        """
        Expand the fire by a certain factor.

        Args:
            fire (np.ndarray): The current fire matrix.
            expansion_factor (float): Factor by which to expand the fire.

        Returns:
            np.ndarray: Expanded fire matrix.
        """
        from scipy import ndimage
        expanded_fire = ndimage.binary_dilation(fire, iterations=int(fire.shape[0] * expansion_factor))
        return expanded_fire.astype(int)

    def apply_wind_effect(self, fire: np.ndarray, wind_direction: float, wind_speed: float) -> np.ndarray:
        """
        Apply wind effect to the fire spread.

        Args:
            fire (np.ndarray): The current fire matrix.
            wind_direction (float): Wind direction in degrees (0-360).
            wind_speed (float): Wind speed (0-1, where 1 is maximum effect).

        Returns:
            np.ndarray: Fire matrix after wind effect.
        """
        rows, cols = fire.shape
        y, x = np.ogrid[:rows, :cols]
        center = (rows // 2, cols // 2)
        
        # Convert wind direction to radians
        wind_rad = np.radians(wind_direction)
        
        # Calculate the effect based on wind direction and speed
        effect = wind_speed * ((x - center[1]) * np.cos(wind_rad) + (y - center[0]) * np.sin(wind_rad))
        
        # Normalize the effect
        effect = (effect - effect.min()) / (effect.max() - effect.min())
        
        # Apply the effect to the fire
        new_fire = fire.copy()
        new_fire[effect > 0.5] = 1
        
        return new_fire

    def generate_action_maskv1(self, fire_intensity: np.ndarray, existing_firelines: np.ndarray, min_distance: int = 1, max_distance: int = 10) -> np.ndarray:
        """
        Generate an action mask based on the current fire perimeter.

        Args:
            fire_intensity (np.ndarray): Current fire intensity matrix.
            existing_firelines (np.ndarray): Existing firelines matrix.
            min_distance (int): Minimum distance from fire to construct firelines.
            max_distance (int): Maximum distance from fire to construct firelines.

        Returns:
            np.ndarray: Boolean mask where True indicates a valid action.
        """
        # Create binary fire map
        fire_binary = fire_intensity > 0

        # Create inner boundary (minimum distance)
        inner_boundary = ndimage.binary_dilation(fire_binary, iterations=min_distance)

        # Create outer boundary (maximum distance)
        outer_boundary = ndimage.binary_dilation(fire_binary, iterations=max_distance)

        # Valid area is between inner and outer boundaries, excluding existing firelines
        valid_area = outer_boundary & ~inner_boundary & (existing_firelines == 0)

        return valid_area
    

    def generate_action_mask_from_files(self, fire_intensity_path: str, existing_firelines_path: str, 
                                        elevation_path: str = None, vegetation_path: str = None,
                                        min_distance: int = 1, max_distance: int = 10,
                                        max_slope: float = None, constructable_veg_types: List[int] = None,
                                        min_effective_length: int = None) -> np.ndarray:
        """
        Generate an action mask based on the provided geospatial data file paths.

        Args:
            fire_intensity_path (str): Path to the fire intensity GeoTIFF.
            existing_firelines_path (str): Path to the existing firelines GeoTIFF.
            elevation_path (str, optional): Path to the elevation GeoTIFF.
            vegetation_path (str, optional): Path to the vegetation GeoTIFF.
            min_distance (int): Minimum distance from fire to construct firelines.
            max_distance (int): Maximum distance from fire to construct firelines.
            max_slope (float, optional): Maximum slope for fireline construction.
            constructable_veg_types (List[int], optional): List of vegetation types suitable for fireline construction.
            min_effective_length (int, optional): Minimum effective length for a fireline.

        Returns:
            np.ndarray: Boolean mask where True indicates a valid action.
        """
        fire_intensity = self.load_tiff(fire_intensity_path)[0] if fire_intensity_path else None
        existing_firelines = self.load_tiff(existing_firelines_path)[0] if existing_firelines_path else None
        elevation = self.load_tiff(elevation_path)[0] if elevation_path else None
        vegetation = self.load_tiff(vegetation_path)[0] if vegetation_path else None

        return self.generate_action_mask(
            fire_intensity=fire_intensity,
            existing_firelines=existing_firelines,
            elevation=elevation,
            vegetation=vegetation,
            min_distance=min_distance,
            max_distance=max_distance,
            max_slope=max_slope,
            constructable_veg_types=constructable_veg_types,
            min_effective_length=min_effective_length
        )

    def generate_action_mask(self, fire_intensity: np.ndarray = None, existing_firelines: np.ndarray = None, 
                             elevation: np.ndarray = None, vegetation: np.ndarray = None,
                             min_distance: int = 1, max_distance: int = 10,
                             max_slope: float = None, constructable_veg_types: List[int] = None,
                             min_effective_length: int = None) -> np.ndarray:
        """
        Generate an action mask based on the provided geospatial data arrays.

        Args:
            fire_intensity (np.ndarray, optional): Fire intensity array.
            existing_firelines (np.ndarray, optional): Existing firelines array.
            elevation (np.ndarray, optional): Elevation array.
            vegetation (np.ndarray, optional): Vegetation array.
            min_distance (int): Minimum distance from fire to construct firelines.
            max_distance (int): Maximum distance from fire to construct firelines.
            max_slope (float, optional): Maximum slope for fireline construction.
            constructable_veg_types (List[int], optional): List of vegetation types suitable for fireline construction.
            min_effective_length (int, optional): Minimum effective length for a fireline.

        Returns:
            np.ndarray: Boolean mask where True indicates a valid action.
        """
        if fire_intensity is None:
            raise ValueError("Fire intensity data is required to generate action mask.")

        # Create binary fire map
        fire_binary = fire_intensity > 0

        # Create inner and outer boundaries
        inner_boundary = ndimage.binary_dilation(fire_binary, iterations=min_distance)
        outer_boundary = ndimage.binary_dilation(fire_binary, iterations=max_distance)

        # Initial valid area
        valid_area = outer_boundary & ~inner_boundary

        if existing_firelines is not None:
            valid_area &= (existing_firelines == 0)

        # Apply additional constraints if data is provided
        if elevation is not None and max_slope is not None:
            slope, _ = self.calculate_slope_aspect(elevation)
            valid_area &= (slope <= max_slope)

        if vegetation is not None and constructable_veg_types is not None:
            valid_area &= np.isin(vegetation, constructable_veg_types)

        # Note: min_effective_length is not used here as it's a property of the fireline, not the individual cells

        return valid_area
    

    def visualize_action_mask(self, fire_intensity: np.ndarray, action_mask: np.ndarray, title: str):
        """
        Visualize the fire intensity and action mask.

        Args:
            fire_intensity (np.ndarray): Current fire intensity matrix.
            action_mask (np.ndarray): Generated action mask.
            title (str): Title for the plot.
        """
        plt.figure(figsize=(12, 10))
        plt.imshow(fire_intensity, cmap='YlOrRd', alpha=0.7)
        plt.imshow(action_mask, cmap='Blues', alpha=0.3)
        plt.title(title)
        plt.colorbar(label='Fire Intensity')
        plt.show()

    def calculate_fire_gradient(self, fire_intensity: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the fire intensity.

        Args:
            fire_intensity (np.ndarray): Current fire intensity matrix.

        Returns:
            np.ndarray: Gradient of fire intensity.
        """
        return np.array(np.gradient(fire_intensity))
    
    def sample_firelines(self, action_mask: np.ndarray, fire_intensity: np.ndarray, num_samples: int = 10, min_length: int = 5, max_length: int = 10) -> List[Tuple[int, int, float, int]]:
        """
        Sample potential firelines based on the action mask, favoring tangential directions to the fire.

        Args:
            action_mask (np.ndarray): Boolean mask of valid actions.
            fire_intensity (np.ndarray): Current fire intensity matrix.
            num_samples (int): Number of firelines to sample.
            min_length (int): Minimum length of firelines.
            max_length (int): Maximum length of firelines.

        Returns:
            List[Tuple[int, int, float, int]]: List of sampled firelines (x, y, angle, length).
        """
        valid_positions = np.argwhere(action_mask)
        firelines = []
        fire_gradient = self.calculate_fire_gradient(fire_intensity)
        
        for _ in range(num_samples):
            if len(valid_positions) == 0:
                break
            
            valid_fireline = False
            attempts = 0
            max_attempts = 50  # Limit the number of attempts to find a valid fireline

            while not valid_fireline and attempts < max_attempts:
                idx = np.random.randint(len(valid_positions))
                y, x = valid_positions[idx]
                
                # Calculate tangential direction
                grad_y, grad_x = fire_gradient[:, y, x]
                tangent_angle = np.arctan2(-grad_x, grad_y)  # Perpendicular to gradient
                
                # Add some randomness to the angle
                angle = tangent_angle + np.random.normal(0, np.pi/6)  # Standard deviation of 30 degrees
                length = np.random.randint(min_length, max_length + 1)
                
                if self._is_valid_fireline((x, y, angle, length), fire_intensity, action_mask):
                    firelines.append((x, y, angle, length))
                    valid_fireline = True
                
                attempts += 1

            # Remove the selected position and its neighbors from valid positions
            valid_positions = valid_positions[~np.all(valid_positions == [y, x], axis=1)]
        
        return firelines
    
    
    def _is_valid_fireline(self, fireline: Tuple[int, int, float, int], fire_intensity: np.ndarray, action_mask: np.ndarray) -> bool:
        """
        Check if a fireline is valid (doesn't intersect with fire and stays within the action mask).

        Args:
            fireline (Tuple[int, int, float, int]): Fireline to check (x, y, angle, length).
            fire_intensity (np.ndarray): Current fire intensity matrix.
            action_mask (np.ndarray): Boolean mask of valid actions.

        Returns:
            bool: True if the fireline is valid, False otherwise.
        """
        x, y, angle, length = fireline
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        
        # Generate points along the fireline
        num_points = int(length * 2)  # Ensure we have enough points to check
        xs = np.linspace(x, x + dx, num_points).astype(int)
        ys = np.linspace(y, y + dy, num_points).astype(int)
        
        # Check if all points are within the grid
        if np.any(xs < 0) or np.any(xs >= fire_intensity.shape[1]) or np.any(ys < 0) or np.any(ys >= fire_intensity.shape[0]):
            return False
        
        # Check if all points are within the action mask and not intersecting with fire
        return np.all(action_mask[ys, xs]) and np.all(fire_intensity[ys, xs] == 0)

    def visualize_firelines(self, fire_intensity: np.ndarray, action_mask: np.ndarray, sampled_firelines: List[Tuple[int, int, float, int]], title: str):
        """
        Visualize fire intensity, action mask, and sampled potential firelines.

        Args:
            fire_intensity (np.ndarray): Current fire intensity matrix.
            action_mask (np.ndarray): Boolean mask of valid actions.
            sampled_firelines (List[Tuple[int, int, float, int]]): List of sampled firelines.
            title (str): Title for the plot.
        """
        plt.figure(figsize=(12, 10))
        plt.imshow(fire_intensity, cmap='YlOrRd', alpha=0.7)
        plt.imshow(action_mask, cmap='Blues', alpha=0.3)
        
        for x, y, angle, length in sampled_firelines:
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            plt.plot([x, x + dx], [y, y + dy], 'g-', linewidth=2)
        
        plt.title(title)
        plt.colorbar(label='Fire Intensity')
        plt.show()

 

def main():
    config = OverseerConfig()
    gsm = GeoSpatialManager(config)
    
    print("Creating fake data...")
    data = create_fake_data()
    
    fire_intensity = data['fire']
    action_mask = gsm.generate_action_mask(fire_intensity, np.zeros_like(fire_intensity), min_distance=1, max_distance=10)
    fire_gradient = gsm.calculate_fire_gradient(fire_intensity)
    sampled_firelines = gsm.sample_firelines(action_mask, fire_intensity, min_length=5, max_length=10)

    # Prepare matrices for visualization
    matrices_to_visualize = {
        "Initial Fire": fire_intensity,
        "Action Mask": action_mask,
        "Fire Gradient": fire_gradient,
        "Sampled Firelines": sampled_firelines
    }

    # Visualize all matrices
    visualize_multiple(matrices_to_visualize)

if __name__ == "__main__":
    main()