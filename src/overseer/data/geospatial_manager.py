# src/overseer/geospatial/geospatial_manager.py

import rasterio
import numpy as np
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape, Polygon
from scipy import ndimage
from typing import Dict, Any, Tuple, List, Optional
from overseer.config.config import OverseerConfig
from overseer.utils.logging import OverseerLogger

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

    def calculate_fire_perimeter(self, fire_intensity: np.ndarray, threshold: float) -> np.ndarray:
        """Calculate the fire perimeter based on fire intensity."""
        binary_fire = (fire_intensity > threshold).astype(np.uint8)
        perimeter = ndimage.binary_erosion(binary_fire) ^ binary_fire
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