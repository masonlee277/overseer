# src/overseer/geospatial/geospatial_manager.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import traceback
import rasterio
import numpy as np
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape, Polygon, Point, LineString 

from scipy import ndimage
from scipy.ndimage import convolve2d

from typing import Dict, Any, Tuple, List, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.transform import Affine

from overseer.config.config import OverseerConfig
from overseer.utils.logging import OverseerLogger
from overseer.utils import fix_path
from overseer.data.utils import create_fake_data, visualize_multiple
from overseer.core.models import SimulationState, SimulationMetrics

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
        self.pixel_size = 30  # 30x30m grid size

    def load_tiff(self, filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a GeoTIFF file and return the data and metadata."""
        try:
            with rasterio.open(filepath) as src:
                data = src.read(1)  # Assuming single band
                metadata = src.meta
            self.logger.info(f"Successfully loaded GeoTIFF from {filepath}")
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


    def calculate_fire_perimeter(self, input_data: Union[str, np.ndarray]) -> float:
        """
        Calculate the fire perimeter based on the input data.

        This method calculates the fire perimeter by identifying the boundary cells
        of the fire and summing their lengths. It assumes that diagonal connections
        contribute sqrt(2) * resolution to the perimeter, while orthogonal connections
        contribute 1 * resolution.

        Args:
            input_data (Union[str, np.ndarray]): Either a file path to a GeoTIFF or a numpy array
                                                 representing the fire (e.g., time of arrival or fire intensity).

        Returns:
            float: The calculated fire perimeter in kilometers.

        Note:
            - If a file path is provided, it will be opened using open_tiff method.
            - The method assumes a 30x30m grid size for each pixel.
            - Non-zero values in the input are considered as part of the fire.
        """
        self.logger.info("Calculating fire perimeter")

        if isinstance(input_data, str):
            self.logger.debug(f"Opening file: {input_data}")
            input_data = self.open_tiff(input_data)['data']

        # Create a binary fire map
        fire_map = (input_data > 0).astype(int)

        # Identify boundary cells
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        boundary = fire_map - (fire_map & (convolve2d(fire_map, kernel, mode='same') == 8))

        # Calculate perimeter
        perimeter = np.sum(boundary) * self.resolution  # Orthogonal connections
        
        # Add diagonal connections
        diag_kernel = np.array([[1, 0, 1],
                                [0, 0, 0],
                                [1, 0, 1]])
        diag_connections = convolve2d(boundary, diag_kernel, mode='same')
        perimeter += np.sum(diag_connections) * (np.sqrt(2) - 1) * self.resolution

        # Convert perimeter from meters to kilometers
        perimeter_km = perimeter / 1000

        self.logger.info(f"Calculated fire perimeter: {perimeter_km:.2f} kilometers")
        return perimeter_km

    def calculate_fire_size(self, input_data: Union[str, np.ndarray]) -> float:
        """
        Calculate the fire size (burned area) based on the input data.

        This method calculates the total area of the fire by counting the number of
        cells that are considered part of the fire and multiplying by the cell area.

        Args:
            input_data (Union[str, np.ndarray]): Either a file path to a GeoTIFF or a numpy array
                                                 representing the fire (e.g., time of arrival or fire intensity).

        Returns:
            float: The calculated fire size in square acres.

        Note:
            - If a file path is provided, it will be opened using open_tiff method.
            - The method assumes a 30x30m grid size for each pixel.
            - Non-zero values in the input are considered as part of the fire.
        """
        self.logger.info("Calculating fire size")

        if isinstance(input_data, str):
            self.logger.debug(f"Opening file: {input_data}")
            input_data = self.open_tiff(input_data)['data']

        # Count non-zero cells and multiply by cell area
        fire_size_sq_meters = np.sum(input_data > 0) * (self.resolution ** 2)

        # Convert fire size from square meters to square acres
        # 1 square meter = 0.000247105 acres
        fire_size_acres = fire_size_sq_meters * 0.000247105

        self.logger.info(f"Calculated fire size: {fire_size_acres:.2f} square acres")
        return fire_size_acres
    
    def raster_to_vector(self, raster_data: np.ndarray, metadata: Dict[str, Any]) -> gpd.GeoDataFrame:
        """Convert raster data to vector format."""
        shapes = features.shapes(raster_data, transform=metadata['transform'])
        geometries = [shape(s) for s, v in shapes if v == 1]
        return gpd.GeoDataFrame({'geometry': geometries}, crs=self.crs)

    def vector_to_raster(self, vector_data: gpd.GeoDataFrame, like_raster: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Convert vector data to raster format."""
        rasterized = features.rasterize(vector_data.geometry, out_shape=like_raster.shape, transform=metadata['transform'])
        return rasterized



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
    
    def calculate_fire_gradient(self, fire_intensity: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the fire intensity.

        Args:
            fire_intensity (np.ndarray): Current fire intensity matrix.

        Returns:
            np.ndarray: Gradient of fire intensity.
        """
        return np.array(np.gradient(fire_intensity))
    
    def calculate_fire_growth_rate(self, toa_path: str, time_interval: float) -> float:
        """
        Calculate the fire growth rate based on the time of arrival (TOA) data.

        Args:
            toa_path (str): Path to the time of arrival raster file.
            time_interval (float): Time interval in seconds to calculate the growth rate.

        Returns:
            float: Fire growth rate in square meters per second.
        """
        try:
            with rasterio.open(toa_path) as src:
                toa_data = src.read(1)
                pixel_area = abs(src.transform.a * src.transform.e)  # Calculate pixel area

            # Calculate the area burned at the start and end of the interval
            start_area = np.sum(toa_data > 0) * pixel_area
            end_area = np.sum(toa_data <= time_interval) * pixel_area

            # Calculate the growth rate
            growth_rate = (end_area - start_area) / time_interval

            return growth_rate

        except Exception as e:
            self.logger.error(f"Error calculating fire growth rate: {str(e)}")
            return 0.0

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

    def calculate_state_metrics(self, state: SimulationState) -> SimulationMetrics:
        """
        Calculate simulation metrics based on the current simulation state.

        Args:
            state (SimulationState): The current simulation state.

        Returns:
            SimulationMetrics: Updated simulation metrics.
        """
        metrics = SimulationMetrics(
            burned_area=0.0,
            fire_perimeter_length=0.0,
            containment_percentage=0.0,
            execution_time=state.metrics.execution_time,
            performance_metrics=state.metrics.performance_metrics,
            fire_intensity=np.zeros((1, 1))
        )

        try:
            time_of_arrival_path = fix_path(str(state.paths.output_paths.time_of_arrival), add_tif=True)
            fire_intensity_path = fix_path(str(state.paths.output_paths.fire_intensity), add_tif=True)
            # flame_length_path = fix_path(str(state.paths.output_paths.flame_length), add_tif=True)
            # spread_rate_path = fix_path(str(state.paths.output_paths.spread_rate), add_tif=True)

            self.logger.info(f"[calculate_state_metrics] Paths:")
            self.logger.info(f"Time of arrival: {time_of_arrival_path}")
            self.logger.info(f"Fire intensity: {fire_intensity_path}")
            # self.logger.info(f"Flame length: {flame_length_path}")
            # self.logger.info(f"Spread rate: {spread_rate_path}")

            # Load necessary data
            for path, name in [
                (time_of_arrival_path, "Time of arrival"),
                (fire_intensity_path, "Fire intensity"),
                # (flame_length_path, "Flame length"),
                # (spread_rate_path, "Spread rate")
            ]:
                try:
                    data, _ = self.load_tiff(str(path))
                    if name == "Time of arrival":
                        time_of_arrival = data
                    elif name == "Fire intensity":
                        fire_intensity = data
                        metrics.fire_intensity = fire_intensity
                except Exception as e:
                    self.logger.warning(f"Error loading {name} data: {str(e)}")
                    self.logger.debug(f"Stack trace: {traceback.format_exc()}")
                    self.logger.info(f"Continuing with other calculations...")
                    continue

            # Calculate metrics
            try:
                metrics.burned_area = np.sum(time_of_arrival > 0) * (self.resolution ** 2)  # in square meters
                self.logger.info(f"Calculated burned area: {metrics.burned_area}")
            except Exception as e:
                self.logger.warning(f"Error calculating burned area: {str(e)}")
                self.logger.debug(f"Stack trace: {traceback.format_exc()}")

            try:
                metrics.fire_perimeter_length = self.calculate_fire_perimeter(time_of_arrival)
                self.logger.info(f"Calculated fire perimeter length: {metrics.fire_perimeter_length}")
            except Exception as e:
                self.logger.warning(f"Error calculating fire perimeter length: {str(e)}")
                self.logger.debug(f"Stack trace: {traceback.format_exc()}")

            try:
                metrics.containment_percentage = self.calculate_containment_percentage(time_of_arrival)
                self.logger.info(f"Calculated containment percentage: {metrics.containment_percentage}")
            except Exception as e:
                self.logger.warning(f"Error calculating containment percentage: {str(e)}")
                self.logger.debug(f"Stack trace: {traceback.format_exc()}")

        except Exception as e:
            self.logger.error(f"Unexpected error in calculate_state_metrics: {str(e)}")
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")

        self.logger.info(f"Final metrics: {metrics}")
        return metrics



    
    def generate_action_from_files(self, fire_intensity_path: str, existing_firelines_path: str, 
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
    
    
    def open_tiff(self, filepath: str) -> Dict[str, Any]:
        """
        Open a GeoTIFF file and return its data and metadata.

        Args:
            filepath (str): Path to the GeoTIFF file.

        Returns:
            Dict[str, Any]: A dictionary containing the raster data and metadata.
        """
        try:
            with rasterio.open(filepath) as src:
                data = src.read(1)
                metadata = {
                    'driver': src.driver,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': src.dtypes[0],
                    'crs': src.crs.to_string(),
                    'transform': src.transform.to_gdal(),
                }
                self.logger.info(f"Opened GeoTIFF: {filepath}")
                self.logger.info(f"Metadata: {metadata}")
                return {'data': data, 'metadata': metadata}
        except rasterio.errors.RasterioIOError as e:
            self.logger.error(f"Error opening GeoTIFF file {filepath}: {str(e)}")
            raise


    def update_fuel_file(self, filepath: str, fireline_coords: List[Tuple[int, int]]) -> None:
        self.logger.info(f"Updating fuel file: {filepath}")
        self.logger.info(f"Fireline coordinates: {fireline_coords}")

        try:
            with rasterio.open(filepath) as src:
                data = src.read(1)
                metadata = src.meta.copy()

            # Update the data
            for x, y in fireline_coords:
                if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]:
                    data[y, x] = 0  # Set fireline cells to 0 (or another appropriate value)
                else:
                    self.logger.warning(f"Coordinate ({x}, {y}) is out of bounds and will be skipped")

            # Update the metadata to use Affine transform
            transform = metadata['transform']
            if isinstance(transform, Affine):
                # If it's already an Affine object, use it directly
                new_transform = transform
            elif len(transform) == 6:
                # If it's a tuple of 6 elements, create Affine from it
                new_transform = Affine.from_gdal(*transform)
            elif len(transform) == 9:
                # If it's a tuple of 9 elements (3x3 matrix), create Affine from the first 6
                new_transform = Affine.from_gdal(*transform[:6])
            else:
                # If it's something else, log an error and raise an exception
                self.logger.error(f"Unexpected transform format: {transform}")
                raise ValueError(f"Unexpected transform format: {transform}")

            metadata.update({
                'driver': 'GTiff',
                'height': data.shape[0],
                'width': data.shape[1],
                'transform': new_transform,
            })

            # Write the updated data
            with rasterio.open(filepath, 'w', **metadata) as dst:
                dst.write(data, 1)

            self.logger.info(f"Successfully updated fuel file: {filepath}")
        except Exception as e:
            self.logger.error(f"Error updating fuel file {filepath}: {str(e)}")
            raise



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
    
    # File paths
    toa_path = "/teamspace/studios/this_studio/overseer/data/mock/ex/toa_ex.tif"
    flin_path = "/teamspace/studios/this_studio/overseer/data/mock/ex/flin_ex.tif"

    # Calculate fire perimeter and size using time of arrival data
    toa_perimeter = gsm.calculate_fire_perimeter(toa_path)
    toa_size = gsm.calculate_fire_size(toa_path)

    print(f"Fire perimeter (based on time of arrival): {toa_perimeter:.2f} kilometers")
    print(f"Fire size (based on time of arrival): {toa_size:.2f} square acres")

    # Calculate fire perimeter and size using fire line intensity data
    flin_perimeter = gsm.calculate_fire_perimeter(flin_path)
    flin_size = gsm.calculate_fire_size(flin_path)

    print(f"Fire perimeter (based on fire line intensity): {flin_perimeter:.2f} kilometers")
    print(f"Fire size (based on fire line intensity): {flin_size:.2f} square acres")



    # print("Creating fake data...")
    # data = create_fake_data()
    
    # fire_intensity = data['fire']
    # action_mask = gsm.generate_action_mask(fire_intensity, np.zeros_like(fire_intensity), min_distance=1, max_distance=10)
    # fire_gradient = gsm.calculate_fire_gradient(fire_intensity)
    # sampled_firelines = gsm.sample_firelines(action_mask, fire_intensity, min_length=5, max_length=10)

    # # Prepare matrices for visualization
    # matrices_to_visualize = {
    #     "Initial Fire": fire_intensity,
    #     "Action Mask": action_mask,
    #     "Fire Gradient": fire_gradient,
    #     "Sampled Firelines": sampled_firelines
    # }

    # # Visualize all matrices
    # visualize_multiple(matrices_to_visualize)

if __name__ == "__main__":
    main()