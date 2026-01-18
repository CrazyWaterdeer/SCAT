"""
Spatial analysis module for SCAT.
Analyzes deposit distribution patterns on films.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .detector import Deposit

# scipy is imported lazily when needed


@dataclass
class SpatialResult:
    """Container for spatial analysis results."""
    # Nearest Neighbor Distance
    nnd_mean: float
    nnd_std: float
    nnd_min: float
    nnd_max: float
    nnd_values: np.ndarray
    
    # Clustering
    clark_evans_r: float  # R < 1: clustered, R = 1: random, R > 1: dispersed
    clustering_interpretation: str
    
    # Quadrant analysis
    quadrant_counts: Dict[str, int]  # Q1, Q2, Q3, Q4
    quadrant_chi2: float
    quadrant_p_value: float
    quadrant_uniform: bool  # True if uniformly distributed
    
    # Edge vs Center
    edge_count: int
    center_count: int
    edge_fraction: float
    
    # Density map
    density_map: np.ndarray
    
    # Image dimensions
    image_width: int
    image_height: int


class SpatialAnalyzer:
    """Analyze spatial distribution of deposits."""
    
    def __init__(self, edge_margin_fraction: float = 0.15):
        """
        Args:
            edge_margin_fraction: Fraction of image considered "edge" (default 15%)
        """
        self.edge_margin_fraction = edge_margin_fraction
        self._scipy_loaded = False
    
    def _load_scipy(self):
        """Lazy load scipy modules."""
        if self._scipy_loaded:
            return
        from scipy import ndimage
        from scipy.spatial.distance import cdist, pdist
        from scipy.stats import pearsonr
        self.ndimage = ndimage
        self.cdist = cdist
        self.pdist = pdist
        self.pearsonr = pearsonr
        self._scipy_loaded = True
    
    def analyze(
        self, 
        deposits: List[Deposit], 
        image_shape: Tuple[int, int],
        exclude_artifacts: bool = True
    ) -> SpatialResult:
        """
        Perform complete spatial analysis.
        
        Args:
            deposits: List of detected deposits
            image_shape: (height, width) of image
            exclude_artifacts: Whether to exclude artifacts from analysis
        """
        self._load_scipy()  # Lazy load scipy
        
        height, width = image_shape[:2]
        
        # Filter deposits
        if exclude_artifacts:
            deposits = [d for d in deposits if d.label != 'artifact']
        
        if len(deposits) < 2:
            return self._empty_result(width, height)
        
        # Get centroids
        centroids = np.array([d.centroid for d in deposits])
        
        # Nearest Neighbor Distance
        nnd_result = self._calculate_nnd(centroids)
        
        # Clark-Evans clustering index
        ce_result = self._clark_evans_index(centroids, width, height)
        
        # Quadrant analysis
        quad_result = self._quadrant_analysis(centroids, width, height)
        
        # Edge vs Center
        edge_result = self._edge_center_analysis(centroids, width, height)
        
        # Density map
        density_map = self._generate_density_map(centroids, width, height)
        
        return SpatialResult(
            nnd_mean=nnd_result['mean'],
            nnd_std=nnd_result['std'],
            nnd_min=nnd_result['min'],
            nnd_max=nnd_result['max'],
            nnd_values=nnd_result['values'],
            clark_evans_r=ce_result['r'],
            clustering_interpretation=ce_result['interpretation'],
            quadrant_counts=quad_result['counts'],
            quadrant_chi2=quad_result['chi2'],
            quadrant_p_value=quad_result['p_value'],
            quadrant_uniform=quad_result['uniform'],
            edge_count=edge_result['edge'],
            center_count=edge_result['center'],
            edge_fraction=edge_result['edge_fraction'],
            density_map=density_map,
            image_width=width,
            image_height=height
        )
    
    def _calculate_nnd(self, centroids: np.ndarray) -> Dict:
        """Calculate Nearest Neighbor Distances."""
        if len(centroids) < 2:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'values': np.array([])}
        
        # Calculate pairwise distances
        dist_matrix = self.cdist(centroids, centroids)
        
        # Set diagonal to infinity to exclude self-distance
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Get minimum distance for each point
        nnd_values = np.min(dist_matrix, axis=1)
        
        return {
            'mean': float(np.mean(nnd_values)),
            'std': float(np.std(nnd_values)),
            'min': float(np.min(nnd_values)),
            'max': float(np.max(nnd_values)),
            'values': nnd_values
        }
    
    def _clark_evans_index(
        self, 
        centroids: np.ndarray, 
        width: int, 
        height: int
    ) -> Dict:
        """
        Calculate Clark-Evans R index for clustering.
        
        R < 1: Clustered
        R = 1: Random (Poisson)
        R > 1: Dispersed (regular)
        """
        n = len(centroids)
        if n < 2:
            return {'r': 1.0, 'interpretation': 'insufficient_data'}
        
        area = width * height
        
        # Observed mean NND
        nnd = self._calculate_nnd(centroids)
        observed_mean = nnd['mean']
        
        # Expected mean NND for random distribution
        density = n / area
        expected_mean = 0.5 / np.sqrt(density)
        
        # Clark-Evans R
        r = observed_mean / expected_mean if expected_mean > 0 else 1.0
        
        # Interpretation
        if r < 0.5:
            interpretation = 'highly_clustered'
        elif r < 0.8:
            interpretation = 'clustered'
        elif r < 1.2:
            interpretation = 'random'
        elif r < 1.5:
            interpretation = 'dispersed'
        else:
            interpretation = 'highly_dispersed'
        
        return {'r': float(r), 'interpretation': interpretation}
    
    def _quadrant_analysis(
        self, 
        centroids: np.ndarray, 
        width: int, 
        height: int
    ) -> Dict:
        """
        Divide image into quadrants and test for uniform distribution.
        """
        from scipy.stats import chi2_contingency, chisquare
        
        mid_x, mid_y = width / 2, height / 2
        
        counts = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
        
        for x, y in centroids:
            if x < mid_x and y < mid_y:
                counts['Q1'] += 1  # Top-left
            elif x >= mid_x and y < mid_y:
                counts['Q2'] += 1  # Top-right
            elif x < mid_x and y >= mid_y:
                counts['Q3'] += 1  # Bottom-left
            else:
                counts['Q4'] += 1  # Bottom-right
        
        # Chi-square test for uniform distribution
        observed = list(counts.values())
        n = sum(observed)
        expected = [n / 4] * 4
        
        if n > 0 and all(e >= 5 for e in expected):
            chi2, p_value = chisquare(observed, expected)
        else:
            chi2, p_value = 0, 1.0
        
        return {
            'counts': counts,
            'chi2': float(chi2),
            'p_value': float(p_value),
            'uniform': p_value > 0.05
        }
    
    def _edge_center_analysis(
        self, 
        centroids: np.ndarray, 
        width: int, 
        height: int
    ) -> Dict:
        """
        Count deposits near edges vs center of image.
        """
        margin_x = width * self.edge_margin_fraction
        margin_y = height * self.edge_margin_fraction
        
        edge_count = 0
        center_count = 0
        
        for x, y in centroids:
            if (x < margin_x or x > width - margin_x or 
                y < margin_y or y > height - margin_y):
                edge_count += 1
            else:
                center_count += 1
        
        total = edge_count + center_count
        edge_fraction = edge_count / total if total > 0 else 0
        
        return {
            'edge': edge_count,
            'center': center_count,
            'edge_fraction': float(edge_fraction)
        }
    
    def _generate_density_map(
        self, 
        centroids: np.ndarray, 
        width: int, 
        height: int,
        grid_size: int = 50
    ) -> np.ndarray:
        """
        Generate density heatmap of deposit locations.
        
        Args:
            centroids: Array of (x, y) coordinates
            width, height: Image dimensions
            grid_size: Number of grid cells per dimension
        """
        # Create empty grid
        density = np.zeros((grid_size, grid_size))
        
        if len(centroids) == 0:
            return density
        
        # Map centroids to grid
        cell_width = width / grid_size
        cell_height = height / grid_size
        
        for x, y in centroids:
            grid_x = min(int(x / cell_width), grid_size - 1)
            grid_y = min(int(y / cell_height), grid_size - 1)
            density[grid_y, grid_x] += 1
        
        # Smooth with Gaussian filter
        density = self.ndimage.gaussian_filter(density, sigma=1.5)
        
        return density
    
    def _empty_result(self, width: int, height: int) -> SpatialResult:
        """Return empty result when insufficient data."""
        return SpatialResult(
            nnd_mean=0, nnd_std=0, nnd_min=0, nnd_max=0,
            nnd_values=np.array([]),
            clark_evans_r=1.0,
            clustering_interpretation='insufficient_data',
            quadrant_counts={'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0},
            quadrant_chi2=0, quadrant_p_value=1.0, quadrant_uniform=True,
            edge_count=0, center_count=0, edge_fraction=0,
            density_map=np.zeros((50, 50)),
            image_width=width, image_height=height
        )
    
    def get_summary_dict(self, result: SpatialResult) -> Dict:
        """Convert SpatialResult to dictionary for reporting."""
        return {
            'nnd_mean': result.nnd_mean,
            'nnd_std': result.nnd_std,
            'nnd_min': result.nnd_min,
            'nnd_max': result.nnd_max,
            'clark_evans_r': result.clark_evans_r,
            'clustering': result.clustering_interpretation,
            'quadrant_Q1': result.quadrant_counts['Q1'],
            'quadrant_Q2': result.quadrant_counts['Q2'],
            'quadrant_Q3': result.quadrant_counts['Q3'],
            'quadrant_Q4': result.quadrant_counts['Q4'],
            'quadrant_chi2': result.quadrant_chi2,
            'quadrant_p_value': result.quadrant_p_value,
            'quadrant_uniform': result.quadrant_uniform,
            'edge_count': result.edge_count,
            'center_count': result.center_count,
            'edge_fraction': result.edge_fraction
        }


def analyze_spatial_batch(
    results_list: List[Tuple[List[Deposit], Tuple[int, int]]]
) -> List[SpatialResult]:
    """
    Analyze spatial distribution for multiple images.
    
    Args:
        results_list: List of (deposits, image_shape) tuples
        
    Returns:
        List of SpatialResult objects
    """
    analyzer = SpatialAnalyzer()
    return [analyzer.analyze(deposits, shape) for deposits, shape in results_list]


def aggregate_spatial_stats(spatial_results: List[SpatialResult]) -> Dict:
    """
    Aggregate spatial statistics across multiple images.
    """
    if not spatial_results:
        return {}
    
    valid_results = [r for r in spatial_results if r.nnd_mean > 0]
    
    if not valid_results:
        return {}
    
    return {
        'mean_nnd': float(np.mean([r.nnd_mean for r in valid_results])),
        'std_nnd': float(np.std([r.nnd_mean for r in valid_results])),
        'mean_clark_evans': float(np.mean([r.clark_evans_r for r in valid_results])),
        'mean_edge_fraction': float(np.mean([r.edge_fraction for r in valid_results])),
        'n_clustered': sum(1 for r in valid_results if 'cluster' in r.clustering_interpretation),
        'n_random': sum(1 for r in valid_results if r.clustering_interpretation == 'random'),
        'n_dispersed': sum(1 for r in valid_results if 'dispers' in r.clustering_interpretation),
        'n_images': len(valid_results)
    }
