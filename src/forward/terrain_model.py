import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple, Optional


class TriangularMesh:
    def __init__(self, points: np.ndarray):
        self.points = points
        self.tri = Delaunay(points)
        self.n_triangles = len(self.tri.simplices)

    def get_triangle_vertices(self, idx: int) -> np.ndarray:
        return self.points[self.tri.simplices[idx]]

    def calculate_area(self, idx: int) -> float:
        vertices = self.get_triangle_vertices(idx)
        a = np.linalg.norm(vertices[1] - vertices[0])
        b = np.linalg.norm(vertices[2] - vertices[1])
        c = np.linalg.norm(vertices[0] - vertices[2])
        s = (a + b + c) / 2
        return np.sqrt(s * (s - a) * (s - b) * (s - c))

    def get_neighbors(self, idx: int) -> List[int]:
        neighbors = []
        edges = self.tri.simplices[idx]
        for i in range(3):
            edge = tuple(sorted([edges[i], edges[(i+1)%3]]))
            for j, other_simplex in enumerate(self.tri.simplices):
                if j != idx:
                    other_edges = [tuple(sorted([other_simplex[k], other_simplex[(k+1)%3]])) 
                                   for k in range(3)]
                    if edge in other_edges:
                        neighbors.append(j)
                        break
        return neighbors

    def interpolate_height(self, x: float, y: float) -> float:
        try:
            simplex_idx = self.tri.find_simplex([x, y])
            if simplex_idx < 0:
                return self._extrapolate_height(x, y)
            
            simplex = self.tri.simplices[simplex_idx]
            vertices = self.points[simplex]
            
            A = np.array([[vertices[0][0], vertices[0][1], 1],
                         [vertices[1][0], vertices[1][1], 1],
                         [vertices[2][0], vertices[2][1], 1]])
            b = vertices[:, 2]
            
            coef = np.linalg.solve(A, b)
            return coef[0] * x + coef[1] * y + coef[2]
        except:
            return self._extrapolate_height(x, y)

    def _extrapolate_height(self, x: float, y: float) -> float:
        dists = np.sqrt((self.points[:, 0] - x)**2 + (self.points[:, 1] - y)**2)
        nearest_idx = np.argmin(dists)
        return self.points[nearest_idx, 2]


class TerrainModel:
    def __init__(self, grid_size: int = 50):
        self.grid_size = grid_size
        self.mesh = None

    def generate_random_terrain(self, x_range: Tuple[float, float] = (-100, 100),
                                y_range: Tuple[float, float] = (-100, 100),
                                max_height: float = 50) -> TriangularMesh:
        n_points = self.grid_size * self.grid_size // 4
        x = np.random.uniform(x_range[0], x_range[1], n_points)
        y = np.random.uniform(y_range[0], y_range[1], n_points)
        z = np.random.uniform(-max_height, max_height, n_points)
        
        points = np.column_stack([x, y, z])
        self.mesh = TriangularMesh(points)
        return self.mesh

    def generate_synthetic_terrain(self, terrain_type: str = 'rolling',
                                   x_range: Tuple[float, float] = (-100, 100),
                                   y_range: Tuple[float, float] = (-100, 100)) -> TriangularMesh:
        n_points = self.grid_size * self.grid_size
        x = np.linspace(x_range[0], x_range[1], self.grid_size)
        y = np.linspace(y_range[0], y_range[1], self.grid_size)
        xx, yy = np.meshgrid(x, y)
        
        if terrain_type == 'rolling':
            z = 10 * np.sin(xx / 20) + 10 * np.cos(yy / 20) + np.random.normal(0, 2, xx.shape)
        elif terrain_type == 'mountain':
            cx, cy = (x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            z = 30 * np.exp(-dist**2 / 500) + np.random.normal(0, 3, xx.shape)
        elif terrain_type == 'valley':
            cx, cy = (x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            z = -20 * np.exp(-dist**2 / 800) + np.random.normal(0, 2, xx.shape)
        elif terrain_type == 'cliff':
            z = np.where(xx < 0, 20 + np.random.normal(0, 3, xx.shape), 
                        -20 + np.random.normal(0, 3, xx.shape))
        else:
            z = np.random.normal(0, 10, xx.shape)
        
        points = np.column_stack([xx.flatten(), yy.flatten(), z.flatten()])
        self.mesh = TriangularMesh(points)
        return self.mesh

    def calculate_terrain_response(self, station_locations: np.ndarray,
                                   conductivities: List[float], 
                                   thicknesses: List[float],
                                   forward_model) -> np.ndarray:
        responses = []
        for station in station_locations:
            height = self.mesh.interpolate_height(station[0], station[1])
            
            adjusted_thicknesses = thicknesses.copy()
            adjusted_thicknesses[0] = max(0, adjusted_thicknesses[0] - height / 2)
            
            response = forward_model.layered_earth_response(conductivities, adjusted_thicknesses)
            responses.append(response)
        
        return np.array(responses)

    def save_terrain(self, file_path: str):
        if self.mesh is not None:
            np.save(file_path, self.mesh.points)

    def load_terrain(self, file_path: str):
        points = np.load(file_path)
        self.mesh = TriangularMesh(points)
        return self.mesh