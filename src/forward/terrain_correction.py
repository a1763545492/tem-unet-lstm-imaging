import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple, List, Optional


class FEMTerrainCorrection:
    def __init__(self, grid_size: int = 50):
        self.grid_size = grid_size
        self.mesh = None
        self.stiffness_matrix = None
        self.mass_matrix = None
        self.dof_map = None

    def build_fem_grid(self, x_range: Tuple[float, float] = (-100, 100),
                       y_range: Tuple[float, float] = (-100, 100),
                       z_range: Tuple[float, float] = (-500, 50)):
        x = np.linspace(x_range[0], x_range[1], self.grid_size)
        y = np.linspace(y_range[0], y_range[1], self.grid_size)
        z = np.linspace(z_range[0], z_range[1], self.grid_size // 2)
        
        xx, yy, zz = np.meshgrid(x, y, z)
        self.nodes = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
        
        self.n_nodes = len(self.nodes)
        self.n_elements = (self.grid_size - 1) ** 2 * (self.grid_size // 2 - 1)
        
        self.dof_map = np.arange(self.n_nodes)

    def compute_element_stiffness(self, element_nodes: np.ndarray, 
                                  conductivity: float) -> np.ndarray:
        J = np.array([[element_nodes[1, 0] - element_nodes[0, 0], 0, 0],
                      [0, element_nodes[3, 1] - element_nodes[0, 1], 0],
                      [0, 0, element_nodes[4, 2] - element_nodes[0, 2]]])
        
        det_J = np.linalg.det(J)
        inv_J = np.linalg.inv(J)
        
        B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        B_T = B.T
        
        k_element = conductivity * B_T @ inv_J.T @ inv_J @ B * det_J / 8
        
        return k_element

    def assemble_stiffness_matrix(self, conductivities: np.ndarray):
        self.stiffness_matrix = lil_matrix((self.n_nodes, self.n_nodes))
        
        element_idx = 0
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size - 1):
                for k in range(self.grid_size // 2 - 1):
                    node_idx = i * self.grid_size * (self.grid_size // 2) + \
                               j * (self.grid_size // 2) + k
                    
                    element_nodes = self.nodes[node_idx:node_idx+8]
                    
                    elem_conductivity = conductivities[min(element_idx, len(conductivities)-1)]
                    
                    k_element = self.compute_element_stiffness(element_nodes, elem_conductivity)
                    
                    for p in range(8):
                        for q in range(8):
                            self.stiffness_matrix[node_idx + p, node_idx + q] += k_element[p, q]
                    
                    element_idx += 1
        
        self.stiffness_matrix = csc_matrix(self.stiffness_matrix)

    def solve_potential(self, boundary_conditions: dict) -> np.ndarray:
        if self.stiffness_matrix is None:
            raise ValueError("Stiffness matrix not assembled")
        
        K = self.stiffness_matrix.copy()
        f = np.zeros(self.n_nodes)
        
        for node_idx, value in boundary_conditions.items():
            K[node_idx, :] = 0
            K[node_idx, node_idx] = 1
            f[node_idx] = value
        
        potential = spsolve(K, f)
        return potential

    def calculate_terrain_effect(self, station_locations: np.ndarray,
                                 conductivities: List[float],
                                 terrain_heights: np.ndarray) -> np.ndarray:
        flat_response = np.zeros((len(station_locations), len(conductivities)))
        terrain_response = np.zeros((len(station_locations), len(conductivities)))
        
        for i, station in enumerate(station_locations):
            for j, sigma in enumerate(conductivities):
                distance = np.sqrt(station[0]**2 + station[1]**2)
                flat_response[i, j] = 1 / (2 * np.pi * sigma * distance)
                
                height = terrain_heights[i] if i < len(terrain_heights) else 0
                terrain_response[i, j] = flat_response[i, j] * (1 + height / distance)
        
        correction_factor = terrain_response / (flat_response + 1e-10)
        return correction_factor

    def apply_correction(self, data: np.ndarray, correction_factors: np.ndarray) -> np.ndarray:
        corrected_data = data.copy()
        for i in range(len(data)):
            corrected_data[i] = data[i] * correction_factors[min(i, len(correction_factors)-1)]
        return corrected_data

    def build_3d_terrain_database(self, terrain_types: List[str] = None,
                                   x_range: Tuple[float, float] = (-100, 100),
                                   y_range: Tuple[float, float] = (-100, 100)) -> dict:
        if terrain_types is None:
            terrain_types = ['flat', 'rolling', 'mountain', 'valley', 'cliff']
        
        database = {}
        
        for terrain_type in terrain_types:
            heights = self._generate_terrain_heights(terrain_type, x_range, y_range)
            database[terrain_type] = heights
        
        return database

    def _generate_terrain_heights(self, terrain_type: str,
                                   x_range: Tuple[float, float],
                                   y_range: Tuple[float, float]) -> np.ndarray:
        x = np.linspace(x_range[0], x_range[1], self.grid_size)
        y = np.linspace(y_range[0], y_range[1], self.grid_size)
        xx, yy = np.meshgrid(x, y)
        
        if terrain_type == 'flat':
            z = np.zeros(xx.shape)
        elif terrain_type == 'rolling':
            z = 10 * np.sin(xx / 20) + 10 * np.cos(yy / 20)
        elif terrain_type == 'mountain':
            cx, cy = (x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            z = 30 * np.exp(-dist**2 / 500)
        elif terrain_type == 'valley':
            cx, cy = (x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            z = -20 * np.exp(-dist**2 / 800)
        elif terrain_type == 'cliff':
            z = np.where(xx < 0, 20, -20)
        else:
            z = np.random.normal(0, 5, xx.shape)
        
        return z.flatten()

    def save_correction_model(self, file_path: str):
        model_data = {
            'grid_size': self.grid_size,
            'nodes': self.nodes,
            'stiffness_matrix': self.stiffness_matrix,
            'dof_map': self.dof_map
        }
        np.save(file_path, model_data)

    def load_correction_model(self, file_path: str):
        model_data = np.load(file_path, allow_pickle=True).item()
        self.grid_size = model_data['grid_size']
        self.nodes = model_data['nodes']
        self.stiffness_matrix = model_data['stiffness_matrix']
        self.dof_map = model_data['dof_map']
        self.n_nodes = len(self.nodes)