import sqlite3
import csv
import pandas as pd
import numpy as np
import json
import pyvista as pv
import time
import random
import math
from tqdm import tqdm
from scipy.spatial.transform import Rotation


class MeshContainer:
    def __init__(self, vertices, triangles):
        self.vertices = np.array(vertices).reshape(-1, 3)
        self.triangles = np.array(triangles).reshape(-1, 3)

    @classmethod
    def from_db(cls, vertices, triangles):
        return cls(vertices, triangles)


def meshcontainer_to_pv(mesh):
    """
    Convert a MeshContainer instance to a PyVista PolyData mesh.
    mesh: MeshContainer with .vertices (N, 3) and .triangles (M, 3)
    """
    n_faces = len(mesh.triangles)
    face_array = np.hstack([
        np.full((n_faces, 1), 3),
        mesh.triangles
    ]).astype(np.int64)

    return pv.PolyData(mesh.vertices, face_array)


def get_point_coordinates(vertices, triangles, point_translations, point_triangle_ids):
    coordinates = []

    for i in range(len(point_triangle_ids)):
        idx0, idx1, idx2 = triangles[point_triangle_ids[i]], triangles[point_triangle_ids[i]+1], triangles[point_triangle_ids[i]+2]

        v0 = [vertices[idx0 * 3], vertices[idx0 * 3 + 1], vertices[idx0 * 3 + 2]]
        v1 = [vertices[idx1 * 3], vertices[idx1 * 3 + 1], vertices[idx1 * 3 + 2]]
        v2 = [vertices[idx2 * 3], vertices[idx2 * 3 + 1], vertices[idx2 * 3 + 2]]

        r0 = point_translations[i // 3][0]
        r1 = point_translations[i // 3][1]

        b0 = 1 - math.sqrt(r0)
        b1 = math.sqrt(r0) * (1 - r1)
        b2 = r1 * math.sqrt(r0)

        coordinates.append([b0*v0[0]+b1*v1[0]+b2*v2[0], b0*v0[1]+b1*v1[1]+b2*v2[1], b0*v0[2]+b1*v1[2]+b2*v2[2]])
    return coordinates
    

def extract_data(db_path, total_points, lines):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM strike LIMIT {int(lines)};", conn)

    conn.close()

    all_point_coordinates = []
    all_steps = []
    all_positions = []
    all_rotations = []

    for series_id in tqdm(df['series_id'].unique(), desc="Processing series"):
        group_df = df[df['series_id'] == series_id].reset_index(drop=True)

        # Use first frame to generate sampling pattern
        base_mesh = json.loads(group_df.loc[0, "result"])
        vertices = base_mesh["Vertices"]
        triangles = base_mesh["Triangles"]
        tmp_mesh = MeshContainer.from_db(vertices, triangles)
        pv_mesh = meshcontainer_to_pv(tmp_mesh)
        pv_mesh = pv_mesh.compute_cell_sizes()
        triangle_areas = pv_mesh.cell_data["Area"]

        point_translations = []
        point_triangle_ids = []

        for i in range(0, len(triangles), 3):
            tri_area = triangle_areas[i // 3]
            num_points = math.floor(tri_area / pv_mesh.area * total_points)
            for _ in range(num_points):
                point_translations.append([random.random(), random.random()])
                point_triangle_ids.append(i)

        while len(point_translations) < total_points:
            i = random.randint(0, len(triangle_areas) - 1)
            point_translations.append([random.random(), random.random()])
            point_triangle_ids.append(i * 3)

        # Loop over i and i+1 pairs, skipping last row
        for i in tqdm(range(len(group_df) - 1), desc=f"Series {series_id}", leave=False):
            row_i = group_df.loc[i]
            row_ip1 = group_df.loc[i + 1]

            # Input mesh (coords from frame i)
            mesh_data = json.loads(row_i["result"])
            vertices = mesh_data["Vertices"]
            triangles = mesh_data["Triangles"]
            coords = get_point_coordinates(vertices, triangles, point_translations, point_triangle_ids)

            # Get data from frame i+1
            mesh_data_next = json.loads(row_ip1["result"])
            s = np.sum(np.array(mesh_data_next["Steps"]))
            p = json.loads(row_ip1["position"])
            r = json.loads(row_ip1["rotation"])


            all_point_coordinates.append(coords)
            all_steps.append(s)
            all_positions.append(p)
            all_rotations.append(r)

    return np.array(all_point_coordinates), np.array(all_steps).reshape(-1, 1), np.array(all_positions), np.array(all_rotations)
