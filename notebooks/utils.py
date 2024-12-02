import struct
import numpy as np

def serialize_matrix(x: np.ndarray, filepath: str):
    shape_size = len(x.shape)
    metadata = [shape_size] + list(x.shape)
    with open(filepath, "wb") as f:
        for el in metadata:
            b = struct.pack("I", el)
            f.write(b)
        for el in x.flatten():
            b = struct.pack("f", el)
            f.write(b)