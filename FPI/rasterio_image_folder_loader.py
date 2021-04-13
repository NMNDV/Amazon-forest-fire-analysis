import os
import rasterio
import numpy as np



def LoadImageFolder(addr: str, extend_width: int = 0, scale: float = 1.0) -> list:
    if not extend_width:
        return [
            rasterio.open(os.path.join(addr, image)).read(1) * scale
            for image in os.listdir(addr)
        ]
    else:
        return [
            extend_col(rasterio.open(os.path.join(addr, image)).read(1), extend_width) * scale
            for image in os.listdir(addr)
        ]

def extend_col(array: np.ndarray, new_width: int) -> np.ndarray:
    extended_clone = []
    left = new_width - array.shape[1]
    insert_ratio = array.shape[1] // left
    if left <= 0:
        raise("invalid i/p new_width must be greater than the actual length of array")
    for i in range(array.shape[0]):
        temp = list(array[i])
        for j in range(left):
            temp.insert(insert_ratio*j + j, temp[insert_ratio*j + j])
        extended_clone.append(temp)
    return np.asarray(extended_clone, dtype=array.dtype)


if __name__ == "__main__":
    """
    All the other functions are the same so you just need to import this function and rerun the first file.
    """
    pass
