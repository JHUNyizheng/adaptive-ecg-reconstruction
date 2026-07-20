import numpy as np

def gram_schmidt_orthogonalize(xyz_signal):
    """
    Perform Gram-Schmidt orthogonalization on XYZ signals.
    :param xyz_signal: Input signal, supports two shapes:
                      - 3D: (num_samples, 3, seq_length) (samples x channels x time_points)
                      - 2D: (3, seq_length) (channels x time_points)
    :return: orthogonalized_xyz: Orthogonalized signal with the same shape as input
    """
    xyz = xyz_signal.copy()
    
    if len(xyz.shape) == 3:
        num_samples, n_channels, seq_length = xyz.shape
        orthogonalized_xyz = np.zeros_like(xyz)
        
        for i in range(num_samples):
            sample = xyz[i, :, :]
            
            x = sample[0, :]
            x = x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else x
            
            y = sample[1, :]
            y_proj = np.dot(y, x) * x
            y_ortho = y - y_proj
            y_ortho = y_ortho / np.linalg.norm(y_ortho) if np.linalg.norm(y_ortho) != 0 else y_ortho
            
            z = sample[2, :]
            z_proj_x = np.dot(z, x) * x
            z_proj_y = np.dot(z, y_ortho) * y_ortho
            z_ortho = z - z_proj_x - z_proj_y
            z_ortho = z_ortho / np.linalg.norm(z_ortho) if np.linalg.norm(z_ortho) != 0 else z_ortho
            
            orthogonalized_xyz[i, :, :] = np.vstack([x, y_ortho, z_ortho])
    
    elif len(xyz.shape) == 2:
        n_channels, seq_length = xyz.shape
        
        x = xyz[0, :]
        x = x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else x
        
        y = xyz[1, :]
        y_proj = np.dot(y, x) * x
        y_ortho = y - y_proj
        y_ortho = y_ortho / np.linalg.norm(y_ortho) if np.linalg.norm(y_ortho) != 0 else y_ortho
        
        z = xyz[2, :]
        z_proj_x = np.dot(z, x) * x
        z_proj_y = np.dot(z, y_ortho) * y_ortho
        z_ortho = z - z_proj_x - z_proj_y
        z_ortho = z_ortho / np.linalg.norm(z_ortho) if np.linalg.norm(z_ortho) != 0 else z_ortho
        
        orthogonalized_xyz = np.vstack([x, y_ortho, z_ortho])
    
    else:
        raise ValueError(f"Unsupported input shape: {xyz.shape}, only 3D or 2D shapes are supported")
    
    return orthogonalized_xyz