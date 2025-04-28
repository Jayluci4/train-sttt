# 1. Optimized mutual information calculation that stays on GPU
def approx_mutual_information(self, x, y, bins=16):
    """
    Approximate mutual information between two continuous variables using histogram estimator.
    Optimized to stay on GPU when possible.
    
    MI(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    # Make sure we have tensors on the same device
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, device=self.device)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, device=self.device)
        
    # Normalize to [0, 1] for binning - keep on device for speed
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Avoid division by zero
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    if x_range.item() == 0 or y_range.item() == 0:
        return 0.0
        
    x_norm = (x - x_min) / x_range
    y_norm = (y - y_min) / y_range
    
    # Compute histograms directly on device (torch.histc works on CUDA)
    x_hist = torch.histc(x_norm, bins=bins, min=0, max=1)
    y_hist = torch.histc(y_norm, bins=bins, min=0, max=1)
    
    # Normalize histograms to get probability distributions
    x_hist = x_hist / x_hist.sum()
    y_hist = y_hist / y_hist.sum()
    
    # Compute joint histogram
    joint_hist = torch.zeros((bins, bins), device=self.device)
    
    # Bin indices for each sample
    x_indices = torch.clamp((x_norm * bins).long(), 0, bins-1)
    y_indices = torch.clamp((y_norm * bins).long(), 0, bins-1)
    
    # Count joint occurrences
    for i in range(len(x_indices)):
        # joint_hist[x_indices[i], y_indices[i]] += 1
        joint_hist.index_add_(0, x_indices * bins + y_indices, torch.ones_like(x_indices, dtype=joint_hist.dtype))

    # Normalize joint histogram
    joint_hist = joint_hist / joint_hist.sum()
    
    # Compute entropy terms
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    
    # H(X)
    h_x = -torch.sum(x_hist * torch.log2(x_hist + epsilon))
    
    # H(Y)
    h_y = -torch.sum(y_hist * torch.log2(y_hist + epsilon))
    
    # H(X,Y)
    h_xy = -torch.sum(joint_hist * torch.log2(joint_hist + epsilon))
    
    # I(X;Y) = H(X) + H(Y) - H(X,Y)
    mi = h_x + h_y - h_xy
    
    return mi.item()

# 2. SVD with explicit device and dtype handling for mixed precision
def _svd_based_rank_change(self, current_A, current_B, new_rank, device, dtype):
    """
    Perform SVD-based rank change with explicit device and dtype handling.
    This avoids mixed-precision issues when using SVD on CPU then moving back to GPU.
    
    Args:
        current_A: Current A weights [r, in_features]
        current_B: Current B weights [out_features, r]
        new_rank: New rank value
        device: Device for tensors
        dtype: Data type for tensors
        
    Returns:
        new_A_weight, new_B_weight: New weights with changed rank
    """
    try:
        # Compute AB decomposition
        AB = current_B @ current_A  # [out_features, in_features]
        
        # Compute SVD
        U, S, Vh = torch.linalg.svd(AB, full_matrices=False)
        
        # Prepare new weights with correct size
        if new_rank > current_A.size(0):  # Expanding
            # Calculate new weights with preserved subspace
            S_sqrt = torch.sqrt(S)
            
            # Initialize new weights based on SVD components
            new_A = torch.zeros((new_rank, current_A.size(1)), device=device, dtype=dtype)
            new_B = torch.zeros((current_B.size(0), new_rank), device=device, dtype=dtype)
            
            # Copy existing components
            new_A[:current_A.size(0), :] = (
                torch.diag(S_sqrt[:current_A.size(0)]) @ Vh[:current_A.size(0), :]
            ).to(device=device, dtype=dtype)
            new_B[:, :current_A.size(0)] = (
                U[:, :current_A.size(0)] @ torch.diag(S_sqrt[:current_A.size(0)])
            ).to(device=device, dtype=dtype)
            
            # Initialize new dimensions with orthogonal initialization
            if new_rank > current_A.size(0):
                # Calculate scaling factor for orthogonal initialization
                fan_in = current_A.size(1)
                gain = 1.0 / math.sqrt(fan_in)
                
                # Initialize new rows of A with orthogonal values
                # Use a temporary tensor on CPU for orthogonal init, then transfer
                temp = torch.zeros((new_rank - current_A.size(0), current_A.size(1)))
                nn.init.orthogonal_(temp)
                new_A[current_A.size(0):, :] = (temp * gain).to(device=device, dtype=dtype)
                
                # Initialize new columns of B with zeros (non-expansive)
                new_B[:, current_A.size(0):].zero_()
        else:  # Reducing
            # Use top singular values/vectors for reduction
            S_sqrt = torch.sqrt(S[:new_rank])
            
            # Create new tensors with correct device and dtype
            new_A = (torch.diag(S_sqrt) @ Vh[:new_rank, :]).to(device=device, dtype=dtype)
            new_B = (U[:, :new_rank] @ torch.diag(S_sqrt)).to(device=device, dtype=dtype)
        
        return new_A, new_B
        
    except Exception as e:
        logger.warning(f"SVD-based rank change failed: {e}")
        return None, None
