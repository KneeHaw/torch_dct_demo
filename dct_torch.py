import torch

def dct2(x: torch.Tensor, norm=None, axis=-1):
    s = x.shape
    assert norm in [None, "ortho"]
    assert axis in [-1] + list(range(0, len(s)))
    N = s[axis]
    
    
    def get_coeff(vec, coeff):
        y = torch.zeros(N)  
        for k in range(N):
            y[k] = coeff[k] * 2 * torch.sum(torch.tensor([vec[n] * torch.cos(torch.tensor(torch.pi * k * (2*n + 1) / (2*N))) for n in range(N)]))
        return y
    
    _x = torch.transpose(x, axis, -1).unsqueeze(0)
    _x = _x.reshape((-1, N))
    _x = torch.split(_x, 1, 0)
    coeffs = [1 for _ in range(N)] if norm==None else [torch.sqrt(torch.tensor([1 / (4*N)]))] + [torch.sqrt(torch.tensor([1 / (2*N)])) for _ in range(N-1)]
    out = []
    for sample in _x:
        print("sample: ", sample.shape)
        out.append(get_coeff(sample, coeffs))
    return out

def dct_loop(tensor):
    coeffs = []
    for idx in range(len(tensor.shape)):
        coeffs.append(dct2(tensor, norm="ortho", axis=idx))
    
    return coeffs

def test_data():
    test_tensor_shapes = [[100], [480, 680], [3, 480, 480]]
    for shape in test_tensor_shapes:
        rand_mat = torch.rand(shape)
        coeffs = dct_loop(rand_mat)
        
        print([ci.shape for ci in coeffs])
    
    
if __name__ == "__main__":
    test_data()