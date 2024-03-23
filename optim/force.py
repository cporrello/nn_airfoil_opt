import torch

# Compute aerodynamic forces from coefficient of pressure
def get_aerodynamic_forces(x, y, p, cnx2=120) -> tuple:

    # Number of edges is one less than the number of points
    ne = len(p) - 1

    # Integration of the pressure distribution to get forces
    d = torch.matmul(y[0:cnx2]-y[1:cnx2+1], (p[0:cnx2] + p[1:cnx2+1])/2.0)
    l = torch.matmul(x[1:cnx2+1]-x[0:cnx2], (p[0:cnx2] + p[1:cnx2+1])/2.0)

    # l = torch.matmul(x[1:ne+1]-x[0:ne], (p[0:ne] + p[1:ne+1])/2.0)
    # d = torch.matmul(y[0:ne]-y[1:ne+1], (p[0:ne] + p[1:ne+1])/2.0)
    
    return l, d

