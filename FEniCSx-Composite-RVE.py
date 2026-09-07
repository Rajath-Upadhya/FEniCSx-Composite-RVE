#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""

# Import
import numpy as np
import dolfinx
import ufl
from mpi4py import MPI
import gmsh
import basix
from dolfinx.fem.petsc import LinearProblem

#---------------------- Inputs ----------------------------------------------

# --- Composite strength ---
sigma_matrix_strength = 80.0   # MPa — adjust to your chosen material data

# 1 = Case A, 2 = Case B
case_a_or_b = 1

# Geometry parameters (in micrometers)

L = 16.0    # RVE side length

R = 3.5     # fibre radius

# Material constants per cell
E_vals  = {1: 3.462e3, 2: 17.5e3}   # matrix, fibre (transverse)
nu_vals = {1: 0.418,  2: 0.46}

target_strain = 0.1

#--------------------------------------------------------------------



def generate_case_a_mesh():
    
    gmsh.initialize()
    gmsh.model.add('RVE_Case_A')
    
    
    occ = gmsh.model.occ
    
    rectangle = occ.addRectangle(0,0,0,L,L)
    
    fiber1 = occ.addDisk(4.0, 8.0, 0, R, R)
    fiber2 = occ.addDisk(12.0, 8.0, 0, R, R)
    
    out_dimtags, out_dimtags_map = occ.fragment([(2, rectangle)], [(2, fiber1), (2, fiber2)])
    occ.synchronize()
    
    
    fiber_tags = []
    matrix_tags = []

    centers = [(4.0, 8.0), (12.0, 8.0)]
    
    for dim, tag in gmsh.model.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        is_fiber = any(
            (com[0] - cx)**2 + (com[1] - cy)**2 < R**2
            for cx, cy in centers
        )
        if is_fiber:
            fiber_tags.append(tag)
        else:
            matrix_tags.append(tag)
    
    gmsh.model.addPhysicalGroup(2, matrix_tags, 1, name="Matrix")
    gmsh.model.addPhysicalGroup(2, fiber_tags,  2, name="Fibers")

    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)

    gmsh.model.mesh.generate(2)

    gmsh.write("rve_case_a.msh")
    
    # open the GUI to inspect the mesh
    # gmsh.fltk.run()


    gmsh.finalize()
   
def generate_case_b_mesh():
    
    gmsh.initialize()
    gmsh.model.add('RVE_Case_B')
    
    
    occ = gmsh.model.occ
    
    rectangle = occ.addRectangle(0,0,0,L,L)
    
    fiber1 = occ.addDisk(4.5,11.5, 0, R, R)
    fiber2 = occ.addDisk(11.5, 4.5, 0, R, R)
    
    out_dimtags, out_dimtags_map = occ.fragment([(2, rectangle)], [(2, fiber1), (2, fiber2)])
    occ.synchronize()
    
    
    fiber_tags = []
    matrix_tags = []

    centers = [(4.5,11.5), (11.5, 4.5)]
    
    for dim, tag in gmsh.model.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        is_fiber = any(
            (com[0] - cx)**2 + (com[1] - cy)**2 < R**2
            for cx, cy in centers
        )
        if is_fiber:
            fiber_tags.append(tag)
        else:
            matrix_tags.append(tag)
    
    gmsh.model.addPhysicalGroup(2, matrix_tags, 1, name="Matrix")
    gmsh.model.addPhysicalGroup(2, fiber_tags,  2, name="Fibers")

    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)

    gmsh.model.mesh.generate(2)

    gmsh.write("rve_case_b.msh")

    gmsh.finalize()

if case_a_or_b == 1:
    generate_case_a_mesh()
    
    domain, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("rve_case_a.msh", MPI.COMM_WORLD, 0, gdim=2) 
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh_case_a.xdmf", "w") as f:
        f.write_mesh(domain)
        
elif case_a_or_b == 2:
    generate_case_b_mesh()
    domain, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("rve_case_b.msh", MPI.COMM_WORLD, 0, gdim=2) 
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh_case_b.xdmf", "w") as f:
        f.write_mesh(domain)
else:
    print('Select case 1 or case 2')


    
def left(x):
    return np.isclose(x[0], 0.0)

def right(x):
    return np.isclose(x[0], L)

def top(x):
    return np.isclose(x[1], L)

def bottom(x):
    return np.isclose(x[1], 0.0)

fdim = domain.topology.dim - 1

left_side = dolfinx.mesh.locate_entities_boundary(domain, fdim, left)
right_side = dolfinx.mesh.locate_entities_boundary(domain, fdim, right)
bottom_side = dolfinx.mesh.locate_entities_boundary(domain, fdim, bottom)
top_side = dolfinx.mesh.locate_entities_boundary(domain, fdim, top)

facets = np.hstack([left_side, right_side, bottom_side, top_side])

marked_values = np.hstack([
    np.full_like(left_side, 1),           # Mark left facets with 1
    np.full_like(right_side, 2),          # Mark right facets with 2
    np.full_like(bottom_side, 3),         # Mark bottom facets with 3
    np.full_like(top_side, 4)             # Mark top facets with 4
    ])

sorted_sides = np.argsort(facets)

facet_tags = dolfinx.mesh.meshtags(domain, fdim, facets[sorted_sides], marked_values[sorted_sides])
gdim = domain.geometry.dim



element_type_1 = 'Lagrange'

disp_element = basix.ufl.element(element_type_1, domain.topology.cell_name(), degree = 1, shape=(gdim,))

U = dolfinx.fem.functionspace(domain, disp_element)
disp_d = ufl.TrialFunction(U)
disp_delta = ufl.TestFunction(U)
u = dolfinx.fem.Function(U, name= "Displacement")


# Lame parameters as DG0 fields
V0  = dolfinx.fem.functionspace(domain, ("DG", 0))
lam = dolfinx.fem.Function(V0)
mu  = dolfinx.fem.Function(V0)

matrix_cells = cell_tags.find(1)
fibre_cells  = cell_tags.find(2)

for tag, cells in [(1, matrix_cells), (2, fibre_cells)]:
    E, nu = E_vals[tag], nu_vals[tag]
    lam.x.array[cells] = (E*nu)/((1+nu)*(1-2*nu))
    mu.x.array[cells]  = E/(2*(1+nu))
    

# 1. Corner Node: Prevent rigid body motion in Y (Allows Poisson contraction)
def corner_locator(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

corner_entities = dolfinx.mesh.locate_entities(domain, 0, corner_locator)
corner_dofs = dolfinx.fem.locate_dofs_topological(U.sub(1), 0, corner_entities)
bc_corner_y = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), corner_dofs, U.sub(1)) 


delta_x = (target_strain * L) / 2.0  

# 2. Left Boundary: Pull LEFT (Negative X direction)
val_left = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(-delta_x))
left_dofs_x = dolfinx.fem.locate_dofs_topological(U.sub(0), fdim, left_side)  
load_left = dolfinx.fem.dirichletbc(val_left, left_dofs_x, U.sub(0))  
    
# 3. Right Boundary: Pull RIGHT (Positive X direction)
val_right = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(delta_x))
right_dofs_x = dolfinx.fem.locate_dofs_topological(U.sub(0), fdim, right_side)
load_right = dolfinx.fem.dirichletbc(val_right, right_dofs_x, U.sub(0))  
    
bcs = [bc_corner_y, load_left, load_right]

def eps(u): return ufl.sym(ufl.grad(u))
def sigma(u): return lam*ufl.tr(eps(u))*ufl.Identity(2) + 2*mu*eps(u)

zero_body = dolfinx.fem.Constant(domain, np.zeros(gdim))
lhs_form  = ufl.inner(sigma(disp_d), eps(disp_delta)) * ufl.dx
rhs_form  = ufl.inner(zero_body, disp_delta) * ufl.dx


problem = LinearProblem(lhs_form, rhs_form, bcs=bcs,
                            petsc_options={"ksp_type": "preonly",
                                           "pc_type": "lu"})

uh = problem.solve()
uh.name ='Displacement-field solution'

V0 = dolfinx.fem.functionspace(domain, ("DG", 0, (gdim,gdim)))
sig_exp = dolfinx.fem.Expression(sigma(uh), V0.element.interpolation_points())
sig = dolfinx.fem.Function(V0, name="Stress")
sig.interpolate(sig_exp)

if case_a_or_b == 1:
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results_case_a.xdmf", "w") as f:
        f.write_mesh(domain)
        f.write_function(uh)
        f.write_function(sig)
elif case_a_or_b == 2:
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results_case_b.xdmf", "w") as f:
        f.write_mesh(domain)
        f.write_function(uh)
        f.write_function(sig)

area = L * L
sigma_xx = sigma(uh)[0, 0] 
stress_integral_form = dolfinx.fem.form(sigma_xx * ufl.dx)
avg_sigma_xx = dolfinx.fem.assemble_scalar(stress_integral_form) / area
E_transverse = avg_sigma_xx / target_strain
print(f"Effective Transverse Modulus: {E_transverse} MPa")

sigma_avg_xx = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(sigma(uh)[0, 0] * ufl.dx)
) / area


V_scalar = dolfinx.fem.functionspace(domain, ("DG", 0))
sig_xx_scalar_expr = dolfinx.fem.Expression(
    sigma(uh)[0, 0], V_scalar.element.interpolation_points()
)
sig_xx_scalar = dolfinx.fem.Function(V_scalar)
sig_xx_scalar.interpolate(sig_xx_scalar_expr)

matrix_cells = cell_tags.find(1)
sig_xx_values = sig_xx_scalar.x.array

sig_xx_matrix = sig_xx_values[matrix_cells]
sigma_local_max = np.max(sig_xx_matrix)

print(f"Peak matrix σ_xx = {sigma_local_max:.2f} MPa")

# --- Stress concentration factor ---
K = sigma_local_max / sigma_avg_xx
print(f"Stress concentration factor K = {K:.3f}")


composite_strength = sigma_matrix_strength / K
print(f"Predicted transverse strength = {composite_strength:.1f} MPa")
















