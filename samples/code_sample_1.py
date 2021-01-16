# -*- coding: utf-8 -*-
# Here are some samples about 2D-frame analysis

import os
import sys
os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.abspath('..'))
from pyfacade.pyeng import Beam2, Bsolver
from pyfacade.strcals import build_frame, load_frame


# region <Sample A: Build 2D Frame Model Manually>

# Step 1. Define Coordinate of Node in format: [[x1,y1],[x2,y2]...]
nodes = [[6750, 0], [6250, 0], [5000, 0], [3750, 0], [3000, 0], [0, 0]]

# Step 2. Define Beam Set in format: [[node1, node2]...]
beams = [[i, i + 1] for i in range(len(nodes) - 1)]

# Step 3. Define modulus of elasticity of beam
E_s = 205000  # steel

# Step 4. Define section area and moment of inertia of beam
sec_A = 50*100  # 50x100 solid bar
sec_I = 50*100**3/12

# Step 5. Create list of beam object
b = [Beam2(*x, sec_A, sec_I, E_s, nodes) for x in beams]

# Step 6. Define restrain in format: {node:[x, y, rotate]...}, 1=restrained
restr = {1: [1, 1, 0], 4: [1, 1, 0], 5: [0, 1, 0]}

# Step 7. Define end release of each beam in format: {beam:[start, end]...},  1=released
brels = {2: [(0, 0), (0, 1)]}  # release rotation at end of beam 2

# Step 8. Apply load on beams in format: {beam:(ual, udl)...}
Q = {i: (0, 2) for i in range(len(beams))}  # udl=2N/mm on all beams

# Step 9. Apply global node force in format: {node:[Fx, Fy, Mz]...}
F = {2: [1000, 0, 0]}  # Fx = 1000N @ node 2

# Step 10. Setup solver instance
model = Bsolver(b, nodes, restr, brels, Q, F)

# Step 11. Run solve
model.solve()

# Step 12. Show diagram
model.show_mdl()
model.show_diag('Axial Force', DivN=5, Scale=1, Accr=4)
model.show_diag('Shear', DivN=5, Scale=10, Accr=4)
model.show_diag('Moment', DivN=20, Scale=3000, Accr=4)
model.show_diag('Deflection', DivN=20, Scale=0.005, Accr=4)

# Step 12. Output Results
print(model.get_summary("Axial Force", envelop=False))
print(model.get_summary("Shear", envelop=False))
print(model.get_summary("Moment", envelop=False))
print(model.get_summary("Deflection", envelop=False))

# endregion


# region <Sample B: Build 2D Frame Model Interactively via AutoCAD, and save model information to a file>

sketch = os.path.abspath('.') + "\\frame_sketch.dwg"  # sample sketch
_, model = build_frame(drawing=sketch,
                           sec_mod=[205000],
                           sec_area=[50*100],
                           sec_inert=[50*100**3/12],
                           auto_assign=True,
                           file_name="model.json")
model.show_diag('Moment', DivN=20, Scale=2000, Accr=4)

# endregion


# region <Sample C: Read Saved Model Information, Modify and Re-calculate>

md = load_frame("model.json", build=False)  # read data from file, but don't solve
# Define Beam Set
beams = [Beam2(*md.beams[x], md.A[x], md.I[x], md.E[x], md.nodes) for x in range(len(md.beams))]
# Setup solver instance, based on new load case
new_udl = {}
new_pl = {4: [0, -2000, 0]}
model = Bsolver(beams, md.nodes, md.restrain, md.release, new_udl, new_pl)
model.solve()
model.show_diag('Moment', DivN=20, Scale=2000, Accr=4)

# endregion


