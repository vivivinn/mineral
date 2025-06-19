from firedrake import *
from echemfem import TransientEchemSolver, RectangleBoundaryLayerMesh
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--family", type=str, default='CG')
args, _ = parser.parse_known_args()
SUPG = args.family == "CG"

class FlowSea(TransientEchemSolver):
    def __init__(self, inlet_velocity=(0.0, 0.0), outlet_pressure=0.0):



        # Store flow parameters for later use when setting up the solver
        self.inlet_velocity = Constant(inlet_velocity)
        self.outlet_pressure = Constant(outlet_pressure)

        self.set_boundary_markers()

        # Define reaction rate constants
        k1f = 3.7e-2
        k1r = 26.05
        k2f = 2.1756
        k2r = 9.71e-5
        k3f = 4.88e7
        k3r = 59.44
        k4f = 5.88e6
        k4r = 3.06e5
        k5f = 2.31e7
        k5r = 1.4

        Kw = 1e-8  # mol^2/m^6 in mol/m^3 units
        c_oh_bulk = 1e-3
        c_h_bulk = 1e-5
        dic = 2.5

        k_co2 = 10 ** (-6.4)
        k_co3 = 10 ** (-10.3)

        c_co2_bulk = 2.5 / (1 + k_co2 / (1e-8) + k_co2 * k_co3 / ((1e-8) ** 2))  # [CO2]
        c_hco3_bulk = 2.5 / (1 + (1e-8) / k_co2 + k_co3 / (1e-8))  # [HCO3-]
        c_co3_bulk = 2.5 / (1 + (1e-8) / k_co3 + ((1e-8) ** 2) / (k_co2 * k_co3))  # [CO3^2-]


        cl = 500
        na = 500

        conc_params = []
        conc_params.append({"name": "co2", "diffusion coefficient": 1.9e-9, "bulk": c_co2_bulk, "z": 0})
        conc_params.append({"name": "oh",  "diffusion coefficient": 4.47e-9, "bulk": c_oh_bulk,  "z": -1})
        conc_params.append({"name": "h",   "diffusion coefficient": 9.13e-9, "bulk": c_h_bulk,   "z": 1})
        conc_params.append({"name": "co3", "diffusion coefficient": 9.2e-10, "bulk": c_co3_bulk, "z": -2})
        conc_params.append({"name": "hco3", "diffusion coefficient": 9.7e-10, "bulk": c_hco3_bulk, "z": -1})
        conc_params.append({"name": "cl",  "diffusion coefficient": 2.03e-9, "bulk": cl, "z": -1})
        conc_params.append({"name": "na",  "diffusion coefficient": 1.33e-9, "bulk": na, "z": 1})

        def bulk_reaction(c):
            from ufl import conditional, ge, lt
            cpos = [conditional(ge(ci, 0.0), ci, 0.0) for ci in c]
            c_co2 = cpos[0]
            c_oh = cpos[1]
            c_h = cpos[2]
            c_co3 = cpos[3]
            c_hco3 = cpos[4]

            penalty_strength = 1e10
            penalties = [penalty_strength * conditional(lt(ci, 0.0), -ci, 0.0) for ci in c]
            

            dic_target = 2.5
            total_dic = c_co2 + c_co3 + c_hco3
            penalty_dic = 1e8 * (dic_target - total_dic)

            return [
                # d[CO2]/dt
                - k1f * c_co2 * c_oh + k1r * c_hco3 + penalties[0] + penalty_dic,
                # d[OH-]/dt
                - k2f * c_co2 + k2r * c_hco3 * c_h - k5f * c_h * c_oh + penalties[1],
                # d[H+]/dt
                + k2f * c_co2 - k2r * c_hco3 * c_h - k5f * c_h * c_oh + penalties[2],
                # d[CO3^2-]/dt
                - k3f * c_co3 * c_oh + k3r * c_hco3 - k4f * c_co3 * c_h + k4r * c_hco3 + penalties[3] + penalty_dic,
                # d[HCO3-]/dt
                + k1f * c_co2 * c_oh - k1r * c_hco3 + k3f * c_co3 * c_oh - k3r * c_hco3 + k4f * c_co3 * c_h - k4r * c_hco3 + penalties[4] + penalty_dic,
                # d[Cl-]/dt (no reaction)
                penalties[5],
                # d[Na+]/dt (no reaction)
                penalties[6]
            ]


        def current(cef):
            j = 10.0
            def curr(u):
                return cef * j
            return curr

        echem_params = [{
            "reaction": current(1.0),
            "stoichiometry": {"oh":1},
            "electrons": 1,
            "boundary": "electrode",
        }]

        physical_params = {
            "flow": ["diffusion", "convection"],
            "F": 96485,
            "bulk reaction": bulk_reaction,
        }

        lx = 1e-3 #m
        ly = 1e-3 #m
        mesh = RectangleBoundaryLayerMesh(50, 50, lx, ly, 50, 1e-6)

        super().__init__(conc_params, physical_params, mesh, echem_params=echem_params, family=args.family, SUPG=SUPG)
 
        self.dirichlet_bcs = {
            ("cl", "farfield"): cl,
            ("na", "farfield"): na,
        }

        #super().__init__(conc_params, physical_params, mesh, echem_params=echem_params)
        self.timestep = 0.01
        self.num_steps = 10
        self.store_all_solutions = False

    def set_boundary_markers(self):
        self.boundary_markers = {
            "electrode": (2,),
            "farfield": (1, 3, 4),
        }



# Run
if __name__ == "__main__":
    solver = FlowSea()
    solver.setup_solver()
    times = [n * solver.timestep for n in range(solver.num_steps + 1)]
    solver.solve(times)




