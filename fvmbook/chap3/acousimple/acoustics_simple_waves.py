#!/usr/bin/env python
# encoding: utf-8
    
def fig_31_38(kernel_language='Fortran',solver_type='classic',iplot=False,htmlplot=False,outdir='./_output'):
    r"""Produces the output shown in Figures 3.1 and 3.8 of the FVM book.
    These involve simple waves in the acoustics system."""
    import pyclaw
    import numpy as np

    #=================================================================
    # Import the appropriate solver type, depending on the options passed
    #=================================================================
    if solver_type=='classic':
        solver = pyclaw.ClawSolver1D()
    elif solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D()
    else: raise Exception('Unrecognized value of solver_type.')

    #========================================================================
    # Instantiate the solver and define the system of equations to be solved
    #========================================================================
    solver.kernel_language=kernel_language
    from riemann import rp_acoustics
    solver.mwaves=rp_acoustics.mwaves
    if kernel_language=='Python': 
        solver.rp = rp_acoustics.rp_acoustics_1d
 
    solver.limiters = pyclaw.limiters.tvd.MC
    solver.bc_lower[0] = pyclaw.BC.reflecting
    solver.bc_upper[0] = pyclaw.BC.outflow

    #========================================================================
    # Instantiate the grid and set the boundary conditions
    #========================================================================
    x = pyclaw.Dimension('x',-1.0,1.0,800)
    grid = pyclaw.Grid(x)
    meqn = 2
    state = pyclaw.State(grid,meqn)

    #========================================================================
    # Set problem-specific variables
    #========================================================================
    rho = 1.0
    bulk = 0.25
    state.aux_global['rho']=rho
    state.aux_global['bulk']=bulk
    state.aux_global['zz']=np.sqrt(rho*bulk)
    state.aux_global['cc']=np.sqrt(bulk/rho)

    #========================================================================
    # Set the initial condition
    #========================================================================
    xc=grid.x.center
    beta=100; gamma=0; x0=0.75
    state.q[0,:] = 0.5*np.exp(-80 * xc**2) + 0.5*(np.abs(xc+0.2)<0.1)
    state.q[1,:] = 0.
    
    #========================================================================
    # Set up the controller object
    #========================================================================
    claw = pyclaw.Controller()
    claw.solution = pyclaw.Solution(state)
    claw.solver = solver
    claw.outdir = outdir
    claw.tfinal = 3.0
    claw.nout   = 30

    # Solve
    status = claw.run()

    # Plot results
    if htmlplot:  pyclaw.plot.html_plot(outdir=outdir)
    if iplot:     pyclaw.plot.interactive_plot(outdir=outdir)

if __name__=="__main__":
    from pyclaw.util import run_app_from_main
    output = run_app_from_main(fig_31_38)
