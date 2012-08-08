#!/usr/bin/env python
# encoding: utf-8

def flux_advection(q_l, q_r, aux_l, aux_r, problem_data):
    r""" Flux function for advection."""
    import numpy as np

    s = np.zeros((1,q_l.shape[-1])) + problem_data['u']
    if problem_data['u']>0:
        flux = q_l
    else:
        flux = q_r
    return flux, s

def advection(kernel_language='Python',iplot=False,htmlplot=False,use_petsc=False,solver_type='classic',outdir='./_output'):
    """
    Example python script for solving the 1d advection equation.
    """
    import numpy as np

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D()
    elif solver_type=='classic':
        solver = pyclaw.ClawSolver1D()
    elif solver_type=='fluxdiff':
        solver = pyclaw.FluxDiffSolver1D()
        solver.flux = flux_advection


    solver.kernel_language = kernel_language
    from clawpack.riemann import rp_advection
    solver.num_waves = rp_advection.num_waves
    if solver.kernel_language=='Python': 
        solver.rp = rp_advection.rp_advection_1d
    else:
        from clawpack import riemann
        solver.rp = riemann.rp1_advection

    solver.bc_lower[0] = 2
    solver.bc_upper[0] = 2

    x = pyclaw.Dimension('x',0.0,1.0,100)
    domain = pyclaw.Domain(x)
    num_eqn = 1
    state = pyclaw.State(domain,num_eqn)
    state.problem_data['u']=1.

    grid = state.grid
    xc=grid.x.centers
    beta=100; gamma=0; x0=0.75
    state.q[0,:] = np.exp(-beta * (xc-x0)**2) * np.cos(gamma * (xc - x0))

    claw = pyclaw.Controller()
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.outdir = outdir

    claw.tfinal =1.0
    status = claw.run()

    if htmlplot:  pyclaw.plot.html_plot(outdir=outdir)
    if iplot:     pyclaw.plot.interactive_plot(outdir=outdir)

if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(advection)
