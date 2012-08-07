r"""
A finite volume flux-differencing solver.
"""
# Solver superclass
from pyclaw.solver import Solver, CFLError

# Reconstructor
try:
    # load c-based WENO reconstructor (PyWENO)
    from pyclaw.limiters import reconstruct as recon
except ImportError:
    # load old WENO5 reconstructor
    from pyclaw.limiters import recon


def start_step(solver,solution):
    r"""
    Dummy routine called before each step
    
    Replace this routine if you want to do something before each time step.
    """
    pass

class FluxDiffSolver(Solver):
    r"""
    Superclass for all FluxDiffND solvers.

    Implements Runge-Kutta time stepping and the basic form of a 
    semi-discrete step (the dq() function).  If another method-of-lines
    solver is implemented in the future, it should be based on this class,
    which then ought to be renamed to something like "MOLSolver".

    .. attribute:: start_step
    
        Function called before each time step is taken.
        The required signature for this function is:
        
        def start_step(solver,solution)

    .. attribute:: lim_type

        Limiter(s) to be used.
        0: No limiting.
        1: TVD reconstruction.
        2: WENO reconstruction.
        ``Default = 2``

    .. attribute:: weno_order

        Order of the WENO reconstruction. From 1st to 17th order (PyWENO)
        ``Default = 5``

    .. attribute:: time_integrator

        Time integrator to be used.
        Euler: forward Euler method.
        SSP33: 3-stages, 3rd-order SSP Runge-Kutta method.
        SSP104: 10-stages, 4th-order SSP Runge-Kutta method.
        ``Default = 'SSP104'``

    .. attribute:: char_decomp

        Type of WENO reconstruction.
        0: conservative variables WENO reconstruction (standard).
        1: characteristic-wise WENO reconstruction.
        2: transmission-based WENO reconstruction.
        ``Default = 0``

    .. attribute:: tfluct_solver

        Whether a total fluctuation solver have to be used. If True the function
        that calculates the total fluctuation must be provided.
        ``Default = False``

    .. attribute:: aux_time_dep

        Whether the auxiliary array is time dependent.
        ``Default = False``
    
    .. attribute:: kernel_language

        Specifies whether to use wrapped Fortran routines ('Fortran')
        or pure Python ('Python').  
        ``Default = 'Fortran'``.

    .. attribute:: mbc

        Number of ghost cells.
        ``Default = 3``

    .. attribute:: fwave
    
        Whether to split the flux jump (rather than the jump in Q) into waves; 
        requires that the Riemann solver performs the splitting.  
        ``Default = False``

    .. attribute:: cfl_desired

        Desired CFL number.
        ``Default = 2.45``

    .. attribute:: cfl_max

        Maximum CFL number.
        ``Default = 2.50``

    .. attribute:: dq_src

        Whether a source term is present. If it is present the function that 
        computes its contribution must be provided.
        ``Default = None``
    """
    
    # ========================================================================
    #   Initialization routines
    # ========================================================================
    def __init__(self):
        r"""
        Set default options for SharpClawSolvers and call the super's __init__().
        """
        self.start_step = start_step
        self.weno_order = 5
        self.time_integrator = 'SSP104'
        self.char_decomp = 0
        self.aux_time_dep = False
        self.kernel_language = 'Python'
        self.mbc = (self.weno_order+1)/2
        self.cfl_desired = 0.45
        self.cfl_max = 0.5
        self.dq_src = None
        self._method = None
        self._rk_stages = None
        self.flux = None
        self.limiters = None
        
        # Call general initialization function
        super(FluxDiffSolver,self).__init__()
        
    # ========== Time stepping routines ======================================
    def step(self,solution):
        """Evolve q over one time step.

        Take one Runge-Kutta time step using the method specified by
        self..time_integrator.  Currently implemented methods:

        'Euler'  : 1st-order Forward Euler integration
        'SSP33'  : 3rd-order strong stability preserving method of Shu & Osher
        'SSP104' : 4th-order strong stability preserving method Ketcheson
        """
        state = solution.states[0]

        self.start_step(self,solution)

        try:
            if self.time_integrator=='Euler':
                deltaq=self.dq(state)
                state.q+=deltaq

            elif self.time_integrator=='SSP33':
                deltaq=self.dq(state)
                self._rk_stages[0].q=state.q+deltaq
                self._rk_stages[0].t =state.t+self.dt
                deltaq=self.dq(self._rk_stages[0])
                self._rk_stages[0].q= 0.75*state.q + 0.25*(self._rk_stages[0].q+deltaq)
                self._rk_stages[0].t = state.t+0.5*self.dt
                deltaq=self.dq(self._rk_stages[0])
                state.q = 1./3.*state.q + 2./3.*(self._rk_stages[0].q+deltaq)

            elif self.time_integrator=='SSP104':
                s1=self._rk_stages[0]
                s2=self._rk_stages[1]
                s1.q = state.q.copy()

                deltaq=self.dq(state)
                s1.q = state.q + deltaq/6.
                s1.t = state.t + self.dt/6.

                for i in xrange(4):
                    deltaq=self.dq(s1)
                    s1.q=s1.q + deltaq/6.
                    s1.t =s1.t + self.dt/6.

                s2.q = state.q/25. + 9./25 * s1.q
                s1.q = 15. * s2.q - 5. * s1.q
                s1.t = state.t + self.dt/3.

                for i in xrange(4):
                    deltaq=self.dq(s1)
                    s1.q=s1.q + deltaq/6.
                    s1.t =s1.t + self.dt/6.

                deltaq = self.dq(s1)
                state.q = s2.q + 0.6 * s1.q + 0.1 * deltaq
            else:
                raise Exception('Unrecognized time integrator')
        except CFLError:
            return False


    def dq(self,state):
        """
        Evaluate dq/dt * (delta t)
        """

        deltaq = self.dq_hyperbolic(state)

        # Check here if we violated the CFL condition, if we did, return 
        # immediately to evolve_to_time and let it deal with picking a new
        # dt
        if self.cfl.get_cached_max() > self.cfl_max:
            raise CFLError('cfl_max exceeded')

        if self.dq_src is not None:
            deltaq+=self.dq_src(self,state,self.dt)

        return deltaq

    def dq_hyperbolic(self,state):
        raise NotImplementedError('You must subclass SharpClawSolver.')

         
    def allocate_rk_stages(self,solution):
        r"""
        Instantiate State objects for Runge--Kutta stages.

        This routine is only used by method-of-lines solvers
        not by the Classic solvers.  It allocates additional State objects
        to store the intermediate stages used by Runge--Kutta time integrators.

        If we create a MethodOfLinesSolver subclass, this should be moved there.
        """
        if self.time_integrator   == 'Euler':  nregisters=1
        elif self.time_integrator == 'SSP33':  nregisters=2
        elif self.time_integrator == 'SSP104': nregisters=3
 
        state = solution.states[0]
        # use the same class constructor as the solution for the Runge Kutta stages
        State = type(state)
        self._rk_stages = []
        for i in xrange(nregisters-1):
            #Maybe should use State.copy() here?
            self._rk_stages.append(State(state.patch,state.meqn,state.maux))
            self._rk_stages[-1].aux_global       = state.aux_global
            self._rk_stages[-1].set_mbc(self.mbc)
            self._rk_stages[-1].t                = state.t
            if state.maux > 0:
                self._rk_stages[-1].aux              = state.aux




# ========================================================================
class FluxDiffSolver1D(FluxDiffSolver):
# ========================================================================
    """
    FluxDiff solver for one-dimensional problems.
    
    Used to solve 1D hyperbolic systems using the FluxDiff algorithms,
    which are based on WENO reconstruction and Runge-Kutta time stepping.
    """
    def __init__(self):
        r"""
        See :class:`FluxDiffSolver1D` for more info.
        """   
        self.ndim = 1
        super(FluxDiffSolver1D,self).__init__()


    def setup(self,solution):
        """
        Allocate RK stage arrays and fortran routine work arrays.
        """
        self.mbc = (self.weno_order+1)/2

        # This is a hack to deal with the fact that petsc4py
        # doesn't allow us to change the stencil_width (mbc)
        state = solution.state
        state.set_mbc(self.mbc)
        # End hack

        self.allocate_rk_stages(solution)
        state = solution.states[0]
        self.allocate_bc_arrays(state)


    def dq_hyperbolic(self,state):
        r"""
        Compute dq/dt * (delta t) for the hyperbolic hyperbolic system.
        """
    
        import numpy as np

        self.apply_q_bcs(state)
        q = self.qbc 

        patch = state.patch
        mx = patch.ng[0]

        ixy=1

        if self.kernel_language=='Fortran':
            raise NotImplementedError

        elif self.kernel_language=='Python':

            dtdx = np.zeros( (mx+2*self.mbc) ,order='F')
            dq   = np.zeros( (state.meqn,mx+2*self.mbc) ,order='F')
            f    = np.zeros( (state.meqn,mx+2*self.mbc) ,order='F')

            # Find local value for dt/dx
            if state.mcapa>=0:
                dtdx = self.dt / (patch.d[0] * state.aux[state.mcapa,:])
            else:
                dtdx += self.dt/patch.d[0]
 
            aux=self.auxbc
            if aux.shape[0]>0:
                aux_l=aux[:,:-1]
                aux_r=aux[:,1: ]
            else:
                aux_l = None
                aux_r = None

            if self.char_decomp==0: #No characteristic decomposition
                ql,qr=recon.weno(self.weno_order,q)
            else:
                raise NotImplementedError

            # Solve Riemann problem at each interface
            q_l=qr[:,:-1]
            q_r=ql[:,1: ]
            flux,s = self.flux(q_l,q_r,aux_l,aux_r,state.aux_global)

            # Loop limits for local portion of patch
            # THIS WON'T WORK IN PARALLEL!
            LL = self.mbc - 1
            UL = patch.ng[0] + self.mbc + 1

            # Compute maximum wave speed
            cfl = 0.0
            for mw in xrange(self.mwaves):
                smax1 = np.max( dtdx[LL  :UL]  *s[mw,LL-1:UL-1])
                smax2 = np.max(-dtdx[LL-1:UL-1]*s[mw,LL-1:UL-1])
                cfl = max(cfl,smax1,smax2)

            # Compute dq
            for m in xrange(state.meqn):
                dq[m,LL:UL] = -dtdx[LL:UL]*(flux[m,LL:UL] - flux[m,LL-1:UL-1])

        else: raise Exception('Unrecognized value of solver.kernel_language.')

        self.cfl.update_global_max(cfl)
        return dq[:,self.mbc:-self.mbc]
