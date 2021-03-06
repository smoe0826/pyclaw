r"""Convenience routines for easily plotting with VisClaw."""

def interactive_plot(outdir='./_output',format='ascii'):
    """
    Convenience function for launching an interactive plotting session.
    """
    from visclaw.plotters import Iplotclaw
    ip=Iplotclaw.Iplotclaw()
    ip.plotdata.outdir=outdir
    ip.plotdata.format=format
    ip.plotloop()

def html_plot(outdir='./_output',format='ascii'):
    """
    Convenience function for creating html page with plots.
    """
    from visclaw.plotters import plotclaw
    plotclaw.plotclaw(outdir,format=format)
