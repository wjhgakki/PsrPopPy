c==============================================================================
      real function gammafunc(p,a,m)
c==============================================================================
      implicit none
      real p,a,m,x
      x=p/m
      gammafunc = x**(a-1.0)*exp(-1.0*x)
      end
c==============================================================================
