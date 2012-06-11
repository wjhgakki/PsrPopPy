PsrPopPy
========

Python implementation of PSRPOP (which was written by D Lorimer).
Several of the old models from that (e.g. NE2001) are still included in their native fortran, since re-writing those is beyond the scope of this work. Currently, only a rudimentary makefile is included. This is something that needs work from a willing volunteer!

The other main requirement is [matplotlib](matplotlib.sourceforge.net), which is used for the visualization stuff. It has a very useful API for making simple GUIs, as well as making beautiful plots.

Compiling
---------

To compile on mac or linux, go into the fortran directory, edit make_mac.csh or make_linux.csh (as appropriate).
Change the variable gf to point to your local gfortran compiler, then run the script. Fingers crossed, it should all work.

A brief description of the "executables" follows.

populate.py
-----------

Create a population mode using user-defined parameters.

dosurvey.py 
-----------

Run a population model through a survey. Pre-defined surveys are given, but a user may also create their own.

view.py
-------

Program for making quick histograms of population model

visualize.py
------------

More detailed population model viewer. Make histograms, scatter plots, etc. This is a little slow, I think due to the large number of points being plotted. Thi s might want some work.