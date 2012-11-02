#!/usr/bin/python

import math
import random

class Pulsar:
    """ Class to store an individual pulsar"""
    def __init__(self,
                 period=None,
                 dm=None,
                 gl=None,
                 gb=None,
                 galCoords=None,
                 r0=None,
                 dtrue=None,
                 lum_1400=None,
                 spindex=None,
                 alpha=None,
                 rho=None,
                 width_degree=None,
                 snr=None,
                 beaming=None,
                 scindex=-3.86,
                 gpsFlag=0,
                 gpsA=None,
                 brokenFlag=0,
                 brokenSI=None):
        """___init___ function for the Pulsar class"""
        self.period = period
        self.dm = dm
        
        # convert to -180->+180 range
        if gl >180.:
            gl -= 360.

        self.gl = gl
        self.gb = gb
        self.galCoords = galCoords
        self.r0 = r0
        self.dtrue = dtrue

        self.lum_1400 = lum_1400
        self.spindex = spindex
        self.scindex = scindex

        self.alpha = alpha
        self.rho = rho
        self.width_degree = width_degree
            
        self.beaming = beaming

        self.gpsFlag = gpsFlag
        self.gpsA = gpsA

        self.brokenFlag= brokenFlag
        self.brokenSI = brokenSI

        self.snr=snr

    # methods to calculate derived properties
    def s_1400(self):
        """Calculate the flux of the pulsar"""
        
        try:
            return self.lum_1400 / self.dtrue / self.dtrue
        except TypeError:
            print "lum_1400 or dtrue not defined"

    def width_ms(self):
        """Return the pulse width in milliseconds"""
        try:
            return self.width_degree * self.period / 360.0
        except TypeError:
            print "period or width_degree not defined"

