# Copyright (C) 2021 Anarya Ray
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import numpy as np
from GWXtreme.eos_model_selection import Stacking
import h5py
import corner
from multiprocessing import cpu_count, Pool
import time
import emcee as mc
from GWXtreme.eos_prior import is_valid_eos,eos_p_of_rho, spectral_eos,polytrope_eos
import lalsimulation
import matplotlib.pyplot as plt
        
def plot(infile,outputDir,cornerplot={'plot':True,'true vals':None},p_vs_rho={'plot':True,'true_eos':'AP4'}):
    '''
    This method plots the posterior of the spectral
    parameters in a corner plot and also the pressure
    density credible intervals inferred from these
    posteriors in a separate plot. It reurns an array
    containing matplotlib.pyplot.figure and axes objects
    corresponding to each plot
    
    cornerplot :: dictionary saying whether to plot corner or
                  not and whether or not to mark true values
                  (if any) on the plot.
                  deafult is
                  {'plot':False,'true vals':None}
    
    p_vs_rho   :: Mentions whether or not to plot 
                  Pressure Density Credible intervals
                  (default is False)
    '''

    eos=polytrope_eos
    fig={'corner':None,'p_vs_rho':None}
    data=h5py.File(infile,'r')
    Samples = np.array(data.get('chains'))
    Ns=Samples.shape
    burn_in=int(Ns[0]/2.)
    samples=[]
    thinning=int(Ns[0]/50.)
    
    try:
        thinning=int(max(mc.autocorr.integrated_time(Samples))/2.)
    except mc.autocorr.AutocorrError as e:
        print(e)
    for i in range(burn_in,Ns[0],thinning):
        for j in range(Ns[1]):
            samples.append(Samples[i,j,:])

    samples=np.array(samples)
    
    if cornerplot['plot']:
                                                                 
        fig_corner=corner.corner(samples,labels=[r'$logP$',r'$\gamma_1$',r'$\gamma_2$',r'$\gamma_3$'],smooth=1.2,quantiles=[0.1,0.5,0.9],color='b',show_titles=True,title_kwargs={"fontsize": 12,"color":'b'},use_math_text=True,label='GWXtreme',hist_kwargs={"density":True})
        if(cornerplot['true vals'] is not None):
            ndim=4
            axes=np.array(fig_corner.axes).reshape((ndim,ndim))

            Tr=cornerplot['true vals']
    
            for i in range(ndim):
                ax=axes[i,i]
                ax.axvline(x=Tr[i],color='orange')


            for yi in range(ndim):
                for xi in range(yi):
                    ax=axes[yi,xi]
                    ax.axvline(x=Tr[xi],color='orange')
                    ax.axhline(y=Tr[yi],color='orange')
                    ax.plot(Tr[xi],Tr[yi],color='orange')
        fig['corner']=fig_corner
                    
    if(p_vs_rho['plot']):
        logp=[]
        rho=np.logspace(17.25,18.25,1000)
        

        for s in samples:
            params=(s[0], s[1], s[2], s[3])
            
            p=eos_p_of_rho(rho,eos(params))
            
            logp.append(p)

        logp=np.array(logp)
        logp_CIup=np.array([np.quantile(logp[:,i],0.95) for i in range(len(rho))])
        logp_CIlow=np.array([np.quantile(logp[:,i],0.05) for i in range(len(rho))])
        logp_med=np.array([np.quantile(logp[:,i],0.5) for i in range(len(rho))])
        fig_eos,ax_eos=plt.subplots(1,figsize=(12,12))
        ax_eos.errorbar(np.log10(rho),logp_med,color='cyan',
                    yerr=[logp_med-logp_CIlow,logp_CIup-logp_med],elinewidth=2.0,
                    capsize=1.5,ecolor='cyan',fmt='')
        ax_eos.set_xlabel(r'$\log10{\frac{\rho}{g cm^-3}}$',fontsize=20)
        ax_eos.set_ylabel(r'$log10(\frac{p}{dyne cm^{-2}})$',fontsize=20)
        if(p_vs_rho['true_eos'] is not None):
                if type(p_vs_rho['true_eos']) is tuple:
                    logp=eos_p_of_rho(rho,eos(p_vs_rho['true_eos']))
                    
                else:
                    logp=eos_p_of_rho(rho,lalsimulation.SimNeutronStarEOSByName(p_vs_rho['true_eos']))
                    
                ax_eos.plot(np.log10(rho),logp,color='black', linewidth=3.5)
                    
        fig['p_vs_rho']=(fig_eos,ax_eos)

    fig['corner'].savefig('{}corner4_O3.png'.format(outputDir))
    fig['p_vs_rho'][0].savefig('{}eos4_O3.png'.format(outputDir))
