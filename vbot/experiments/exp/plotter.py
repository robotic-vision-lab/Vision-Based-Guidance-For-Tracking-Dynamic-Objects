import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from .settings import *

class LOS1DataPlotter:
    def __init__(self, 
                 save_path, 
                 t, 
                 r1_t, 
                 r1_m, 
                 r1_e,
                 r2_t,
                 r2_m,
                 r2_e,
                 r3_t,
                 r3_m,
                 r3_e,
                 f1_r,
                 f2_r,
                 th1_t,
                 th1_m,
                 th1_e,
                 th2_t,
                 th2_m,
                 th2_e,
                 th3_t,
                 th3_m,
                 th3_e,
                 f1_th,
                 f2_th):

        self.save_path = save_path
        self.t = t
        self.r1_t = r1_t
        self.r1_m = r1_m
        self.r1_e = r1_e
        self.r2_t = r2_t
        self.r2_m = r2_m
        self.r2_e = r2_e
        self.r3_t = r3_t
        self.r3_m = r3_m
        self.r3_e = r3_e
        self.f1_r = f1_r
        self.f2_r = f2_r
        self.th1_t = th1_t
        self.th1_m = th1_m
        self.th1_e = th1_e
        self.th2_t = th2_t
        self.th2_m = th2_m
        self.th2_e = th2_e
        self.th3_t = th3_t
        self.th3_m = th3_m
        self.th3_e = th3_e
        self.f1_th = f1_th
        self.f2_th = f2_th


        self.window_title = 'LOS Kinematics - I'
        self.fig = None
        self.axs = None

        self.set_params()


    def set_params(self):
        # r1 params
        self.r1_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$r_{1}$')
        self.r1_m_params = dict(color='limegreen',  alpha=0.75,  ls='--', lw=1.5, label=r'$r_{1m}$')
        self.r1_e_params = dict(color='limegreen',  alpha=0.75, ls='-', lw=1, label=r'$\hat{r}_{1}$')
        
        # r2 params
        self.r2_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$r_{2}$')
        self.r2_m_params = dict(color='limegreen',  alpha=0.75,  ls='--', lw=1.5, label=r'$r_{2m}$')
        self.r2_e_params = dict(color='limegreen',  alpha=0.75, ls='-', lw=1, label=r'$\hat{r}_{2}$')
        
        # r3 params
        self.r3_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$r_{3}$')
        self.r3_m_params = dict(color='limegreen',  alpha=0.75,  ls='--', lw=1.5, label=r'$r_{3m}$')
        self.r3_e_params = dict(color='limegreen',  alpha=0.75, ls='-', lw=1, label=r'$\hat{r}_{3}$')
        
        # fp1r params
        self.f1_r_params = dict(color='crimson',  alpha=0.85, ls='-', lw=2.5, label=r'$\hat{r}_{fp1}$')
        self.f2_r_params = dict(color='crimson',  alpha=0.85, ls='-', lw=2.5, label=r'$\hat{r}_{fp2}$')

        # th1 params
        self.th1_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$\theta_{1}$')
        self.th1_m_params = dict(color='limegreen',  alpha=0.75,  ls='--', lw=1.5, label=r'$\theta_{1m}$')
        self.th1_e_params = dict(color='limegreen',  alpha=0.75, ls='-', lw=1, label=r'$\hat{\theta}_{1}$')
        
        # th2 params
        self.th2_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$\theta_{2}$')
        self.th2_m_params = dict(color='limegreen',  alpha=0.75,  ls='--', lw=1.5, label=r'$\theta_{2m}$')
        self.th2_e_params = dict(color='limegreen',  alpha=0.75, ls='-', lw=1, label=r'$\hat{\theta}_{2}$')
        
        # th3 params
        self.th3_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$\theta_{3}$')
        self.th3_m_params = dict(color='limegreen',  alpha=0.75,  ls='--', lw=1.5, label=r'$\theta_{3m}$')
        self.th3_e_params = dict(color='limegreen',  alpha=0.75, ls='-', lw=1, label=r'$\hat{\theta}_{3}$')
        
        # fp1r params
        self.f1_th_params = dict(color='crimson',  alpha=0.85, ls='-', lw=2.5, label=r'$\hat{\theta}_{fp1}$')
        self.f2_th_params = dict(color='crimson',  alpha=0.85, ls='-', lw=2.5, label=r'$\hat{\theta}_{fp2}$')

        # rcParams
        params = {'xtick.direction'     : 'in',
                  'xtick.top'           : True,
                  'xtick.minor.visible' : True,
                  'xtick.color'         : 'gray',
                  'ytick.direction'     : 'in',
                  'ytick.right'         : True,
                  'ytick.minor.visible' : True,
                  'ytick.color'         : 'gray',
                #   'text.usetex'         : True,           # slows rendering significantly
                #   'toolbar'             : 'None',         # with this none, zoom keymap 'o' does not work
                  'pdf.compression'     : 0,
                  'legend.fontsize'     : 'large',
                  'axes.labelsize'      : 'x-large',
                  'axes.titlesize'      : 'x-large',
                  'xtick.labelsize'     : 'large',
                  'ytick.labelsize'     : 'large',
                  'axes.edgecolor'      : 'gray'} 

        mpl.rcParams.update(params)
        
        


    def plot(self):
        self.fig, self.axs = plt.subplots(2,1, dpi=100, figsize=(10,10), sharex=True, gridspec_kw={'hspace': 0.25})
        # self.fig.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ I}$', fontsize=TITLE_FONT_SIZE)
        self.fig.canvas.manager.set_window_title(self.window_title)

        # r1, r2
        self.axs[0].plot(self.t, self.r1_t, **self.r1_t_params)
        self.axs[0].plot(self.t, self.r1_m, **self.r1_m_params)
        self.axs[0].plot(self.t, self.r1_e, **self.r1_e_params)

        self.axs[0].plot(self.t, self.r2_t, **self.r2_t_params)
        self.axs[0].plot(self.t, self.r2_m, **self.r2_m_params)
        self.axs[0].plot(self.t, self.r2_e, **self.r2_e_params)

        self.axs[0].plot(self.t, self.r3_t, **self.r3_t_params)
        self.axs[0].plot(self.t, self.r3_m, **self.r3_m_params)
        self.axs[0].plot(self.t, self.r3_e, **self.r3_e_params)

        self.axs[0].plot(self.t, self.f1_r, **self.f1_r_params)
        self.axs[0].plot(self.t, self.f2_r, **self.f2_r_params)
        

        # th1, th2
        self.axs[1].plot(self.t, self.th1_t, **self.th1_t_params)
        self.axs[1].plot(self.t, self.th1_m, **self.th1_m_params)
        self.axs[1].plot(self.t, self.th1_e, **self.th1_e_params)

        self.axs[1].plot(self.t, self.th2_t, **self.th2_t_params)
        self.axs[1].plot(self.t, self.th2_m, **self.th2_m_params)
        self.axs[1].plot(self.t, self.th2_e, **self.th2_e_params)

        self.axs[1].plot(self.t, self.th3_t, **self.th3_t_params)
        self.axs[1].plot(self.t, self.th3_m, **self.th3_m_params)
        self.axs[1].plot(self.t, self.th3_e, **self.th3_e_params)

        self.axs[1].plot(self.t, self.f1_th, **self.f1_th_params)
        self.axs[1].plot(self.t, self.f2_th, **self.f2_th_params)

        # set axes decorations
        self.add_axes_decor()

        # save and show figure
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.94, top=0.94)
        self.fig.savefig(f'{self.save_path}/1_los1.pdf')
        self.fig.show()

    def add_axes_decor(self):
        self.axs[0].set_title(r'$\mathbf{r}$')
        self.axs[0].legend(loc='upper right')
        self.axs[0].set(ylabel=r'$r\ (m)$')
        self.axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[0].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[0].grid(True, which='minor', alpha=0.1)
        self.axs[0].grid(True, which='major', alpha=0.3)
        [tl.set_color('black') for tl in self.axs[0].get_xticklabels()]
        [tl.set_color('black') for tl in self.axs[0].get_yticklabels()]
        
        self.axs[1].set_title(r'$\mathbf{\theta}$')
        self.axs[1].legend(loc='upper right')
        self.axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$\theta\ (^{\circ})$')
        self.axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[1].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[1].grid(True, which='minor', alpha=0.1)
        self.axs[1].grid(True, which='major', alpha=0.3)
        [tl.set_color('black') for tl in self.axs[1].get_xticklabels()]
        [tl.set_color('black') for tl in self.axs[1].get_yticklabels()]


    

    



    