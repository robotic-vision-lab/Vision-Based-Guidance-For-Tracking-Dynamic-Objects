import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from .settings import *

class LOS1DataPlotter:
    def __init__(self, save_path, t, r1_t, r1_m, r1_e, r2_t, r2_m, r2_e, th1_t, th1_m, th1_e, th2_t, th2_m, th2_e):
        self.save_path = save_path
        self.t = t
        self.r1_t = r1_t
        self.r1_m = r1_m
        self.r1_e = r1_e
        self.r2_t = r2_t
        self.r2_m = r2_m
        self.r2_e = r2_e
        self.th1_t = th1_t
        self.th1_m = th1_m
        self.th1_e = th1_e
        self.th2_t = th2_t
        self.th2_m = th2_m
        self.th2_e = th2_e

        self.window_title = 'LOS Kinematics - I'
        self.fig = None
        self.axs = None

        self.set_params()


    def set_params(self):
        # r1 params
        self.r1_t_params = dict(color='darkorchid',  alpha=0.7,  ls='-', lw=2,   label=r'$r_1$')
        self.r1_m_params = dict(color='forestgreen', alpha=0.85, ls='-', lw=1.5, label=r'$r_{1m}$')
        self.r1_e_params = dict(color='darkorange',  alpha=0.85, ls='-', lw=1.5, label=r'$\hat{r_1}$')
        
        # r2 params
        self.r2_t_params = dict(color='darkorchid',  alpha=0.7,  ls='-', lw=2,   label=r'$r_2$')
        self.r2_m_params = dict(color='forestgreen', alpha=0.85, ls='-', lw=1.5, label=r'$r_{2m}$')
        self.r2_e_params = dict(color='darkorange',  alpha=0.85, ls='-', lw=1.5, label=r'$\hat{r_2}$')

        # th1 params
        self.th1_t_params = dict(color='darkorchid',  alpha=0.7,  ls='-', lw=2,   label=r'$\theta_1$')
        self.th1_m_params = dict(color='forestgreen', alpha=0.85, ls='-', lw=1.5, label=r'$theta_{1m}$')
        self.th1_e_params = dict(color='darkorange',  alpha=0.85, ls='-', lw=1.5, label=r'$\hat{\theta_1}$')
        
        # th2 params
        self.th2_t_params = dict(color='darkorchid',  alpha=0.7,  ls='-', lw=2,   label=r'$\theta_2$')
        self.th2_m_params = dict(color='forestgreen', alpha=0.85, ls='-', lw=1.5, label=r'$\theta_{2m}$')
        self.th2_e_params = dict(color='darkorange',  alpha=0.85, ls='-', lw=1.5, label=r'$\hat{\theta_2}$')

        # rcParams
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['toolbar'] = 'None'
        mpl.rcParams['pdf.compression'] = 0
        
        


    def plot(self):
        self.fig, self.axs = plt.subplots(2,1, sharex=True, gridspec_kw={'hspace': 0.25})
        self.fig.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ I}$', fontsize=TITLE_FONT_SIZE)
        self.fig.canvas.manager.set_window_title(self.window_title)

        # r1, r2
        self.axs[0].plot(self.t, self.r1_t, **self.r1_t_params)
        self.axs[0].plot(self.t, self.r1_m, **self.r1_m_params)
        self.axs[0].plot(self.t, self.r1_e, **self.r1_e_params)

        self.axs[0].plot(self.t, self.r2_t, **self.r2_t_params)
        self.axs[0].plot(self.t, self.r2_m, **self.r2_m_params)
        self.axs[0].plot(self.t, self.r2_e, **self.r2_e_params)

        # th1, th2
        self.axs[1].plot(self.t, self.th1_t, **self.th1_t_params)
        self.axs[1].plot(self.t, self.th1_m, **self.th1_m_params)
        self.axs[1].plot(self.t, self.th1_e, **self.th1_e_params)

        self.axs[1].plot(self.t, self.th2_t, **self.th2_t_params)
        self.axs[1].plot(self.t, self.th2_m, **self.th2_m_params)
        self.axs[1].plot(self.t, self.th2_e, **self.th2_e_params)

        # set axes decorations
        self.add_axes_decor()

        # save and show figure
        self.fig.savefig(f'{self.save_path}/1_los1.pdf')
        self.fig.show()

    def add_axes_decor(self):
        self.axs[0].set_title(r'$\mathbf{r}$', fontsize=SUB_TITLE_FONT_SIZE)
        self.axs[0].legend(loc='upper right')
        self.axs[0].set(ylabel=r'$r\ (m)$')
        self.axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[0].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[0].grid(True, which='minor', alpha=0.1)
        self.axs[0].grid(True, which='major', alpha=0.3)
        
        self.axs[1].set_title(r'$\mathbf{\theta}$', fontsize=SUB_TITLE_FONT_SIZE)
        self.axs[1].legend(loc='upper right')
        self.axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$\theta\ (^{\circ})$')
        self.axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[1].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[1].grid(True, which='minor', alpha=0.1)
        self.axs[1].grid(True, which='major', alpha=0.3)


    

    



    