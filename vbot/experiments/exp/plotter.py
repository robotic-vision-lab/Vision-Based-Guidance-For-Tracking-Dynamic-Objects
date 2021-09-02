import numpy as np
import matplotlib.pyplot as plt
from .settings import *

class LOS1DataPlotter:
    def __init__(self, t, r1_t, r1_m, r1_e, r2_t, r2_m, r2_e, th1_t, th1_m, th1_e, th2_t, th2_m, th2_e):
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

        self.set_params()

    def set_params(self):
        # r1 params
        self.r1_t_params = dict(color='dimgray',     alpha=0.85, ls='-', lw=2,   label=r'$r$')
        self.r1_m_params = dict(color='forestgreen', alpha=0.85, ls='-', lw=1.5, label=r'$r_m$')
        self.r1_e_params = dict(color='darkorange',  alpha=0.85, ls='-', lw=1.5, label=r'$\hat{r}$')
        
        # r2 params
        self.r2_t_params = dict(color='dimgray',     alpha=0.85, ls='-', lw=2,   label=r'$r$')
        self.r2_m_params = dict(color='forestgreen', alpha=0.85, ls='-', lw=1.5, label=r'$r_m$')
        self.r2_e_params = dict(color='darkorange',  alpha=0.85, ls='-', lw=1.5, label=r'$\hat{r}$')

        # th1 params
        self.th1_t_params = dict(color='dimgray',     alpha=0.85, ls='-', lw=2,   label=r'$r$')
        self.th1_m_params = dict(color='forestgreen', alpha=0.85, ls='-', lw=1.5, label=r'$r_m$')
        self.th1_e_params = dict(color='darkorange',  alpha=0.85, ls='-', lw=1.5, label=r'$\hat{r}$')
        
        # th2 params
        self.th2_t_params = dict(color='dimgray',     alpha=0.85, ls='-', lw=2,   label=r'$r$')
        self.th2_m_params = dict(color='forestgreen', alpha=0.85, ls='-', lw=1.5, label=r'$r_m$')
        self.th2_e_params = dict(color='darkorange',  alpha=0.85, ls='-', lw=1.5, label=r'$\hat{r}$')
        


    def plot(self):
        fig, axs = plt.subplots(2,1, sharex=True, gridspec_kw={'hspace': 0.25})
        fig.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ I}$', fontsize=TITLE_FONT_SIZE)
        fig.canvas.manager.set_window_title(self.window_title)

        axs[0].plot(t, )


    

    



    