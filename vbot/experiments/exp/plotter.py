import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.lines import Line2D

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
        self.r1_m_params = dict(color='dodgerblue',  alpha=0.75,  ls='--', lw=1.5, label=r'$r_{1m}$')
        self.r1_e_params = dict(color='dodgerblue',  alpha=0.75, ls='-', lw=1, label=r'$\hat{r}_{1}$')
        
        # r2 params
        self.r2_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$r_{2}$')
        self.r2_m_params = dict(color='dodgerblue',  alpha=0.75,  ls='--', lw=1.5, label=r'$r_{2m}$')
        self.r2_e_params = dict(color='dodgerblue',  alpha=0.75, ls='-', lw=1, label=r'$\hat{r}_{2}$')
        
        # r3 params
        self.r3_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$r_{3}$')
        self.r3_m_params = dict(color='dodgerblue',  alpha=0.75,  ls='--', lw=1.5, label=r'$r_{3m}$')
        self.r3_e_params = dict(color='dodgerblue',  alpha=0.75, ls='-', lw=1, label=r'$\hat{r}_{3}$')
        
        # fp1r params
        self.f1_r_params = dict(color='crimson',  alpha=0.85, ls='-', lw=2.5, label=r'$\hat{r}_{fp1}$')
        self.f2_r_params = dict(color='crimson',  alpha=0.85, ls='-', lw=2.5, label=r'$\hat{r}_{fp2}$')

        # th1 params
        self.th1_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$\theta_{1}$')
        self.th1_m_params = dict(color='dodgerblue',  alpha=0.75,  ls='--', lw=1.5, label=r'$\theta_{1m}$')
        self.th1_e_params = dict(color='dodgerblue',  alpha=0.75, ls='-', lw=1, label=r'$\hat{\theta}_{1}$')
        
        # th2 params
        self.th2_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$\theta_{2}$')
        self.th2_m_params = dict(color='dodgerblue',  alpha=0.75,  ls='--', lw=1.5, label=r'$\theta_{2m}$')
        self.th2_e_params = dict(color='dodgerblue',  alpha=0.75, ls='-', lw=1, label=r'$\hat{\theta}_{2}$')
        
        # th3 params
        self.th3_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$\theta_{3}$')
        self.th3_m_params = dict(color='dodgerblue',  alpha=0.75,  ls='--', lw=1.5, label=r'$\theta_{3m}$')
        self.th3_e_params = dict(color='dodgerblue',  alpha=0.75, ls='-', lw=1, label=r'$\hat{\theta}_{3}$')
        
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
                  'legend.fontsize'     : 'xx-large',
                  'axes.labelsize'      : 'xx-large',
                  'axes.titlesize'      : 'xx-large',
                  'xtick.labelsize'     : 'x-large',
                  'ytick.labelsize'     : 'x-large',
                  'axes.edgecolor'      : 'gray'} 

        mpl.rcParams.update(params)
        
        
    def make_handles(self, params_list):
        return [Line2D([0], [0], **params) for params in params_list]


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

        # fp1_r, fp2_r
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

        # fp1_theta, fp2_theta
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
        # subplot 0
        legend_handles_r = self.make_handles([self.r1_t_params,
                                              self.r1_m_params,
                                              self.r1_e_params,
                                              self.f1_r_params])

        self.axs[0].set_title(r'$\mathbf{r}$')
        self.axs[0].legend(handles=legend_handles_r, 
                           labels=[r'$r_{B_{i}}$', r'$r_{B_{i},m}$', r'$\hat{r}_{B_{i}}$', r'$r_{fp_{i}}$'],
                           loc='upper right')
        self.axs[0].set(ylabel=r'$r\ (m)$')
        self.axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[0].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[0].grid(True, which='minor', alpha=0.1)
        self.axs[0].grid(True, which='major', alpha=0.3)
        [tl.set_color('black') for tl in self.axs[0].get_xticklabels()]
        [tl.set_color('black') for tl in self.axs[0].get_yticklabels()]
        
        # subplot 1
        legend_handles_th = self.make_handles([self.th1_t_params,
                                               self.th1_m_params,
                                               self.th1_e_params,
                                               self.f1_th_params])
        self.axs[1].set_title(r'$\mathbf{\theta}$')
        self.axs[1].legend(handles=legend_handles_th,
                           labels=[r'${\theta}_{B_{i}}$', r'${\theta}_{B_{i},m}$', r'$\hat{\theta}_{B_{i}}$', r'${\theta}_{fp_{i}}$'],
                           loc='upper right')
        self.axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$\theta\ (^{\circ})$')
        self.axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[1].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[1].grid(True, which='minor', alpha=0.1)
        self.axs[1].grid(True, which='major', alpha=0.3)
        [tl.set_color('black') for tl in self.axs[1].get_xticklabels()]
        [tl.set_color('black') for tl in self.axs[1].get_yticklabels()]


    

    
class LOS2DataPlotter:
    def __init__(self, 
                 save_path, 
                 t, 
                 vr1_t,  
                 vr1_e,
                 vr2_t,
                 vr2_e,
                 vr3_t,
                 vr3_e,
                 f1_vr,
                 f2_vr,
                 vth1_t,
                 vth1_e,
                 vth2_t,
                 vth2_e,
                 vth3_t,
                 vth3_e,
                 f1_vth,
                 f2_vth):

        self.save_path = save_path
        self.t = t
        self.vr1_t = vr1_t
        self.vr1_e = vr1_e
        self.vr2_t = vr2_t
        self.vr2_e = vr2_e
        self.vr3_t = vr3_t
        self.vr3_e = vr3_e
        self.f1_vr = f1_vr
        self.f2_vr = f2_vr
        self.vth1_t = vth1_t
        self.vth1_e = vth1_e
        self.vth2_t = vth2_t
        self.vth2_e = vth2_e
        self.vth3_t = vth3_t
        self.vth3_e = vth3_e
        self.f1_vth = f1_vth
        self.f2_vth = f2_vth


        self.window_title = 'LOS Kinematics - II'
        self.fig = None
        self.axs = None

        self.set_params()


    def set_params(self):
        # vr1 params
        self.vr1_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$V_{r_{1}}$')
        self.vr1_e_params = dict(color='dodgerblue',  alpha=0.75, ls='-', lw=1, label=r'$\hat{V}_{r_1}$')
        
        # vr2 params
        self.vr2_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$V_{r_{2}}$')
        self.vr2_e_params = dict(color='dodgerblue',  alpha=0.75, ls='-', lw=1, label=r'$\hat{V}_{r_2}$')
        
        # vr3 params
        self.vr3_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$V_{r_{3}}$')
        self.vr3_e_params = dict(color='dodgerblue',  alpha=0.75, ls='-', lw=1, label=r'$\hat{V}_{r_3}$')
        
        # fp1vr, fp2vr params
        self.f1_vr_params = dict(color='crimson',  alpha=0.85, ls='-', lw=2.5, label=r'$\hat{V}_{r_{fp1}}$')
        self.f2_vr_params = dict(color='crimson',  alpha=0.85, ls='-', lw=2.5, label=r'$\hat{V}_{r_{fp2}}$')

        # vth1 params
        self.vth1_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$V_{\theta_{1}}$')
        self.vth1_e_params = dict(color='dodgerblue',  alpha=0.75, ls='-', lw=1, label=r'$\hat{V}_{\theta_1}$')
        
        # vth2 params
        self.vth2_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$V_{\theta_{2}}$')
        self.vth2_e_params = dict(color='dodgerblue',  alpha=0.75, ls='-', lw=1, label=r'$\hat{V}_{\theta_2}$')
        
        # vth3 params
        self.vth3_t_params = dict(color='gray',       alpha=0.8,  ls=':', lw=2,   label=r'$V_{\theta_{3}}$')
        self.vth3_e_params = dict(color='dodgerblue',  alpha=0.75, ls='-', lw=1, label=r'$\hat{V}_{\theta_3}$')
        
        # fp1vtheta, fp2vtheta params
        self.f1_vth_params = dict(color='crimson',  alpha=0.85, ls='-', lw=2.5, label=r'$\hat{V}_{\theta_{fp1}}$')
        self.f2_vth_params = dict(color='crimson',  alpha=0.85, ls='-', lw=2.5, label=r'$\hat{V}_{\theta_{fp2}}$')

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
                  'legend.fontsize'     : 'xx-large',
                  'axes.labelsize'      : 'xx-large',
                  'axes.titlesize'      : 'xx-large',
                  'xtick.labelsize'     : 'x-large',
                  'ytick.labelsize'     : 'x-large',
                  'axes.edgecolor'      : 'gray'} 

        mpl.rcParams.update(params)
        
        
            
    def make_handles(self, params_list):
        return [Line2D([0], [0], **params) for params in params_list]


    def plot(self):
        self.fig, self.axs = plt.subplots(2,1, dpi=100, figsize=(10,10), sharex=True, gridspec_kw={'hspace': 0.25})
        # self.fig.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ I}$', fontsize=TITLE_FONT_SIZE)
        self.fig.canvas.manager.set_window_title(self.window_title)

        # vr1, vr2, vr3
        self.axs[0].plot(self.t, self.vr1_t, **self.vr1_t_params)
        self.axs[0].plot(self.t, self.vr1_e, **self.vr1_e_params)

        self.axs[0].plot(self.t, self.vr2_t, **self.vr2_t_params)
        self.axs[0].plot(self.t, self.vr2_e, **self.vr2_e_params)

        self.axs[0].plot(self.t, self.vr3_t, **self.vr3_t_params)
        self.axs[0].plot(self.t, self.vr3_e, **self.vr3_e_params)

        # fp1_vr, fp2_vr
        self.axs[0].plot(self.t, self.f1_vr, **self.f1_vr_params)
        self.axs[0].plot(self.t, self.f2_vr, **self.f2_vr_params)

        # vth1, vth2, vth3
        self.axs[1].plot(self.t, self.vth1_t, **self.vth1_t_params)
        self.axs[1].plot(self.t, self.vth1_e, **self.vth1_e_params)

        self.axs[1].plot(self.t, self.vth2_t, **self.vth2_t_params)
        self.axs[1].plot(self.t, self.vth2_e, **self.vth2_e_params)

        self.axs[1].plot(self.t, self.vth3_t, **self.vth3_t_params)
        self.axs[1].plot(self.t, self.vth3_e, **self.vth3_e_params)

        # fp1_vtheta, fp2_vtheta
        self.axs[1].plot(self.t, self.f1_vth, **self.f1_vth_params)
        self.axs[1].plot(self.t, self.f2_vth, **self.f2_vth_params)

        # set axes decorations
        self.add_axes_decor()

        # save and show figure
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.94, top=0.94)
        self.fig.savefig(f'{self.save_path}/1_los2.pdf')
        self.fig.show()


    def add_axes_decor(self):
        # subplot 0
        legend_handles_vr = self.make_handles([self.vr1_t_params, 
                                               self.vr1_e_params,
                                               self.f1_vr_params])
        self.axs[0].set_title(r'$\mathbf{V_{r}}$')
        self.axs[0].legend(handles=legend_handles_vr,
                           labels=[r'$V_{rB_{i}}$', r'$V_{rB_{i},m}$', r'$\hat{V}_{rB_{i}}$', r'$V_{rfp_{i}}$'],
                           loc='upper right')
        self.axs[0].set(ylabel=r'$V_{r}\ (\frac{m}{s})$')
        self.axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[0].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[0].grid(True, which='minor', alpha=0.1)
        self.axs[0].grid(True, which='major', alpha=0.3)
        [tl.set_color('black') for tl in self.axs[0].get_xticklabels()]
        [tl.set_color('black') for tl in self.axs[0].get_yticklabels()]
        
        # subplot 1
        legend_handles_vth = self.make_handles([self.vth1_t_params,
                                                self.vth1_e_params,
                                                self.f1_vth_params])
        self.axs[1].set_title(r'$\mathbf{V_{\theta}}$')
        self.axs[1].legend(handles=legend_handles_vth,
                           labels=[r'$V_{\theta B_{i}}$', r'$V_{\theta B_{i},m}$', r'$\hat{V}_{\theta B_{i}}$', r'$V_{\theta fp_{i}}$'],
                           loc='upper right')
        self.axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$V_{\theta}\ (\frac{m}{s})$')
        self.axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[1].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[1].grid(True, which='minor', alpha=0.1)
        self.axs[1].grid(True, which='major', alpha=0.3)
        [tl.set_color('black') for tl in self.axs[1].get_xticklabels()]
        [tl.set_color('black') for tl in self.axs[1].get_yticklabels()]


    

    
class AccelerationCommandDataPlotter:
    def __init__(self, 
                 save_path, 
                 t, 
                 a_lat,
                 a_lng,
                 a_z):

        self.save_path = save_path
        self.t = t
        self.a_lat = a_lat
        self.a_lng = a_lng
        self.a_z = a_z

        self.window_title = 'Commanded Accelerations'
        self.fig = None
        self.axs = None

        self.set_params()


    def set_params(self):
        # params
        self.a_lat_params = dict(color='forestgreen', alpha=0.7,  ls='-', lw=2,   label=r'$a_{lat}$')
        self.a_lng_params = dict(color='deeppink', alpha=0.7,  ls='-', lw=2,   label=r'$a_{long}$')
        self.a_z_params = dict(color='royalblue', alpha=0.7,  ls='-', lw=2,   label=r'$a_{z}$')

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
                  'legend.fontsize'     : 'xx-large',
                  'axes.labelsize'      : 'xx-large',
                  'axes.titlesize'      : 'xx-large',
                  'xtick.labelsize'     : 'x-large',
                  'ytick.labelsize'     : 'x-large',
                  'axes.edgecolor'      : 'gray'} 

        mpl.rcParams.update(params)
        
            
    def make_handles(self, params_list):
        return [Line2D([0], [0], **params) for params in params_list]

    def plot(self):
        self.fig, self.axs = plt.subplots(dpi=100, figsize=(10,5))
        # self.fig.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ I}$', fontsize=TITLE_FONT_SIZE)
        self.fig.canvas.manager.set_window_title(self.window_title)

        # a_lat, a_long, a_z
        self.axs.plot(self.t, self.a_lat, **self.a_lat_params)
        self.axs.plot(self.t, self.a_lng, **self.a_lng_params)
        self.axs.plot(self.t, self.a_z, **self.a_z_params)

        # set axes decorations
        self.add_axes_decor()

        # save and show figure
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.94, top=0.94)
        self.fig.savefig(f'{self.save_path}/2_accel.pdf')
        self.fig.show()


    def add_axes_decor(self):
        self.axs.set_title(r'$a_{lat}, a_{long}, a_{z}$')
        self.axs.legend()
        self.axs.set(xlabel=r'$time\ (s)$', ylabel=r'$acceleration\ (\frac{m}{s_{2}})$')
        self.axs.xaxis.set_minor_locator(AutoMinorLocator())
        self.axs.yaxis.set_minor_locator(AutoMinorLocator())
        self.axs.grid(True, which='minor', alpha=0.1)
        self.axs.grid(True, which='major', alpha=0.3)
        [tl.set_color('black') for tl in self.axs.get_xticklabels()]
        [tl.set_color('black') for tl in self.axs.get_yticklabels()]
        


    
class ObjectiveFunctionDataPlotter:
    def __init__(self, 
                 save_path, 
                 t, 
                 y1,
                 y2):

        self.save_path = save_path
        self.t = t
        self.y1 = y1
        self.y2 = y2
        self.y1d = [K_W for _ in t]
        self.y2d = [0.0 for _ in t]

        self.window_title = 'Objective functions'
        self.fig = None
        self.axs = None

        self.set_params()


    def set_params(self):
        # params
        self.y1_params = dict(color='royalblue', alpha=0.8,  ls='-', lw=2,   label=r'$\hat{y}_{1}$')
        self.y2_params = dict(color='royalblue', alpha=0.8,  ls='-', lw=2,   label=r'$\hat{y}_{2}$')
        self.y1d_params = dict(color='darkorange', alpha=0.8,  ls='--', lw=2,   label=r'$y_{1d}$')
        self.y2d_params = dict(color='darkorange', alpha=0.8,  ls='--', lw=2,   label=r'$y_{2d}$')

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
        

                
    def make_handles(self, params_list):
        return [Line2D([0], [0], **params) for params in params_list]
    
    def plot(self):
        self.fig, self.axs = plt.subplots(2,1, dpi=100, figsize=(10,10), sharex=True, gridspec_kw={'hspace': 0.25})
        # self.fig.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ I}$', fontsize=TITLE_FONT_SIZE)
        self.fig.canvas.manager.set_window_title(self.window_title)

        # y1, y2
        self.axs[0].plot(self.t, self.y1, **self.y1_params)
        self.axs[0].plot(self.t, self.y1d, **self.y1d_params)
        self.axs[1].plot(self.t, self.y2, **self.y2_params)
        self.axs[1].plot(self.t, self.y2d, **self.y2d_params)

        # set axes decorations
        self.add_axes_decor()

        # save and show figure
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.94, top=0.94)
        self.fig.savefig(f'{self.save_path}/3_objfunc.pdf')
        self.fig.show()


    def add_axes_decor(self):
        self.axs[0].set_title(r'$y_{1}$')
        self.axs[0].legend()
        self.axs[0].set(ylabel=r'$y_1$')
        self.axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[0].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[0].grid(True, which='minor', alpha=0.1)
        self.axs[0].grid(True, which='major', alpha=0.3)
        [tl.set_color('black') for tl in self.axs[0].get_xticklabels()]
        [tl.set_color('black') for tl in self.axs[0].get_yticklabels()]
        
        self.axs[1].set_title(r'$y_{2}$')
        self.axs[1].legend()
        self.axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$y_2$')
        self.axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[1].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[1].grid(True, which='minor', alpha=0.1)
        self.axs[1].grid(True, which='major', alpha=0.3)
        [tl.set_color('black') for tl in self.axs[1].get_xticklabels()]
        [tl.set_color('black') for tl in self.axs[1].get_yticklabels()]
        


class SpeedsHeadingsDataPlotter:
    def __init__(self, 
                 save_path, 
                 t, 
                 t1_s,
                 t2_s,
                 t3_s,
                 fp1_s,
                 fp2_s,
                 d_s,
                 t1_h,
                 t2_h,
                 t3_h,
                 fp1_h,
                 fp2_h,
                 d_h):

        self.save_path = save_path
        self.t = t
        self.t1_s = t1_s
        self.t2_s = t2_s
        self.t3_s = t3_s
        self.fp1_s = fp1_s
        self.fp2_s = fp2_s
        self.d_s = d_s
        self.t1_h = t1_h
        self.t2_h = t2_h
        self.t3_h = t3_h
        self.fp1_h = fp1_h
        self.fp2_h = fp2_h
        self.d_h = d_h


        self.window_title = 'Speeds'
        self.fig = None
        self.axs = None

        self.set_params()


    def set_params(self):
        # target speed params
        self.t1_s_params = dict(color='gray', alpha=0.8,  ls='-', lw=1.5,   label=r'$\vert V_{B_{1}} \vert$')
        self.t2_s_params = dict(color='gray', alpha=0.8,  ls='-', lw=1.5,   label=r'$\vert V_{B_{2}} \vert$')
        self.t3_s_params = dict(color='gray', alpha=0.8,  ls='-', lw=1.5,   label=r'$\vert V_{B_{3}} \vert$')

        # fp speed params
        self.fp1_s_params = dict(color='dodgerblue', alpha=0.8,  ls='-', lw=2,   label=r'$\vert V_{B_{fp1}} \vert$')
        self.fp2_s_params = dict(color='dodgerblue', alpha=0.8,  ls='-', lw=2,   label=r'$\vert V_{B_{fp2}} \vert$')

        # drone speed params
        self.d_s_params = dict(color='orangered', alpha=0.8,  ls='-', lw=2,   label=r'$\vert V_{A} \vert$')


        # target speed params
        self.t1_h_params = dict(color='gray', alpha=0.8,  ls='-', lw=1.5,   label=r'$\angle V_{B_{1}}$')
        self.t2_h_params = dict(color='gray', alpha=0.8,  ls='-', lw=1.5,   label=r'$\angle V_{B_{2}}$')
        self.t3_h_params = dict(color='gray', alpha=0.8,  ls='-', lw=1.5,   label=r'$\angle V_{B_{3}}$')

        # fp speed params
        self.fp1_h_params = dict(color='dodgerblue', alpha=0.8,  ls='-', lw=2,   label=r'$\angle V_{B_{fp1}}$')
        self.fp2_h_params = dict(color='dodgerblue', alpha=0.8,  ls='-', lw=2,   label=r'$\angle V_{B_{fp2}}$')
        
        # drone heading params
        self.d_h_params = dict(color='orangered', alpha=0.8,  ls='-', lw=2,   label=r'$\angle V_{A}$')

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
        

                
    def make_handles(self, params_list):
        return [Line2D([0], [0], **params) for params in params_list]


    def plot(self):
        self.fig, self.axs = plt.subplots(2,1, dpi=100, figsize=(10,10), sharex=True, gridspec_kw={'hspace': 0.25})
        # self.fig.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ I}$', fontsize=TITLE_FONT_SIZE)
        self.fig.canvas.manager.set_window_title(self.window_title)

        # targets speeds
        self.axs[0].plot(self.t, self.t1_s, **self.t1_s_params)
        self.axs[0].plot(self.t, self.t2_s, **self.t2_s_params)
        self.axs[0].plot(self.t, self.t3_s, **self.t3_s_params)

        # focal points speeds
        self.axs[0].plot(self.t, self.fp1_s, **self.fp1_s_params)
        self.axs[0].plot(self.t, self.fp2_s, **self.fp2_s_params)

        # drone speed
        self.axs[0].plot(self.t, self.d_s, **self.d_s_params)

        # targets speeds
        self.axs[1].plot(self.t, self.t1_h, **self.t1_h_params)
        self.axs[1].plot(self.t, self.t2_h, **self.t2_h_params)
        self.axs[1].plot(self.t, self.t3_h, **self.t3_h_params)

        # focal points speeds
        self.axs[1].plot(self.t, self.fp1_h, **self.fp1_h_params)
        self.axs[1].plot(self.t, self.fp2_h, **self.fp2_h_params)

        # drone speed
        self.axs[1].plot(self.t, self.d_h, **self.d_h_params)


        # set axes decorations
        self.add_axes_decor()

        # save and show figure
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.94, top=0.94)
        self.fig.savefig(f'{self.save_path}/4_speeds_headings.pdf')
        self.fig.show()


    def add_axes_decor(self):
        legends_handles_s = self.make_handles([self.t1_s_params, self.fp1_s_params, self.d_s_params])
        self.axs[0].set_title(r'speeds')
        self.axs[0].legend(handles=legends_handles_s,
                           labels=[r'\vert V_{B_{i}} \vert', r'$\vert V_{B_{fpi}} \vert$', r'\vert V_{A} \vert'])
        self.axs[0].set(ylabel=r'$\vert V \vert\ (\frac{m}{s})$')
        self.axs[0].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[0].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[0].grid(True, which='minor', alpha=0.1)
        self.axs[0].grid(True, which='major', alpha=0.3)
        [tl.set_color('black') for tl in self.axs[0].get_xticklabels()]
        [tl.set_color('black') for tl in self.axs[0].get_yticklabels()]
        
        legends_handles_h = self.make_handles([self.t1_h_params, self.fp1_h_params, self.d_h_params])
        self.axs[1].set_title(r'headings')
        self.axs[1].legend(handles=legends_handles_s,
                           labels=[r'\angle V_{B_{i}}', r'$\angle V_{B_{fpi}} $', r'\angle V_{A} '])
        self.axs[1].set(xlabel=r'$time\ (s)$', ylabel=r'$\angle V\ (^{\circ})$')
        self.axs[1].xaxis.set_minor_locator(AutoMinorLocator())
        self.axs[1].yaxis.set_minor_locator(AutoMinorLocator())
        self.axs[1].grid(True, which='minor', alpha=0.1)
        self.axs[1].grid(True, which='major', alpha=0.3)
        [tl.set_color('black') for tl in self.axs[1].get_xticklabels()]
        [tl.set_color('black') for tl in self.axs[1].get_yticklabels()]
        

class TrajectoryWorldDataPlotter:
    def __init__(self, 
                 save_path, 
                 t, 
                 t1_x,
                 t1_y,
                 t2_x,
                 t2_y,
                 t3_x,
                 t3_y,
                 fp1_x,
                 fp1_y,
                 fp2_x,
                 fp2_y,
                 d_x,
                 d_y
                 ):

        self.save_path = save_path
        self.t = t
        self.t1_x = t1_x
        self.t1_y = t1_y
        self.t2_x = t2_x
        self.t2_y = t2_y
        self.t3_x = t3_x
        self.t3_y = t3_y
        self.fp1_x = fp1_x
        self.fp1_y = fp1_y
        self.fp2_x = fp2_x
        self.fp2_y = fp2_y
        self.d_x = d_x
        self.d_y = d_y

        self.window_title = 'Trajectory (World)'
        self.fig = None
        self.axs = None

        self.set_params()


    def set_params(self):
        # target traj params
        self.t1_params = dict(color='gray', alpha=0.7,  ls='-', lw=1,   label=r'$B_{1}$')
        self.t2_params = dict(color='gray', alpha=0.7,  ls='-', lw=1,   label=r'$B_{2}$')
        self.t3_params = dict(color='gray', alpha=0.7,  ls='-', lw=1,   label=r'$B_{3}$')

        # focal point traj params
        self.fp1_params = dict(color='dodgerblue', alpha=0.8,  ls='-', lw=2,   label=r'$B_{fp1}$')
        self.fp2_params = dict(color='dodgerblue', alpha=0.8,  ls='-', lw=2,   label=r'$B_{fp2}$')

        # drone params
        self.d_params = dict(color='orangered', alpha=0.8,  ls='-', lw=2.5,   label=r'$A$')
        


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


    def make_handles(self, params_list):
        return [Line2D([0], [0], **params) for params in params_list]


    def plot(self):
        self.fig, self.axs = plt.subplots(dpi=100, figsize=(10,5))
        # self.fig.suptitle(r'$\mathbf{Line\ of\ Sight\ Kinematics\ -\ I}$', fontsize=TITLE_FONT_SIZE)
        self.fig.canvas.manager.set_window_title(self.window_title)

        # targets speeds
        self.axs.plot(self.t1_x, self.t1_y, **self.t1_params)
        self.axs.plot(self.t2_x, self.t2_y, **self.t2_params)
        self.axs.plot(self.t3_x, self.t3_y, **self.t3_params)

        # focal points speeds
        self.axs.plot(self.fp1_x, self.fp1_y, **self.fp1_params)
        self.axs.plot(self.fp2_x, self.fp2_y, **self.fp2_params)

        # drone speed
        self.axs.plot(self.d_x, self.d_y, **self.d_params)

        # set axes decorations
        self.add_axes_decor()

        # save and show figure
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.94, top=0.94)
        self.fig.savefig(f'{self.save_path}/5_traj_world.pdf')
        self.fig.show()


    def add_axes_decor(self):
        legend_handles = self.make_handles([self.t1_params, self.fp1_params, self.d_params])
        self.axs.set_title(r'Trajectories (world frame)')
        self.axs.legend(handles=legend_handles,
                        labels=[r'$B_{i}$', r'$B_{fpi}$', r'$A$'])
        self.axs.set(xlabel=r'$x\ (m)$', ylabel=r'$y\ (m)$')
        self.axs.xaxis.set_minor_locator(AutoMinorLocator())
        self.axs.yaxis.set_minor_locator(AutoMinorLocator())
        self.axs.grid(True, which='minor', alpha=0.1)
        self.axs.grid(True, which='major', alpha=0.3)
        [tl.set_color('black') for tl in self.axs.get_xticklabels()]
        [tl.set_color('black') for tl in self.axs.get_yticklabels()]
        
        

    