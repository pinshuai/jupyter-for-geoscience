import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import ipywidgets
from ipywidgets import *


# from __future__ import print_function
# from __future__ import absolute_import

class MyApp(ipywidgets.Box):
    def __init__(self, widgets, kwargs):
        self._kwargs = kwargs
        self._widgets = widgets
        super(MyApp, self).__init__(widgets)
        self.layout.display = "flex"
        self.layout.flex_flow = "column"
        self.layout.align_items = "stretch"

    @property
    def kwargs(self):
        return dict(
            [
                (key, val.value)
                for key, val in self._kwargs.items()
                if isinstance(val, (ipywidgets.widget.Widget, ipywidgets.fixed))
            ]
        )


def widgetify(fun, layout=None, manual=False, **kwargs):

    f = fun

    if manual:
        app = ipywidgets.interact_manual(f, **kwargs)
        app = app.widget
    else:
        app = ipywidgets.interactive(f, **kwargs)

    # if layout is None:
    # TODO: add support for changing layouts
    w = MyApp(app.children, kwargs)

    f.widget = w
    # defaults =  #dict([(key, val.value) for key, val in kwargs.iteritems() if isinstance(val, Widget)])
    app.update()
    # app.on_displayed(f(**(w.kwargs)))

    return w

def plot_wavelet_spectral(scale_min, scale_max, wavelet_name):
    url = "http://paos.colorado.edu/research/wavelets/wave_idl/sst_nino3.dat"
    # df_nino = pd.read_table(dataset, header=None)
    dat = np.genfromtxt(url)

    ## detrend
    # variance = dat.std()**2
    # dat = (dat - np.mean(dat))/np.sqrt(variance)

    N = dat.shape[0]
    t0=1871
    dt=0.25
    time = np.arange(0, N) * dt + t0

    dj = 1/16 # the spacing between discrete scales. Default is 0.25. A smaller # will give better scale resolution, but be slower to plot.
#     s0 = 2*dt
#     scale_max = 128
#     J = int(np.log2(scale_max/s0)/dj)
    J = int(np.log2(scale_max/scale_min)/dj)
    scales = scale_min*2**(dj*np.arange(J + 1)) # from 0.5 to 64 yr

    # datetime = pd.date_range(start='1871-01-01', end='1997-01-01', freq='3MS', closed='left')

    # data = np.column_stack([time, dat])

    # np.savetxt('./sst_nino3.txt', data)
    if wavelet_name == 'cmor':
        wavelet_name = 'cmor1.5-1.0'

    coef, freq = pywt.cwt(dat, scales, wavelet_name, dt)
    power = (np.abs(coef))**2

    # # normalize by scale 
    # power = power/(scales[:, None])

    period = 1./freq

    # get global wavelet power
    var = dat.std()**2
    g_power = var * power.mean(axis = 1)

    # levels = [0.0625, 0.125, 0.25, 0.5, 1, 2]
    min_level = -4
    max_level = np.ceil(np.max(np.log2(power)))
    levels = 2**np.arange(min_level, max_level + 1)

    fig = plt.figure()

    gs = gridspec.GridSpec(2, 2, height_ratios = [1,3], width_ratios=[4, 1])
    gs.update(left=0.1,right=0.9,top=0.9,bottom=0.1, hspace = 0.2, wspace = 0.1) # adjust vertical spacing b/w subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax3 = fig.add_subplot(gs[3], sharey=ax2)

    # #------- plot time series
    ax1.plot(time, dat, 'k-', lw = 0.5)
    ax1.set_ylabel('Temperature ($\circ$C)', fontsize = 12)
    ax1.set_title('(a) Nino3 sea surface temperature')
    #------- plot wavelet spectrum
    # cf = ax2.contourf(stage.index, np.log2(period), np.log2(power), np.log2(levels),
    #             cmap = cm.jet,  extend='both')

    cf = ax2.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
        cmap = cm.viridis,  extend='both')

    divider = make_axes_locatable(ax2)
    cax = divider.new_vertical(size="5%", pad=0.4, pack_start=True)
    fig.add_axes(cax)
    cb = plt.colorbar(cf, cax=cax, orientation='horizontal')
    cb.ax.set_xlabel("Spectrum power (log2)",labelpad=0.5)

    # ax.set_title('Wavelet Power Spectrum')
    ax2.set_ylabel('Period (year)', fontsize = 12)
    Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                    np.ceil(np.log2(period.max())) +1)
    ax2.set_yticks(np.log2(Yticks))
    ax2.set_yticklabels(Yticks)
    ax2.invert_yaxis()
    ylim = ax2.get_ylim()
    ax2.set_ylim(ylim[0], -1)

    ax2.set_title('(b) Wavelet power spectrum')


    #--------- plot global spectrum

    ax3.plot(g_power, np.log2(period), 'r-', lw = 0.5)
    ax3.set_xlabel('Power')
    ax3.set_title('(c) Global wavelet\n spectrum')
    # add a divider so that the stratigraphy plot aligns horizontally with heat map
    divider2 = make_axes_locatable(ax3)
    cax2 = divider2.new_vertical(size="5%", pad=0.4, pack_start=True)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    # ax3.get_yaxis().set_ticks([]) # no y-ticks

    fig.set_size_inches(8, 6)


def wavelet_app():
    app = widgetify(
        plot_wavelet_spectral,
        scale_min = FloatSlider(value=0.5,min=0.5, max = 10, step=0.5, continuous_update = False, description='Scale_min:',layout=Layout(width='auto', height='auto')),
        scale_max = FloatSlider(value=128,min=2, max = 500, step=2, continuous_update = False, description='Scale_max:',layout=Layout(width='auto', height='auto')),
        wavelet_name = Dropdown(
            options=['morl','cmor'],
            value='morl',
            description='Wavelet family:',
            disabled=False,
        )

        
    )
    return app

