import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal

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

def plot_wavelet_spectral(url, detrend, scale_min, scale_max, wavelet_name):
#     url = "http://paos.colorado.edu/research/wavelets/wave_idl/sst_nino3.dat"
#     dat = np.genfromtxt(url)

#     df = pd.read_table(url, header=None)
    dat = np.loadtxt(url)
#     dat = np.loadtxt('../data/sst_nino3.txt')
    ## detrend
    # variance = dat.std()**2
    # dat = (dat - np.mean(dat))/np.sqrt(variance)
    if detrend==True:
        dat_value = signal.detrend(dat[:,1])
    else:
        dat_value = dat[:,1]

    N = dat.shape[0]
#     t0=1871
#     dt=0.25
#     time = np.arange(0, N) * dt + t0
    dt = dat[1,0] - dat[0,0]
    time = dat[:,0]

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

    coef, freq = pywt.cwt(dat_value, scales, wavelet_name, dt)
    power = (np.abs(coef))**2

    # # normalize by scale 
    # power = power/(scales[:, None])

    period = 1./freq

    # get global wavelet power
    var = dat_value.std()**2
    g_power = var * power.mean(axis = 1)

##-----------plot wavelet spectrum-----------------##
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
    ax1.plot(time, dat_value, 'k-', lw = 0.5)
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
        url = Text(value='../data/sst_nino3.txt',placeholder='Type something',description='File path:',disabled=False),
        detrend = Checkbox(value=False, description='Detrend',disabled=False),
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

# ------------ geo visualization ------------------
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import imageio
import pathlib
import mapclassify as mc
import shapely

def geospatial_viz(geo_data_url, 
                   point_data_url=None, 
                   att_var=None, 
                   map_type=None):
    
    '''
    function to visualize the attribute information in map. (eg, population in states)
    geo_att_data: geodataframe that contains both geometry and attributes info
    att_var: the attributes to be visualized in the map
    map_type: string, the type of map to be viz. pointplot, choropleth, voronoi
    
    if point_data = None, att_var must be from geo_data
    
    '''
    geo_data = gpd.read_file(geo_data_url)
    print(geo_data.head())
    

    if point_data_url == 'No point attribute data':
        if att_var is None:
            ax = gplt.polyplot(geo_data, figsize = (10,5))
            ax.set_title('plain map of continental USA', fontsize = 16)
        else:
            if map_type == 'choropleth':
                scheme = mc.FisherJenks(geo_data[att_var], k = 5)
                labels = scheme.get_legend_classes()
                ax = gplt.polyplot(geo_data, projection = gcrs.AlbersEqualArea())
                gplt.choropleth(geo_data,
                                hue = att_var,
                                edgecolor = 'white',
                                linewidth = 1,
                                cmap = 'Blues',
                                legend = True,
                                scheme = scheme,
                                legend_labels = labels,
                                ax = ax)
                ax.set_title('{} in the continental US'.format(att_var), 
                             fontsize = 16)
                
            if map_type == "cartogram":
                gplt.cartogram(geo_data,
                               scale = att_var,
                               edgecolor = 'black',
                               projection = gcrs.AlbersEqualArea())
                
                
    else:
        point_data = gpd.read_file(point_data_url)
        scheme = mc.Quantiles(point_data[att_var], k = 5)
        labels = scheme.get_legend_classes()
        
        if map_type == 'pointplot':
            if isinstance(point_data.geometry[0],shapely.geometry.point.Point):
                ax = gplt.polyplot(geo_data,
                                   edgecolor = 'white',
                                   facecolor = 'lightgray',
                                   figsize = (12, 8)
                                   #projection = gcrs.AlbersEqualArea()
                                   )
                gplt.pointplot(point_data,
                               ax = ax,
                               hue = att_var,
                               cmap = 'Blues',
                               scheme = scheme,
                               scale = att_var,
                               legend = True,
                               legend_var = 'scale',
                               legend_kwargs = {"loc": 'lower right'},
                               legend_labels = labels)
                ax.set_title('Cities in the continental US, by population 2010', 
                             fontsize=16)
            else:
                print('Geometry data type not valid')
        
        if map_type == "voronoi":
            # check uniqueness of coordinates
            duplicates = point_data.geometry.duplicated()
            point_data_unique = point_data[-duplicates]
            proj = gplt.crs.AlbersEqualArea(central_longitude=-98,
                               central_latitude=39.5)
            
            ax = gplt.voronoi(point_data_unique, 
                 hue = att_var,
                 clip = geo_data,
                 projection = proj,
                 cmap = 'Blues',
                 legend = True,
                 edgecolor = "white",
                 linewidth = 0.01)
            
            gplt.polyplot(geo_data,
                          ax = ax,
                          extent = geo_data.total_bounds,
                          edgecolor = "black",
                          linewidth = 1,
                          zorder = 1)
            plt.title("{} in US cities".format(att_var), fontsize = 16)
            

            

def viz_app():
    app = widgetify(
            geospatial_viz,
           geo_data_url = Text(value = 'https://raw.githubusercontent.com/ResidentMario/geoplot-data/master/contiguous-usa.geojson',
                        placeholder = 'Type something', 
                        description = 'geo data path1:', 
                        disabled = False),
            point_data_url = Text(value = 'No point attribute data',
                        placeholder = 'Type something', 
                        description = 'geo data path2:', 
                        disabled = False),
            att_var = Text(value = 'population', 
                           placeholder = 'Type something',
                           description = 'attribute variable',
                           disabled = False),
            map_type = Dropdown(
                    options = ['choropleth', 'cartogram', 'pointplot', 'voronoi'],
                    value = 'choropleth',
                    description = 'map type',
                    disabled = False)
    )
    return app
            
