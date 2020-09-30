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

# for geospatial visualization
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import imageio
import pathlib
import mapclassify as mc
import shapely

# for PCA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from IPython.display import display
import seaborn as sns


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
   
# ----------------------- function and app for PCA ---------------------------------
def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(zip(a, b, c), index=names, columns=columns)
    if out:
        print("Importance of components:")
        display(summary)
    return summary


def loading_display(pca, n_components):
    for i in range(n_components):
        print('loadings for PC{}:'.format(i+1), pca.components_[i])
    

def calcpc(standardizedFeatures, loadings):
    """
    calculate the values for each principle component.
    find the number of samples in the data set and the number of variables
    """ 
    variables = standardizedFeatures
    if isinstance(variables, np.ndarray):
        variables = pd.DataFrame(variables)
    
    numsamples, numvariables = variables.shape
    # make a vector to store the component
    pc = np.zeros(numsamples)
    # calculate the value of the component for each sample
    for i in range(numsamples):
        valuei = 0
        for j in range(numvariables):
            valueij = variables.iloc[i, j]
            loadingj = loadings[j]
            valuei = valuei + (valueij * loadingj)
        pc[i] = valuei
    
    return pc

def pc_display(standardizedFeatures, component_to_show, pca):
    
    standardizedFeatures = standardizedFeatures
    numsamples = standardizedFeatures.shape[0]
    loadings = pca.components_[component_to_show-1]
    pc = calcpc(standardizedFeatures, loadings)
    
    pc = pd.DataFrame(pc, index = ['sample_{}'.format(i) for i in range(1, numsamples+1)], 
                              columns = ['PC{} value'.format(component_to_show)])
    print('PC{} value for each Sample:'.format(component_to_show), pc)
    return pc
    
def screeplot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0)**2 # get variance, i.e. (sdev)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["PC"+str(i) for i in x], rotation=0)
    plt.ylabel("Variance")
    plt.show()
    return y

def get_PCnumber(pca, standardised_values):
    # step one: plot the variance w.r.t pc
    y = screeplot(pca, standardised_values)
    
    n = len(y[y > 1])
    
    return n


def pca_biplot(pca, standardised_values, classifs, labels = None):
    """
    Add loading vectors to the PCA scatter plot. note the PCA scores have been normalized.
    
    Inputs:
        pca--pca component
        standardised_values--scaled values with mean of 0 and std of 1
        classifs--hue label
        labels--variable name list
    """
    score = pca.transform(standardised_values) 
    xs = score[:,0]
    ys = score[:,1]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())    
    coeff = np.transpose(pca.components_[0:2, :])
    n = coeff.shape[0]
    bar = pd.DataFrame(zip(xs*scalex, ys*scaley, classifs), columns=["PC1", "PC2", "Class"])
    sns.lmplot("PC1", "PC2", bar, hue="Class", fit_reg=False)
    # add loading vectors
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'slategray',alpha = 1)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'k', 
                     ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'k', 
                     ha = 'center', va = 'center')
            
    # add variance explained on the x,y labels
    summary = pca_summary(pca, standardised_values)
    plt.xlabel('PC1 ({:.1f}%)'.format(summary.varprop.loc['PC1', 'Proportion of Variance']*100))
    plt.ylabel('PC2 ({:.1f}%)'.format(summary.varprop.loc['PC2', 'Proportion of Variance']*100))
    plt.xlim(-1,1)
    plt.ylim(-1,1)

# the main function to be used in widgtify
def PCA_pipeline(data_path, target_col, n_components, loadings, PC_value,
                 component_to_show = 1, get_the_best_number_PC = True, visualization_2D = True, biplot = True):
    
    # step 1, take in all the parameters specified by user
    data_path = data_path
    # check data type
    if data_path.split('.')[-1] in ['csv', 'txt', 'data']:
        data = pd.read_csv(data_path)
    
    if data_path.split('.')[-1] in ['xlsx']:
        data = pd.read_excel('data_path') # need further editing
    
    if target_col == 'no target variable':
        feature_cols = data.columns.tolist()
    else:
        target_col = target_col
        feature_cols = data.columns.tolist()
        feature_cols.remove(target_col)
        
    if n_components > len(feature_cols):
        print('number of components should be equal or less than number of features')
    else:
        n_components = n_components
    
    loadings = loadings
    if PC_value:
        component_to_show = int(component_to_show)
        if component_to_show > len(feature_cols):
            print('no principle component {}'.format(component_to_show))
        else:
            component_to_show = component_to_show
            
    # step 2, based on the parameters taken in, organize the pipeline
    # a) perform pca
    features = data.loc[:, feature_cols].values
    target = data.loc[:, target_col].values
    
    features = StandardScaler().fit_transform(features)

    
    pca = PCA(n_components = n_components).fit(features)
    pca_summary(pca, features)
    
    # loadings for the principle components. 
    if loadings:
        loading_display(pca, n_components)
        
    
    if PC_value:
        pc_display(features, component_to_show, pca)
    
    # b) get the best number of components to be retained
    if get_the_best_number_PC:
        n = get_PCnumber(pca, features)
        print('Best number of PCs to be retained:', n)
        
    if visualization_2D:
        principalComponents = PCA(n_components=2).fit_transform(features)
        principalDf = pd.DataFrame(data = principalComponents, 
                                   columns = ['principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, data[target_col]], axis = 1)
        
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = finalDf[target_col].unique().tolist()
        colors = ['r', 'g', 'b']
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf[target_col] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c = color
                       , s = 50)
        ax.legend(targets)
        ax.grid()
        
    if biplot:
        pca_biplot(pca, features, data[target_col].values, labels = data.columns[1:])
        
        

def pca_app():
    app = widgetify(
            PCA_pipeline,
            data_path = Text(value = '../data/wine.csv',
                             placeholder = 'Type data path',
                             description = 'data path',
                             disabled = False),
            target_col = Text(value = 'class',
                               placeholder = 'Type the target variable',
                               description = 'target variable',
                               disabled = False),
            n_components = IntSlider(value = 3, min = 1, max = 20, step = 1,
                                     continuous_update = False, 
                                     description = 'number of PC to be shown',
                                     layout = Layout(width = 'auto', height = 'auto')),
            loadings = Checkbox(value = False,
                                description = "show loadings",
                                disabled = False),
            PC_value = Checkbox(value = False, 
                                description = 'show principal component values',
                                disabled = False),
            component_to_show = widgets.Dropdown(
               options=['1', '2', '3'],
               value= '1',
               description = 'which PC value to show',
               disabled = False),
            get_the_best_number_PC = Checkbox(value = True,
                                              description = 'number of PC retained',
                                              disabled = False),
            visualization_2D = Checkbox(value = True,
                                        description = 'plot 2 dimensional data',
                                        disabled = False),
            biplot = Checkbox(value = True,
                              description = 'biplot',
                              disabled = False))
    
    return app
    
