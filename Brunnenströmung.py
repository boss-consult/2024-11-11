# load packages
import numpy as np
import geopandas as gpd
from shapely import LineString,Point
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import colorcet as cc

# define aquifer settings
kf = 1.5e-3 # hydraulic conductivity in m/s
I = 2e-3 # ambient hydraulic gradient in m/m
m = 8 # aquifer thickness in m
alpha = 97.5 # angle of ambient flow in °
x_ref = 503040.1 # reference location x
y_ref = 5376067.0 # reference location y
h_ref = 123.4 # reference potential in m

# define well settings
x_well = 503030.0 # x location of well
y_well = 5376067.0 # y location of well
Q = -1*1e-3 # pumping rate in m³/s (negative --> extraction; positive --> injection)
d_well = 0.2 # well diameter in m

# load source zone
source_area = gpd.read_file('Schadstoffherd.gpkg')

# other length settings
xmin = x_well-110
xmax = x_well+40
ymin = y_well-50
ymax = y_well+50
Lx   = xmax-xmin # domain length in m
Ly   = ymax-ymin # domain width in m
dx   = np.min([Lx/100,Ly/100]) # spatial resolution in m

# utility conversions
T = m*kf # transmissivity in m²/s
xy_well =  np.vstack([x_well,y_well])
alpha = alpha/360*2*np.pi
r_well = d_well/2 # well radius in m
if np.abs(Q) > 0:
    well_is_active = True
else:
    well_is_active = False

# initialize the plot
fig, ax = plt.subplots()
# plot source area
source_area.plot(ax=ax,facecolor="#d9d9d9", edgecolor="#636363",linestyle='dashed')
# plot well itself
ax.scatter(x_well,y_well,30,color='black',zorder=6)

ax.axis('scaled')
ax.xaxis.set_major_formatter(lambda val,pos: '{:.1f} m'.format(val-xmin))
ax.yaxis.set_major_formatter(lambda val,pos: '{:.1f} m'.format(val-ymin))

ax.set_xlim(left=xmin,right=xmax)
ax.set_ylim(bottom=ymin,top=ymax)
xticks =  ax.get_xticks()
yticks =  ax.get_yticks()
ax.set_xticks([tick - (xticks[0]-xmin) for tick in xticks])
ax.set_yticks([tick - (yticks[0]-ymin) for tick in yticks])

ax.set_xlim(left=xmin,right=xmax)
ax.set_ylim(bottom=ymin,top=ymax)

# construct a grid for evaluation
x = np.arange(xmin,xmax,dx)
y1 = np.arange(ymin,y_well,dx)
y2 = np.arange(y_well,ymax,dx)
# for y we use a local refinement around the x-axis for improved precision at singularity
y = np.unique(np.concatenate([y1,np.linspace(y_well-dx,y_well+dx,8),y2])) 
X, Y = np.meshgrid(x, y, indexing='xy')

# analytical solution of hydraulic head field
def h_fun(X,Y):
    # determine distance from XY to well
    r = np.sqrt((X-x_well)**2+(Y-y_well)**2)
    # within the well, we use the well radius
    r = np.where(r < r_well, r_well, r)
    
    return h_ref - I * ( np.sin(alpha)*(X-x_ref) + np.cos(alpha)*(Y-y_ref) ) - Q/2/np.pi/T*np.log(r)

h = h_fun(X,Y)

# define contour level values
delta_h = I*np.sqrt(Lx*Ly)/11
print('h-Isolinien im Abstand von {:.3f} m.'.format(delta_h))
lvls_h = np.arange(np.round(np.nanmin(h),2),np.nanmax(h),delta_h) # isoline values

# plot gradient-colored background map
ax.pcolormesh(X, Y, h,cmap=cc.m_CET_D11,alpha=.2,edgecolors='none',zorder=0)

# plot and store contour lines 
potential_lines = ax.contour(X,Y,h,lvls_h,colors='#1f78b4',linewidths=2,zorder=3)

# analytical solution of stream function
def psi_kernel(X,Y):
    return (Y*np.sin(alpha)-X*np.cos(alpha))* T*I + Q/2/np.pi*np.arctan2(Y-y_well,X-x_well)

def psi_fun(X,Y):
    psi = psi_kernel(X,Y)

    # modify integration constant such that psi is 0 on first bounding streamline
    psi = psi - psi_kernel(x_well+np.sin(alpha),y_well+np.cos(alpha))
        
    # set values directly on axis to NaN to account for jump across singularity
    sel = (X<x_well) & (np.abs(Y-y_well) <= 0.1*dx)
    psi = np.where(sel,np.nan,psi)

    return psi

# apply analytical solution of stream function
psi = psi_fun(X,Y)

# contour settings
psi_divisor = 6
if well_is_active:
    delta_psi = np.abs(Q)/psi_divisor # contour line interval
else:
    delta_psi = T*I*np.sqrt(Lx*Ly)/psi_divisor # contour line interval
    
print('Jede Stromröhre führt {:.3f} l/s.'.format(delta_psi*1000))

# construct symmetric streamline isovalues
lvls_psi  = np.concatenate([np.arange(0,np.nanmax(psi),delta_psi),
                          Q,
                          np.arange(0,np.nanmin(psi),-delta_psi)
                         ],axis=None)
lvls_psi = np.unique(lvls_psi)

# append to plot
streamlines = ax.contour(X,Y,psi,lvls_psi,colors='#6a3d9a',linewidths=1.5,linestyles='solid',zorder=2)

# special case if flow is parallel to x axis
if (np.isclose(alpha,np.pi) or np.isclose(alpha,0.0)) and (psi_divisor % 2)==0:
    ax.plot([xmin,x_well],[y_well,y_well],color='#6a3d9a',linewidth=1.5,linestyle='solid',zorder=2)

# add bounding streamlines
if well_is_active:
    # extraction width
    B_A = np.abs(Q)/(T*I)
    print('Entnahmebreite: {:.2f} m'.format(B_A))

# add first bounding streamline
if well_is_active:
    # construct extraction outline in local coordinate system
    y_local   = np.linspace(-0.99,0.99,101)*0.5*B_A
    with np.errstate(divide='ignore',invalid='ignore'):
        x_local = y_local*(1/np.tan(2*np.pi*T*I*y_local/Q))
    xy_local = np.vstack([x_local,y_local])

    # go from local to global
    rot_matrix = np.array([[np.cos(alpha+0.5*np.pi), np.sin(alpha+0.5*np.pi)],
                           [-np.sin(alpha+0.5*np.pi), np.cos(alpha+0.5*np.pi)]])
    xy_E = np.dot(rot_matrix,xy_local) + np.vstack([x_well,y_well])

    # plot extraction outline
    ax.fill(xy_E[0,:],xy_E[1,:],linewidth=2,edgecolor='none',facecolor='#ff7f00',alpha=0.2,zorder=4)
    ax.plot(xy_E[0,:],xy_E[1,:],linewidth=2,color='#ff7f00',zorder=5)

if well_is_active:   
    # second bounding streamline
    maxDist = np.nanmax(np.abs(x_local))
    p = maxDist * np.array([0,1])
    xy_TS = xy_well + p * np.vstack([np.sin(alpha),np.cos(alpha)]) 
    ax.plot(xy_TS[0,:],xy_TS[1,:],linewidth=2,color='#ff7f00',zorder=5)

if well_is_active: 
    # stagnation point
    xy_KP = xy_well + np.abs(Q/2/np.pi/T/I) * np.vstack([np.sin(alpha),np.cos(alpha)]) 
    ax.scatter(xy_KP[0],xy_KP[1],50,color='#ff7f00',zorder=6)

# helper function to convert matplotlib contour output to LineStrings
# source: https://discourse.matplotlib.org/t/collections-attribute-deprecation-in-version-3-8/24164/14
def contours_to_linestring(contour_set):
    contours = []
    paths_by_layer = []
    for i, joined_paths_in_layer in enumerate(contour_set.get_paths()):
        separated_paths_in_layer = []
        path_vertices = []
        path_codes = []
        for verts, code in joined_paths_in_layer.iter_segments():
            if code == Path.MOVETO:
                if path_vertices:
                    separated_paths_in_layer.append(Path(np.array(path_vertices), np.array(path_codes)))
                path_vertices = [verts]
                path_codes = [code]
            elif code == Path.LINETO:
                path_vertices.append(verts)
                path_codes.append(code)
            elif code == Path.CLOSEPOLY:
                path_vertices.append(verts)
                path_codes.append(code)
        if path_vertices:
            separated_paths_in_layer.append(Path(np.array(path_vertices), np.array(path_codes)))
    
        paths_by_layer.append(separated_paths_in_layer)
    
    for i, paths_in_layer in enumerate(paths_by_layer):
        # Process path
        for path in paths_in_layer:
            if not path.vertices.size:
                continue
            # Check the number of vertices
            if len(path.vertices) < 2:
                continue
            # Create linestring if valid
            geom = LineString(path.vertices)
            if geom.is_valid:
                contours.append(geom)
    return contours

# export settings
myCRS = 'EPSG:25832' # reference coordinate system
filename_GPKG = "Strömung.gpkg" # output file

# collect all geometries and assign labels              
geometries = {
    'Potentiallinien': contours_to_linestring(potential_lines),
    'Stromlinien': contours_to_linestring(streamlines),
    'Brunnen': gpd.points_from_xy([x_well], [y_well])
}
if well_is_active:
    geometries_well = {
        'Erfassungsbereich': [LineString(xy_E.T)],
        'Trennstromlinie': [LineString(xy_TS.T)],
        'Kulminationspunkt': gpd.points_from_xy(xy_KP[0], xy_KP[1])
    }
    geometries.update(geometries_well)

# store each geometry in the same geopackage              
for label,geometry in geometries.items():
    gpd.GeoDataFrame(geometry=geometry,crs=myCRS).to_file(filename_GPKG,layer=label) 
