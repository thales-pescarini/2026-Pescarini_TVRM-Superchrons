# Fig. 1: Gondwana APWP (Torsvik et al. 2012, spline path) coloured by age, with the Nama
# pole and the remagnetization poles of West Gondwana carbonates (Pescarini et al. 2025,
# Table 3). Southern African coordinates; the continents are drawn in the Gondwana fit.
# Needs cartopy (Natural Earth coastlines are downloaded on the first run) and pmagpy.
# Usage: python figures/fig1_gondwana_apwp.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
from scipy.interpolate import splprep, splev
from pmagpy.ipmag import equi

AGE_MIN, AGE_MAX = 0.0, 550.0
CMAP = plt.cm.viridis
LABEL_AGES = [0, 100, 200, 300, 400, 450, 500, 525, 550]
A95_MAX = 30.0               # running-mean poles with larger A95 are not drawn
CRATON_COLORS = {'Kalahari': '#d62728', 'Sao Francisco': '#1f77b4',
                 'Congo': '#ff7f0e', 'Amazonia': '#2ca02c'}

# Finite rotations to NW Africa, McElhinny et al. (2003): (lat, lon, angle)
ROT = {
    'southern_africa': (9.3, 5.7, -7.8),
    'NE_africa': (19.2, 352.6, -6.3),
    'arabia': (26.2, 11.2, -14.2),
    'south_america': (53.0, 325.0, 51.0),
    'india': (26.7, 37.3, -69.4),
    'australia': (28.1, 113.2, -52.1),
    'antarctica': (-12.4, 326.2, 53.3),
    'madagascar': (-14.9, 277.6, 15.7),
}
CRATON_BLOCK = {'Kalahari': 'southern_africa', 'Sao Francisco': 'south_america',
                'Amazonia': 'south_america', 'Congo': 'NE_africa'}

# Torsvik et al. (2012), spline path, southern African coordinates: age, lat, lon
apwp = np.array([
    (0, -87.9, 65.6), (10, -85.1, 5.1), (20, -84.6, 1.9), (30, -83.1, 346.5),
    (40, -79.1, 338.3), (50, -77.6, 6.4), (60, -73.9, 35.2), (70, -69.7, 47.0),
    (80, -70.1, 50.6), (90, -69.7, 58.7), (100, -66.1, 73.0), (110, -59.9, 81.7),
    (120, -52.8, 82.3), (130, -49.5, 81.1), (140, -54.5, 72.8), (150, -61.0, 64.0),
    (160, -60.5, 73.0), (170, -52.8, 79.3), (180, -56.8, 83.6), (190, -70.3, 82.9),
    (200, -71.4, 65.6), (210, -62.9, 60.1), (220, -58.1, 50.8), (230, -55.0, 48.9),
    (240, -47.8, 57.7), (250, -43.2, 63.5), (260, -42.3, 62.5), (270, -37.5, 59.0),
    (280, -35.6, 48.8), (290, -36.9, 48.1), (300, -30.8, 48.7), (310, -24.3, 47.9),
    (320, -23.3, 44.2), (330, -23.0, 40.3), (340, -21.7, 36.3), (350, -19.4, 31.4),
    (360, -15.2, 24.4), (370, -7.8, 16.6), (380, -4.3, 13.2), (390, -12.0, 13.2),
    (400, -19.8, 13.4), (410, -31.0, 9.8), (420, -28.9, 2.7), (430, -14.7, 355.3),
    (440, 3.1, 350.4), (450, 20.5, 347.8), (460, 30.5, 349.5), (470, 33.2, 358.6),
    (480, 31.1, 6.8), (490, 34.3, 12.9), (500, 30.7, 10.7), (510, 19.9, 4.0),
    (520, 9.3, 351.0), (530, 4.2, 339.8), (540, -1.3, 328.6), (550, -7.5, 313.4),
])

# Torsvik et al. (2012), running-mean poles: age, A95, lat, lon
rm = np.array([
    (0, 2.5, -87.1, 11.9), (10, 2.6, -86.5, 3.4), (20, 4.3, -85.0, 5.1),
    (30, 6.0, -82.6, 353.6), (40, 6.0, -77.9, 351.4), (50, 8.4, -76.7, 359.5),
    (60, 3.0, -71.0, 44.9), (70, 1.7, -69.7, 46.3), (80, 2.9, -70.5, 53.2),
    (90, 2.6, -70.3, 58.0), (100, 4.2, -68.9, 63.9), (110, 4.0, -56.7, 82.2),
    (120, 3.6, -54.1, 81.3), (130, 4.4, -48.9, 79.1), (140, 8.1, -49.9, 77.4),
    (150, 14.5, -62.8, 63.6), (160, 10.7, -61.1, 74.5), (170, 6.0, -55.7, 76.7),
    (180, 4.5, -60.6, 82.5), (190, 4.4, -66.3, 82.2), (200, 6.4, -70.8, 64.1),
    (210, 8.0, -64.6, 63.0), (220, 18.0, -56.8, 44.9), (230, 5.9, -54.7, 47.1),
    (240, 9.1, -48.3, 56.0), (250, 7.8, -42.6, 59.7), (260, 8.4, -42.6, 62.0),
    (270, 7.4, -38.1, 59.6), (280, 7.4, -36.4, 53.1), (290, 8.3, -38.7, 48.8),
    (300, 6.8, -29.5, 48.4), (310, 4.4, -24.3, 48.0), (320, 5.4, -24.4, 46.2),
    (330, 41.1, -25.3, 42.5), (340, np.nan, -22.0, 37.7), (350, 5.9, -18.6, 33.2),
    (360, 11.8, -10.8, 18.3), (370, 13.1, -6.2, 16.9), (380, 21.4, -0.3, 12.3),
    (390, np.nan, -12.0, 13.2), (400, 105.8, -23.6, 14.2), (410, 6.6, -43.9, 11.7),
    (420, np.nan, -23.9, 2.3), (430, np.nan, -3.5, 355.5), (440, 18.0, 17.0, 349.0),
    (450, 18.0, 17.0, 349.0), (460, 151.4, 31.8, 358.2), (470, 17.0, 33.6, 358.6),
    (480, 11.0, 30.7, 3.6), (490, 13.4, 30.8, 7.9), (500, 10.7, 25.2, 6.5),
    (510, 7.9, 17.0, 4.2), (520, 9.4, 13.8, 357.1), (530, 13.8, 8.5, 344.7),
    (540, 32.4, -3.9, 326.8), (550, 66.3, -14.7, 315.2),
])

# Pescarini et al. (2025), Table 3, present coordinates: code, lon, lat, A95, craton
POLES = [
    ('Nm', 348.7, 30.5, 4.92, 'Kalahari'),
    ('Sl1', 323.0, 33.0, 4.00, 'Sao Francisco'),
    ('Sl2', 321.4, 27.5, 4.90, 'Sao Francisco'),
    ('SL', 321.0, 30.2, 3.80, 'Sao Francisco'),
    ('Mb', 358.0, 34.3, 4.30, 'Congo'),
    ('Gu', 332.6, 29.9, 4.30, 'Amazonia'),
    ('Pg', 326.6, 33.1, 3.20, 'Amazonia'),
]

# countries of each Gondwana block (Natural Earth 110 m); the rest of Africa stays fixed
BLOCKS = {
    'south_america': lambda a: a.get('CONTINENT') == 'South America',
    'antarctica': lambda a: a.get('CONTINENT') == 'Antarctica',
    'australia': lambda a: a.get('NAME') == 'Australia',
    'india': lambda a: a.get('NAME') in ('India', 'Pakistan', 'Bangladesh', 'Nepal', 'Bhutan', 'Sri Lanka'),
    'madagascar': lambda a: a.get('NAME') == 'Madagascar',
    'arabia': lambda a: a.get('NAME') in ('Saudi Arabia', 'Yemen', 'Oman', 'United Arab Emirates', 'Qatar', 'Kuwait'),
}


def to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def to_latlon(v):
    v = v / np.linalg.norm(v, axis=0)
    return np.degrees(np.arcsin(np.clip(v[2], -1, 1))), np.degrees(np.arctan2(v[1], v[0])) % 360.0


def rotate(lat, lon, elat, elon, omega):
    # finite rotation about the Euler pole (elat, elon)
    P = to_xyz(np.atleast_1d(np.asarray(lat, float)), np.atleast_1d(np.asarray(lon, float)))
    E = to_xyz(elat, elon)
    w = np.radians(omega)
    ExP = np.cross(np.broadcast_to(E, P.T.shape), P.T).T
    return to_latlon(P * np.cos(w) + ExP * np.sin(w) + E[:, None] * (E @ P) * (1 - np.cos(w)))


def to_southern_africa(lat, lon, block):
    # block -> NW Africa, then NW Africa -> southern Africa (inverse rotation)
    b, r = ROT[block], ROT['southern_africa']
    lat, lon = rotate(lat, lon, *b)
    return rotate(lat, lon, r[0], r[1], -r[2])


def reconstruct(geom, block):
    def ring(coords):
        c = np.asarray(coords)
        lat, lon = to_southern_africa(c[:, 1], c[:, 0], block)
        return np.column_stack([(lon + 180) % 360 - 180, lat])

    def poly(p):
        return Polygon(ring(p.exterior.coords), [ring(i.coords) for i in p.interiors])

    return poly(geom) if geom.geom_type == 'Polygon' else MultiPolygon([poly(g) for g in geom.geoms])


def draw_gondwana(ax):
    reader = shpreader.Reader(shpreader.natural_earth(resolution='110m', category='cultural',
                                                      name='admin_0_countries'))
    groups = {k: [] for k in BLOCKS}
    groups['southern_africa'] = []
    for rec in reader.records():
        a = rec.attributes
        for block, test in BLOCKS.items():
            if test(a):
                groups[block].append(rec.geometry)
                break
        else:
            if a.get('CONTINENT') == 'Africa':
                groups['southern_africa'].append(rec.geometry)
    for block, geoms in groups.items():
        if geoms:
            ax.add_feature(ShapelyFeature([reconstruct(unary_union(geoms), block)], ccrs.PlateCarree()),
                           facecolor='0.90', edgecolor='0.45', linewidth=0.4, zorder=0)


# APWP spline densified in xyz (avoids the 0/360 wrap)
age, plat, plon = apwp.T
tck, _ = splprep(list(to_xyz(plat, plon)), u=(age - AGE_MIN) / (AGE_MAX - AGE_MIN), s=0.0, k=3)
un = np.linspace(0, 1, 250)
la_s, lo_s = to_latlon(np.array(splev(un, tck)))
age_s = AGE_MIN + un * (AGE_MAX - AGE_MIN)

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 8.0, 'axes.linewidth': 0.6,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
norm = Normalize(AGE_MIN, AGE_MAX)
w = 140.0 / 25.4
fig, ax = plt.subplots(figsize=(w, 0.98 * w),
                       subplot_kw={'projection': ccrs.Orthographic(central_longitude=10, central_latitude=-10)})
ax.set_global()
th = np.linspace(0, 2 * np.pi, 400)
ax.set_boundary(mpath.Path(np.column_stack([0.5 + 0.5 * np.cos(th), 0.5 + 0.5 * np.sin(th)])),
                transform=ax.transAxes)
draw_gondwana(ax)
ax.gridlines(zorder=0, alpha=0.3)

ax.plot(lo_s, la_s, transform=ccrs.Geodetic(), color='0.6', lw=0.6, zorder=2)
ax.scatter(lo_s, la_s, c=age_s, cmap=CMAP, norm=norm, s=16, edgecolors='k', linewidths=0.35,
           transform=ccrs.PlateCarree(), zorder=3)
for a in LABEL_AGES:
    k = int(np.argmin(np.abs(age_s - a)))
    ax.text(lo_s[k] + 2.5, la_s[k] + 2.5, f'{a}', transform=ccrs.PlateCarree(), fontsize=9,
            fontweight='bold', ha='left', va='bottom', zorder=7,
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.7))

# running-mean poles with A95 circles
for age_i, a95, lat_i, lon_i in rm:
    if np.isfinite(a95) and a95 <= A95_MAX:
        col = CMAP(norm(age_i))
        equi(ax, lon_i, lat_i, a95 * 111.32, col, alpha=0.9)
        ax.scatter(lon_i, lat_i, s=12, color=col, edgecolors='none', transform=ccrs.PlateCarree(), zorder=4)

# remagnetization poles, rotated to southern African coordinates
for code, lon, lat, a95, craton in POLES:
    lat_r, lon_r = to_southern_africa(lat, lon, CRATON_BLOCK[craton])
    lat_r, lon_r = lat_r[0], lon_r[0]
    col = CRATON_COLORS[craton]
    nama = code == 'Nm'
    equi(ax, lon_r, lat_r, a95 * 111.32, col, alpha=0.20, outline=True, fill=True)
    ax.scatter(lon_r, lat_r, marker='*' if nama else 'o', s=70 if nama else 30, c=[col], edgecolors='k',
               linewidths=0.5, transform=ccrs.PlateCarree(), zorder=10 if nama else 9)
    ax.text(lon_r + 2, lat_r + 2, code, transform=ccrs.PlateCarree(), fontsize=8,
            fontweight='bold' if nama else 'normal', zorder=11)

handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=c, markeredgecolor='k',
                  markeredgewidth=0.5, markersize=6, label=name) for name, c in CRATON_COLORS.items()]
ax.legend(handles=handles, title='Craton', loc='lower left', fontsize=8, title_fontsize=8,
          framealpha=0.9, handletextpad=0.4, labelspacing=0.3, borderpad=0.4)
sm = ScalarMappable(norm=norm, cmap=CMAP)
cb = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.40, aspect=25, pad=0.02)
cb.set_label('Age (Ma)', fontsize=8.0)
cb.ax.tick_params(labelsize=7.0)
plt.tight_layout()
fig.savefig('Fig1.pdf')
fig.savefig('Fig1.png', dpi=600)
