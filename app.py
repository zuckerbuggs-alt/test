
import streamlit as st
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import math
from skyfield.api import EarthSatellite, load, wgs84
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# Page config
st.set_page_config(layout="wide", page_title="Space Debris Tracker & Launch Planner")

st.title("🚀 Space Debris Tracker & Graveyard Transfer Planner")
st.caption("Prototype app for visualizing debris orbits, planning launch windows, and estimating transfer delta-v.")

# =========================
# Sidebar controls
# =========================
st.sidebar.header("🛰️ TLE Input")
tle_mode = st.sidebar.radio(
    "Provide TLEs via:",
    ("Select from pre-loaded catalogs", "Upload TLE file", "Paste TLEs"),
)
show_all_debris = st.sidebar.checkbox("Show all debris from selected catalog", value=False)

# Define pre-loaded catalogs
CATALOG_URLS = {
    "International Space Station (ISS)": "https://celestrak.org/NORAD/elements/stations.txt",
    "Starlink Constellation": "https://celestrak.org/NORAD/elements/starlink.txt",
    "Iridium Constellation": "https://celestrak.org/NORAD/elements/iridium.txt",
    "Orbital Debris (>10cm)": "https://celestrak.org/NORAD/elements/active.txt", # Using active for a smaller demo
}

@st.cache_data(ttl=3600)  # Cache data for 1 hour to avoid repeated downloads
def get_tles_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching TLEs from {url}: {e}")
        return ""

tle_text = ""
if tle_mode == "Select from pre-loaded catalogs":
    catalog_name = st.sidebar.selectbox("Select a catalog", list(CATALOG_URLS.keys()))
    if catalog_name:
        tle_text = get_tles_from_url(CATALOG_URLS[catalog_name])
elif tle_mode == "Upload TLE file":
    up = st.sidebar.file_uploader("Upload TLE .txt", type=["txt"])
    if up:
        tle_text = up.getvalue().decode("utf-8")
elif tle_mode == "Paste TLEs":
    tle_text = st.sidebar.text_area("Paste TLE text here (name, line1, line2...)", height=200)

# =========================
# Parse TLEs
# =========================
def parse_tles(tle_txt):
    lines = [ln.strip() for ln in tle_txt.splitlines() if ln.strip() != ""]
    sat_list, i = [], 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            name, l1, l2 = f"OBJ_{i}", lines[i], lines[i + 1]
            sat_list.append((name, l1, l2))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
            sat_list.append((name, l1, l2))
            i += 3
        else:
            i += 1
    return sat_list

ts = load.timescale()
sats = []
for name, l1, l2 in parse_tles(tle_text if tle_text else ""):
    try:
        sat = EarthSatellite(l1, l2, name, ts)
        sats.append((name, sat))
    except Exception:
        pass

if not sats:
    st.warning("⚠️ No satellites parsed yet. Provide valid TLEs via the sidebar.")
    st.stop()

# =========================
# Select target satellite
# =========================
names = [n for n, _ in sats]
sel_name = st.selectbox("🎯 Select debris / satellite to target", names)
sel_sat = [s for n, s in sats if n == sel_name][0]

# =========================
# Mission parameters
# =========================
st.markdown("---")
st.subheader("⚙️ Mission Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    graveyard_alt_km = st.number_input(
        "Graveyard altitude (km)",
        value=300.0,
        min_value=50.0,
        max_value=5000.0,
        step=50.0,
        help="For geostationary orbits, the recommended graveyard altitude is about 300 km above the GEO belt (~35,786 km)."
    )
    parking_orbit_km = st.number_input("Parking orbit altitude (km)", value=200.0, min_value=100.0, max_value=1000.0, step=10.0)
with col2:
    search_days = st.number_input("Search window (days)", min_value=1, max_value=30, value=10)
    time_step_seconds = st.number_input("Time step (seconds)", min_value=60, max_value=3600, value=300)
with col3:
    proximity_deg = st.slider("Ground-track proximity (°)", 0.1, 10.0, 2.0)
    max_results = st.number_input("Max launch windows", min_value=1, max_value=20, value=6)

# Launch sites
LAUNCH_SITES = [
    {"name": "Kourou (French Guiana)", "lat": 5.23, "lon": -52.77},
    {"name": "Baikonur (Kazakhstan)", "lat": 45.92, "lon": 63.34},
    {"name": "Vandenberg (USA)", "lat": 34.74, "lon": -120.57},
    {"name": "Satish Dhawan (India)", "lat": 13.72, "lon": 80.23},
    {"name": "Cape Canaveral (USA)", "lat": 28.39, "lon": -80.61},
    {"name": "Tanegashima (Japan)", "lat": 30.39, "lon": 130.97},
    {"name": "Jiuquan (China)", "lat": 40.96, "lon": 100.28},
    {"name": "Wallops Flight Facility (USA)", "lat": 37.85, "lon": -75.49},
    {"name": "Kennedy Space Center (USA)", "lat": 28.52, "lon": -80.60},
    {"name": "Plesetsk (Russia)", "lat": 62.92, "lon": 40.57},
    {"name": "Vostochny (Russia)", "lat": 51.88, "lon": 128.32},
    {"name": "Wenchang (China)", "lat": 19.61, "lon": 110.95},
    {"name": "Xichang (China)", "lat": 28.24, "lon": 102.02},
    {"name": "Uchinoura (Japan)", "lat": 31.25, "lon": 131.08},
    {"name": "Semnan (Iran)", "lat": 35.22, "lon": 53.92},
    {"name": "Alcantara (Brazil)", "lat": -2.37, "lon": -44.39},
    {"name": "Pacific Spaceport Complex (USA)", "lat": 57.43, "lon": -156.41},
    {"name": "Mahia Peninsula (New Zealand)", "lat": -39.26, "lon": 177.86},
    {"name": "Palmachim (Israel)", "lat": 31.91, "lon": 34.68},
    {"name": "Sohae (North Korea)", "lat": 39.66, "lon": 124.70},
    {"name": "Taiyuan (China)", "lat": 37.54, "lon": 112.68},
    {"name": "Mid-Atlantic Regional Spaceport (USA)", "lat": 37.83, "lon": -75.49}
]

# =========================
# Helper functions
# =========================
R_EARTH, MU_EARTH = 6378.137, 398600.4418

def get_subpoint(satellite, t_sf):
    geoc = satellite.at(t_sf)
    sp = wgs84.subpoint(geoc)
    return sp.latitude.degrees, sp.longitude.degrees, sp.elevation.m / 1000

def hohmann_delta_v(r1, r2):
    v1, v2 = math.sqrt(MU_EARTH/r1), math.sqrt(MU_EARTH/r2)
    a = 0.5*(r1 + r2)
    v_trans_perigee = math.sqrt(MU_EARTH*(2/r1 - 1/a))
    v_trans_apogee = math.sqrt(MU_EARTH*(2/r2 - 1/a))
    return abs(v_trans_perigee - v1), abs(v2 - v_trans_apogee)

# =========================
# Current state
# =========================
now_utc = datetime.now(timezone.utc)
t_now = ts.from_datetime(now_utc)
lat0, lon0, alt0 = get_subpoint(sel_sat, t_now)
r_mag = np.linalg.norm(sel_sat.at(t_now).position.km)
current_alt = r_mag - R_EARTH

st.subheader(f"📡 Current State of {sel_name}")
st.metric("Latitude (°)", f"{lat0:.2f}")
st.metric("Longitude (°)", f"{lon0:.2f}")
st.metric("Altitude (km)", f"{current_alt:.1f}")

# Delta-v estimate
dv1, dv2 = hohmann_delta_v(r_mag, R_EARTH + graveyard_alt_km)
dv_total = dv1 + dv2
st.markdown("#### 🔑 Transfer Δv Estimate")
st.write(f"Injection burn: **{dv1:.3f} km/s** | Circularization: **{dv2:.3f} km/s**")
st.success(f"Total Δv (approx): {dv_total:.3f} km/s")

# =========================
# Search launch windows
# =========================
st.markdown("---")
st.subheader("📅 Candidate Launch Windows")

search_start, search_end = now_utc, now_utc + timedelta(days=int(search_days))
times = [search_start + timedelta(seconds=i) for i in range(0, int((search_end - search_start).total_seconds()), int(time_step_seconds))]
sf_times = ts.from_datetimes(times)
subpoints = [wgs84.subpoint(sel_sat.at(t)) for t in sf_times]
lats, lons = np.array([sp.latitude.degrees for sp in subpoints]), np.array([sp.longitude.degrees for sp in subpoints])

candidates = []
for site in LAUNCH_SITES:
    lat_diff, lon_diff = np.abs(lats - site["lat"]), np.minimum(np.abs(lons - site["lon"]), 360 - np.abs(lons - site["lon"]))
    idxs = np.where((lat_diff <= proximity_deg) & (lon_diff <= 30))[0]
    for idx in idxs:
        candidates.append({
            "site": site["name"],
            "lat": site["lat"], "lon": site["lon"],
            "utc_time": times[idx],
            "sat_lat": lats[idx], "sat_lon": lons[idx],
            "alt_km": subpoints[idx].elevation.m/1000,
        })

candidates = sorted(candidates, key=lambda x: x["utc_time"])[:max_results]

if candidates:
    df = pd.DataFrame([{
        "Launch Site": c["site"],
        "UTC Time": c["utc_time"].strftime("%Y-%m-%d %H:%M:%S"),
        "Sat Lat": round(c["sat_lat"], 2),
        "Sat Lon": round(c["sat_lon"], 2),
        "Sat Alt (km)": round(c["alt_km"], 1),
    } for c in candidates])
    st.dataframe(df)

    # =========================
    # Map Visualization of Candidate Launch Sites
    # =========================
    df_candidates = pd.DataFrame(candidates)
    fig_map = px.scatter_mapbox(
        df_candidates,
        lat="lat",
        lon="lon",
        hover_name="site",
        hover_data={"utc_time": True, "alt_km": True, "lat": False, "lon": False},
        color_discrete_sequence=["red"],
        zoom=1,
        height=400,
        title="Candidate Launch Sites"
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",  # free style, no API key needed
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    st.plotly_chart(fig_map, use_container_width=True)

else:
    st.info("No candidate launch windows found. Try adjusting search parameters.")

# =========================
# Map Visualization of Orbit
# =========================
st.markdown("---")
st.subheader("🌍 3D Orbit Visualization")

# Generate satellite orbit path in 3D
positions = sel_sat.at(sf_times).position.km
orbit_x, orbit_y, orbit_z = positions[0], positions[1], positions[2]

# Generate Earth sphere data
phi = np.linspace(0, 2 * np.pi, 100)
theta = np.linspace(0, np.pi, 100)
x_earth = R_EARTH * np.outer(np.cos(phi), np.sin(theta))
y_earth = R_EARTH * np.outer(np.sin(phi), np.sin(theta))
z_earth = R_EARTH * np.outer(np.ones(100), np.cos(theta))

# Create the figure
fig = go.Figure()

# Plot Earth Sphere
fig.add_trace(go.Surface(
    x=x_earth, y=y_earth, z=z_earth,
    colorscale=[[0, 'blue'], [1, 'blue']],
    opacity=0.5,
    showscale=False
))

# Plot the satellite's orbit path
fig.add_trace(go.Scatter3d(
    x=orbit_x, y=orbit_y, z=orbit_z,
    mode='lines',
    line=dict(color='yellow', width=5),
    name='Orbit'
))

# Plot the current satellite position
current_pos = sel_sat.at(t_now).position.km
fig.add_trace(go.Scatter3d(
    x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
    mode='markers',
    marker=dict(size=8, color='red'),
    name='Current Position'
))

# Set the layout and aspect ratio
fig.update_layout(
    title_text=f"3D Orbit of {sel_name}",
    scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)',
        aspectmode='data'  # spherical aspect ratio
    ),
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# Summary
# =========================
st.markdown("---")
st.subheader("📌 Suggested Plan")
if candidates:
    c0 = candidates[0]
    st.success(f"**Earliest launch window**: {c0['site']} at {c0['utc_time'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
    st.write(f"Satellite altitude: {c0['alt_km']:.1f} km | Lat: {c0['sat_lat']:.2f}°, Lon: {c0['sat_lon']:.2f}°")
    st.write(f"Estimated Δv for graveyard transfer: **{dv_total:.3f} km/s**")
else:
    st.warning("No valid windows found. Try adjusting search parameters.")

st.caption("✨ This prototype demonstrates key principles of astrodynamics for mission planning.")
