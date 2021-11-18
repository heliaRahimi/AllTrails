import plotly.express as px
import plotly.graph_objects as go
from GimmeAllTheTrails.dataset import AllTrails

def mapbox(csv_dir):
    llt = AllTrails(csv_dir).lat_lon_trail_id

    # mapbox_access_token = open("https://studio.mapbox.com/tilesets/mapbox.mapbox-traffic-v1").read()
    fig = go.Figure(go.Scattermapbox(
        lat=llt.latitude.tolist(),
        lon=llt.longitude.tolist(),
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=9
        ),
        text=llt.trail_id.tolist()))
    fig.update_layout(
        autosize=True,
        geo=dict(bgcolor='rgba(0,0,0,1)'),
        hovermode='closest',
        mapbox_style="carto-darkmatter",
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=45,
                lon=-63
            ),
            pitch=0,
            zoom=6.5
        ),

    )
    return fig