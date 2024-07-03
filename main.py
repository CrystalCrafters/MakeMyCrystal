from dash import Dash, html, dcc, Input, Output, State
import dash_vtk
from dash_vtk.utils import to_mesh_state
import base64
import os
from web_stl_generator import generate_stl_from_params  # Assuming you have updated the function to take parameters
from vtkmodules.vtkIOGeometry import vtkSTLReader  # Import vtkSTLReader
from vtkmodules.vtkFiltersSources import vtkPlaneSource  # Import vtkPlaneSource

# Create a temporary directory to store uploaded files
UPLOAD_DIRECTORY = "uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Dash setup
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"width": "100%", "height": "100%"},
    children=[
        dcc.Upload(
            id="upload-cif",
            children=html.Div(["Drag and Drop or ", html.A("Select a CIF File")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        html.Div(id="output-upload", style={"margin": "10px"}),
        html.Div([
            html.Label("Number of Unit Cells (x, y, z):"),
            dcc.Input(id="num-unit-cells-x", type="number", value=1),
            dcc.Input(id="num-unit-cells-y", type="number", value=1),
            dcc.Input(id="num-unit-cells-z", type="number", value=1),
        ], style={"margin": "10px"}),
        html.Div([
            html.Label("Rotation Angles (x, y, z):"),
            dcc.Input(id="rotation-x", type="number", value=0),
            dcc.Input(id="rotation-y", type="number", value=0),
            dcc.Input(id="rotation-z", type="number", value=0),
        ], style={"margin": "10px"}),
        html.Div([
            html.Label("Translation Vector (x, y, z):"),
            dcc.Input(id="translation-x", type="number", value=0),
            dcc.Input(id="translation-y", type="number", value=0),
            dcc.Input(id="translation-z", type="number", value=0),
        ], style={"margin": "10px"}),
        html.Div([
            html.Label("Base Level:"),
            dcc.Input(id="base-level", type="number", value=0),
        ], style={"margin": "10px"}),
        html.Button("Generate STL", id="generate-stl", n_clicks=0, style={"margin": "10px"}),
        html.Div(id="output-stl-path", style={"margin": "10px"}),
        html.Div(id="output-stl", style={"margin": "10px", "height": "400px"}),
    ],
)

@app.callback(
    Output("output-upload", "children"),
    Input("upload-cif", "contents"),
    State("upload-cif", "filename"),
)
def save_upload(contents, filename):
    if contents is not None:
        data = contents.encode("utf8").split(b";base64,")[1]
        cif_path = os.path.join(UPLOAD_DIRECTORY, filename)
        with open(cif_path, "wb") as fp:
            fp.write(base64.decodebytes(data))
        return f"Uploaded file: {filename}"
    return "No file uploaded yet."

@app.callback(
    Output("output-stl-path", "children"),
    Input("generate-stl", "n_clicks"),
    State("upload-cif", "filename"),
    State("num-unit-cells-x", "value"),
    State("num-unit-cells-y", "value"),
    State("num-unit-cells-z", "value"),
    State("rotation-x", "value"),
    State("rotation-y", "value"),
    State("rotation-z", "value"),
    State("translation-x", "value"),
    State("translation-y", "value"),
    State("translation-z", "value"),
    State("base-level", "value"))
def generate_stl(n_clicks, filename, num_x, num_y, num_z, rot_x, rot_y, rot_z, trans_x, trans_y, trans_z, base_level):
    if n_clicks > 0 and filename:
        cif_path = os.path.join(UPLOAD_DIRECTORY, filename)
        num_unit_cells = [num_x, num_y, num_z]
        rotation_angles = [rot_x, rot_y, rot_z]
        translation_vector = [trans_x, trans_y, trans_z]

        stl_file_path = generate_stl_from_params(
            cif_path,
            num_unit_cells,
            rotation_angles,
            translation_vector,
            base_level
        )

        if os.path.exists(stl_file_path):
            return stl_file_path
        else:
            return "Failed to generate STL file."

    return "Click 'Generate STL' to create the STL file."

@app.callback(
    Output("output-stl", "children"),
    Input("output-stl-path", "children"),
    State("base-level", "value"),
)
def render_stl(stl_file_path, base_level):
    if stl_file_path and os.path.exists(stl_file_path):
        stl_reader = vtkSTLReader()
        stl_reader.SetFileName(stl_file_path)
        stl_reader.Update()

        dataset = stl_reader.GetOutput()

        if dataset is None:
            return "Failed to read STL file."

        mesh_state = to_mesh_state(dataset)

        # Compute the bounds of the STL model
        bounds = dataset.GetBounds()
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        x_size = x_max - x_min
        y_size = y_max - y_min

        # Create the plane with the computed size
        plane_source = vtkPlaneSource()
        plane_source.SetOrigin(x_min, y_min, base_level)
        plane_source.SetPoint1(x_max, y_min, base_level)
        plane_source.SetPoint2(x_min, y_max, base_level)
        plane_source.Update()

        plane_dataset = plane_source.GetOutput()
        plane_mesh_state = to_mesh_state(plane_dataset)

        content = dash_vtk.View([
            dash_vtk.GeometryRepresentation([
                dash_vtk.Mesh(state=plane_mesh_state),
            ], property={"color": [0.8, 0.8, 0.8]}),
            dash_vtk.GeometryRepresentation([
                dash_vtk.Mesh(state=mesh_state),
            ])
        ])

        return content

    return "No STL file to display."

if __name__ == "__main__":
    app.run_server(debug=True)
