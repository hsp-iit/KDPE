
from rerun.blueprint import (
    Blueprint,
    BlueprintPanel,
    Horizontal,
    Vertical,
    SelectionPanel,
    Spatial3DView,
    TimePanel,
    TimeSeriesView,
    Tabs,
    BarChartView,
    Spatial2DView
)

def get_color(s: str) -> list[float]:
    """
    Convert a string to a RGB color.
    
    Args:
        s (str): Input string
    
    Returns:
        list[float]: RGB color values between 0 and 1
    """
    # Generate a hash of the string
    hash_value = hash(s)
    
    # Use the hash to generate RGB values
    r = ((hash_value & 0xFF0000) >> 16) / 255.0
    g = ((hash_value & 0x00FF00) >> 8) / 255.0
    b = (hash_value & 0x0000FF) / 255.0
    
    return [r, g, b, 1.0]


import rerun.blueprint as rrb
import rerun as rr

eef_entity_path = '/base/fr3_link0/fr3_link1/fr3_link2/fr3_link3/fr3_link4/fr3_link5/fr3_link6/fr3_link7/fr3_link8/fr3_hand'

def build_blueprint(
    data_example = {'poses': ['proprio', 'action'], 'images': ['agentview', 'eye_in_hand']},
):
    """
    Build the blueprint for the visualizer.
    """

    blueprint = Blueprint(
        Horizontal(
            # Left Side
            Vertical(
                Spatial3DView(name="robot view", origin="/", contents=["/**"]),
                Horizontal(*(Spatial2DView(name=f'{img}', origin=f'cameras/{img}') for img in data_example['images'])),
                row_shares=[3, 1],
            ),
            # Right Side
            Tabs(
                # Poses
                Tabs(
                    Vertical(
                        TimeSeriesView(name='x', origin='poses', contents=[f'poses/{pose}/components/x/**' for pose in data_example['poses']],         
                                     overrides={f'poses/{pose}/components/x': rr.SeriesLines.from_fields(names=[pose], widths=[2.0], colors=[get_color(pose)]) for pose in data_example['poses']}),
                        TimeSeriesView(name='y', origin='poses', contents=[f'poses/{pose}/components/y/**' for pose in data_example['poses']],
                                     overrides={f'poses/{pose}/components/y': rr.SeriesLines.from_fields(names=[pose], widths=[2.0], colors=[get_color(pose)]) for pose in data_example['poses']}),
                        TimeSeriesView(name='z', origin='poses', contents=[f'poses/{pose}/components/z/**' for pose in data_example['poses']],
                                     overrides={f'poses/{pose}/components/z': rr.SeriesLines.from_fields(names=[pose], widths=[2.0], colors=[get_color(pose)]) for pose in data_example['poses']}),
                        name='Position'
                    ),
                    Vertical(
                        TimeSeriesView(name='ax', origin='poses', contents=[f'poses/{pose}/components/ax/**' for pose in data_example['poses']],
                                     overrides={f'poses/{pose}/components/ax': rr.SeriesLines.from_fields(names=[pose], widths=[2.0], colors=[get_color(pose)]) for pose in data_example['poses']}),
                        TimeSeriesView(name='ay', origin='poses', contents=[f'poses/{pose}/components/ay/**' for pose in data_example['poses']],
                                     overrides={f'poses/{pose}/components/ay': rr.SeriesLines.from_fields(names=[pose], widths=[2.0], colors=[get_color(pose)]) for pose in data_example['poses']}),
                        TimeSeriesView(name='az', origin='poses', contents=[f'poses/{pose}/components/az/**' for pose in data_example['poses']],
                                     overrides={f'poses/{pose}/components/az': rr.SeriesLines.from_fields(names=[pose], widths=[2.0], colors=[get_color(pose)]) for pose in data_example['poses']}),
                        name='Orientation'
                    ),
                    name='Poses'
                ),
                # EEF View
                Vertical(
                    Spatial3DView(name='End Effector View Front', origin=eef_entity_path, contents=[f'{eef_entity_path}/**', '/trajectories/**', f'- {eef_entity_path}/right_camera']),
                    Spatial3DView(name='End Effector View Right', origin=eef_entity_path, contents=[f'{eef_entity_path}/**', '/trajectories/**', f'- {eef_entity_path}/front_camera']),
                    name='End Effector View',
                )
            ),
            column_shares=[3, 2],
        ),
        BlueprintPanel(state='collapsed'),
        SelectionPanel(state='collapsed'),
        TimePanel(state='collapsed'),
        auto_views=False,
    )

    return blueprint
