import numpy as np
from scipy.spatial.transform import Rotation
from envs.scan_geometry import horizontal_ready_tool

def test_scanner_axis_is_horizontal_for_all_asset_rolls_and_circle_angles():
    rng=np.random.default_rng(12)
    for angle in np.linspace(0,2*np.pi,40):
        functional=np.eye(4);functional[:3,:3]=Rotation.random(random_state=rng).as_matrix();functional[:3,3]=rng.normal(size=3)
        tool=np.eye(4);tool[:3,:3]=Rotation.random(random_state=rng).as_matrix();tool[:3,3]=rng.normal(size=3)
        offset=np.array([0,.1*np.cos(angle),.1*np.sin(angle)])
        center=np.array([-.03,-.02,.95]);ready,target=horizontal_ready_tool(tool,functional,center+offset,offset)
        np.testing.assert_allclose(target[:3,:3]@np.array([0,0,-1]),[1,0,0],atol=1e-12)
        np.testing.assert_allclose(target[:3,3],center+[.05,0,0],atol=1e-12)
        np.testing.assert_allclose(ready@np.linalg.inv(tool)@functional,target,atol=1e-12)
        assert abs(np.linalg.det(target[:3,:3])-1)<1e-12
