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


def test_roll_candidates_preserve_horizontal_direction_and_target_position():
    from envs.scan_geometry import horizontal_roll_candidates,select_clear_wrist_path
    tool=np.eye(4);functional=np.eye(4)
    poses,targets=horizontal_roll_candidates(tool,functional,[0,0,1],[0,0,0],np.eye(4))
    assert len(poses)==10
    for t in targets:
        np.testing.assert_allclose(t[:3,:3]@[0,0,-1],[1,0,0],atol=1e-12)
        np.testing.assert_allclose(t[:3,3],[.05,0,1])
    paths=np.zeros((3,4,6));paths[0,:,4]=-1.8;paths[1,:,4]=1.1;paths[2,:,4]=.9
    assert select_clear_wrist_path(['Success']*3,paths)==2
    paths[2,:,2]=4.5
    assert select_clear_wrist_path(['Success']*3,paths)==1
    import pytest
    with pytest.raises(ValueError):select_clear_wrist_path(['Success','Failure','Failure'],paths)
