"""Rigid functional-frame targets independent of asset-defined scan tilt."""
import numpy as np

def horizontal_ready_tool(tool, functional, object_center, offset, distance=.05):
    tool=np.asarray(tool);functional=np.asarray(functional)
    # Preserve the existing roll as much as possible while aligning local -Z
    # with world +X, the native scanner-to-object separation direction.
    z=np.array([-1.,0.,0.]);x=functional[:3,0].copy();x-=z*np.dot(x,z)
    if np.linalg.norm(x)<1e-8:
        x=np.cross(functional[:3,1],z)
    x/=np.linalg.norm(x);y=np.cross(z,x)
    target=np.eye(4);target[:3,:3]=np.column_stack((x,y,z))
    target[:3,3]=np.asarray(object_center)-np.asarray(offset)+np.array([distance,0.,0.])
    functional_from_tool=np.linalg.inv(functional)@tool
    return target@functional_from_tool,target
