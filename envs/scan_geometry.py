"""Rigid functional-frame targets independent of asset-defined scan tilt."""
import numpy as np

def horizontal_ready_tool(tool, functional, object_center, offset, distance=.05, reference=None):
    tool=np.asarray(tool);functional=np.asarray(functional)
    # Preserve the existing roll as much as possible while aligning local -Z
    # with world +X, the native scanner-to-object separation direction.
    frame=functional if reference is None else np.asarray(reference)
    z=np.array([-1.,0.,0.]);x=frame[:3,0].copy();x-=z*np.dot(x,z)
    if np.linalg.norm(x)<1e-8:
        x=np.cross(frame[:3,1],z)
    x/=np.linalg.norm(x);y=np.cross(z,x)
    target=np.eye(4);target[:3,:3]=np.column_stack((x,y,z))
    target[:3,3]=np.asarray(object_center)-np.asarray(offset)+np.array([distance,0.,0.])
    functional_from_tool=np.linalg.inv(functional)@tool
    return target@functional_from_tool,target


def horizontal_roll_candidates(tool,functional,object_center,offset,reference,count=10):
    _,target=horizontal_ready_tool(tool,functional,object_center,offset,reference=reference)
    relative=np.linalg.inv(functional)@tool
    poses=[];targets=[]
    for angle in np.arange(count)*2*np.pi/count:
        c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[1,0,0],[0,c,-s],[0,s,c]])
        candidate=target.copy();candidate[:3,:3]=rotation@target[:3,:3]
        poses.append(candidate@relative);targets.append(candidate)
    return poses,targets


def select_clear_wrist_path(status,positions,limit=1.4):
    candidates=[]
    for i,state in enumerate(status):
        if state!='Success':continue
        path=positions[i]
        bend=float(np.max(np.abs(path[:,4])))
        if not np.isfinite(path).all() or bend>limit:continue
        travel=float(np.sum(np.linalg.norm(np.diff(path,axis=0),axis=1)))
        candidates.append((bend,travel,i))
    if not candidates:
        raise ValueError('no planned roll avoids wrist self-contact envelope')
    return min(candidates)[2]
