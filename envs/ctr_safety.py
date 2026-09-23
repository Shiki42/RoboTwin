"""Per-physics-step cross-arm checks with conservative collision-shape bounds."""
import itertools
import numpy as np
import sapien
from .paired_timing import CandidateRejected

class CrossArmSafety:
    def __init__(self, task):
        self.task = task
        self.shapes = {'left':[], 'right':[]}
        self.entities = {}
        self.minimum_clearance = float('inf')
        for side,prefix in (('left','fl_link'),('right','fr_link')):
            for link in getattr(task.robot,side+'_entity').get_links():
                if link.name.startswith(prefix) or link.name == side+'_camera':
                    self.add(link.entity,side)
        names = (('left',('red','yellow')),('right',('blue','green'))) if task.task_name == 'blocks_ranking_rgb_ctr' else (('left',('object',)),('right',('scanner',)))
        for side,actors in names:
            for name in actors:
                self.add(getattr(task,name).actor,side)
        if not all(self.shapes.values()):
            raise RuntimeError('missing collision geometry')

    def add(self, entity, side):
        self.entities[entity] = side
        for body in entity.get_components():
            if not isinstance(body,sapien.physx.PhysxRigidBaseComponent):
                continue
            for shape in body.get_collision_shapes():
                if isinstance(shape,sapien.physx.PhysxCollisionShapeBox):
                    half = np.asarray(shape.half_size)
                    lower,upper = -half,half
                elif hasattr(shape,'get_vertices'):
                    vertices = np.asarray(shape.get_vertices())*np.asarray(shape.get_scale())
                    lower,upper = vertices.min(axis=0),vertices.max(axis=0)
                elif isinstance(shape,sapien.physx.PhysxCollisionShapeSphere):
                    lower,upper = np.full(3,-shape.radius),np.full(3,shape.radius)
                elif isinstance(shape,sapien.physx.PhysxCollisionShapeCapsule):
                    upper = np.array([shape.half_length+shape.radius,shape.radius,shape.radius]);lower=-upper
                else:
                    raise TypeError(f'unsupported collision geometry: {type(shape)}')
                corners = np.array(list(itertools.product(*zip(lower,upper))))
                matrix = shape.get_local_pose().to_transformation_matrix()
                self.shapes[side].append((entity,corners@matrix[:3,:3].T+matrix[:3,3]))

    def check(self, step):
        for contact in self.task.scene.get_contacts():
            a,b = (body.entity for body in contact.bodies)
            side_a,side_b = self.entities.get(a),self.entities.get(b)
            if side_a and side_b and side_a != side_b:
                if any(point.separation < -.002 for point in contact.points):
                    raise CandidateRejected(f'cross-arm collision at step {step}: {a.name}, {b.name}')
        if self.task.task_name != 'blocks_ranking_rgb_ctr':
            return
        bounds = {}
        for side,shapes in self.shapes.items():
            boxes = []
            for entity,corners in shapes:
                matrix = entity.get_pose().to_transformation_matrix()
                points = corners@matrix[:3,:3].T+matrix[:3,3]
                boxes.append((points.min(axis=0),points.max(axis=0)))
            bounds[side] = np.asarray(boxes)
        left,right = bounds['left'],bounds['right']
        gap = np.maximum(0,np.maximum(left[:,None,0]-right[None,:,1],right[None,:,0]-left[:,None,1]))
        clearance = float(np.linalg.norm(gap,axis=-1).min())
        self.minimum_clearance = min(self.minimum_clearance,clearance)
        if clearance < .10:
            raise CandidateRejected(f'cross-arm conservative clearance {clearance:.6f} m < 0.10 m at step {step}')
