import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'script'))
from merge_paired_cohorts import reconstruct

class FrozenLabelTests(unittest.TestCase):
    def test_initial_delay_and_finished_hold(self):
        left=np.zeros((2,16));left[:,0]=1;left[:,14]=[.5,0]
        right=left.copy();right[:,14]=[.8,.2]
        reasons=np.array([[0,1],[0,1],[2,0],[2,0],[2,2]])
        indices=np.array([[0,-1],[1,-1],[2,0],[2,1],[2,2]])
        targets,active,first=reconstruct(dict(left=left,right=right,tail=np.empty((0,17))),np.arange(6),reasons,indices,[])
        np.testing.assert_allclose(targets,[[.5,1],[0,1],[0,.8],[0,.2],[0,.2]])
        np.testing.assert_array_equal(active,[[1,0],[1,0],[1,1],[1,1],[1,1]])
        np.testing.assert_array_equal(first,[0,2])

    def test_scan_tail_uses_shared_indices_and_supervised_hold(self):
        left=np.zeros((1,16));left[0,0]=1;left[0,14]=.5
        right=left.copy();right[0,14]=.8
        tail=np.zeros((2,17));tail[:,0]=[0,1];tail[:,1]=1;tail[:,15]=[.7,.9]
        reasons=np.array([[0,1],[4,0],[0,5],[5,0]])
        indices=np.array([[0,-1],[1,0],[0,-1],[-1,1]])
        targets,active,first=reconstruct(dict(left=left,right=right,tail=tail),np.arange(5),reasons,indices,[dict(event='scan_barrier_release',step=2)])
        np.testing.assert_allclose(targets,[[.5,1],[.5,.8],[.7,.8],[.7,.9]])
        np.testing.assert_array_equal(active,[[1,0],[1,1],[1,1],[1,1]])

if __name__=='__main__':unittest.main()
