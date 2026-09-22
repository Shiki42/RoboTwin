import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'script'))
from merge_paired_cohorts import reconstruct

class FrozenLabelTests(unittest.TestCase):
    def test_shards_align_by_column_name(self):
        import tempfile
        import pyarrow as pa
        import pyarrow.parquet as pq
        from merge_paired_cohorts import read_tables
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data').mkdir()
            pq.write_table(pa.table({'seed':[1],'value':[3]}),root/'data/a.parquet')
            pq.write_table(pa.table({'value':[4],'seed':[2]}),root/'data/b.parquet')
            self.assertEqual(read_tables(root,'data').to_pydict(),{'seed':[1,2],'value':[3,4]})

    def test_json_scalar_shape_encodes_as_scalar(self):
        from datasets import Dataset
        from lerobot.datasets.feature_utils import get_hf_features_from_features
        features={'retime.source_cohort':{'dtype':'int64','shape':[1]}}
        hf=get_hf_features_from_features({k:{**v,'shape':tuple(v['shape'])} for k,v in features.items()})
        data=Dataset.from_dict({'retime.source_cohort':[0,1]},features=hf)
        self.assertEqual(list(data['retime.source_cohort']),[0,1])

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
