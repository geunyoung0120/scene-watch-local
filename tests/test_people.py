import unittest
from scripts.people_training import group_splits, candidate_allowed, validate_review


class PeopleTrainingTests(unittest.TestCase):
    def test_review_rejects_changed_image_bytes_with_same_id(self):
        review={'reviewed_ids':[1], 'image_sha256':{'1':'original'}}
        validate_review(review,[{'id':1,'sha256':'original'}])
        with self.assertRaises(ValueError):
            validate_review(review,[{'id':1,'sha256':'changed'}])

    def test_near_duplicate_groups_never_cross_splits(self):
        items=[{'id':i,'group_id':str(i//3),'category':['crowd','close','people'][i%3]} for i in range(90)]
        mapping=group_splits(items)
        for i in range(0,90,3):
            self.assertEqual(len({mapping[x] for x in range(i,i+3)}),1)
        self.assertEqual(mapping,group_splits(items))
        self.assertEqual(set(mapping.values()),{'train','validation','test'})

    def test_rejects_all_safe_model_and_recall_regression(self):
        baseline={'recall':.9,'fpr':.3,'new_fpr':.8}
        self.assertFalse(candidate_allowed(baseline,{'recall':0,'fpr':0,'new_fpr':0}))
        self.assertFalse(candidate_allowed(baseline,{'recall':.8,'fpr':.2,'new_fpr':.1}))
        self.assertTrue(candidate_allowed(baseline,{'recall':.88,'fpr':.30,'new_fpr':.1}))
        self.assertFalse(candidate_allowed(baseline,{'recall':.9,'fpr':.4,'new_fpr':.1}))
