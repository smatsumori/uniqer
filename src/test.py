import unittest
from utils.utils import seed_everything

# TestCase
from guesswhat.utils.gwdataloader import TestDataLoader
from guesswhat.utils.gwdataloader import TestOracleFeatDataloader
from guesswhat.utils.gwdataloader import TestUniqerFeatDataloader
from guesswhat.utils.gwdataloader import TestRLDataloader

# from modules.text_encoder.bert_embedding import TestBertEmbeddings
# from scripts.presave_img_vec import TestPresaveVecs
# from modules.oracle.gworacle import TestOracleModel
# from scripts.presave_objanswer import TestPresaveObjans
# from modules.gw_single_tf import TestGWSingleTF
# from modules.guesswhat.rl.network import TestObjectTargetingNetwork
# from modules.guesswhat.rl.environment import TestEnvironment

import logging
SETUP_DONE = False


def suite():
    seed_everything(76)
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
    suite = unittest.TestSuite()
    # ----------------------- Test GW Dataloaders ---------------------------
    # Data Requirements:
    # - test_dataset_vis: gw
    # - test_dataset_txt: gw
    # - test_oraclefeatdataset: gw, bert_embeddings, img/crpd features
    # - test_uniqerfeatdataset: gw, bert_embeddings, img/crpd features, objans
    # - test_rl_dataloader: gw, bert_embeddings, img/crpd features, objans
    # suite.addTest(TestDataLoader('test_dataset_vis'))
    # suite.addTest(TestDataLoader('test_dataset_txt'))
    # suite.addTest(TestOracleFeatDataloader('test_oraclefeatdataset'))
    # suite.addTest(TestUniqerFeatDataloader('test_uniqerfeatdataset'))
    # suite.addTest(TestRLDataloader('test_rl_dataloader'))

    # ----------------------- Test Presaves ---------------------------
    # Data Requirements:
    # suite.addTest(TestBertEmbeddings('test_tokenizer'))
    # suite.addTest(TestBertEmbeddings('test_embedding'))
    # suite.addTest(TestBertEmbeddings('test_tokenizer'))
    # suite.addTest(TestBertEmbeddings('test_saved_embeddings'))
    # suite.addTest(TestPresaveVecs('test_run_presave_gw_resnet_vector'))

    # ----------------------- Test Oracle Model ---------------------------
    # suite.addTest(TestOracleModel('test_oraclemodels'))
    # suite.addTest(TestPresaveObjans('test_oracle_weights'))

    # ----------------------- Test UniQer SL ---------------------------
    # Data Requirements: None
    # suite.addTest(TestGWSingleTF('testgw_tfenc'))
    # suite.addTest(TestGWSingleTF('testgw_tfdec'))
    # suite.addTest(TestGWSingleTF('testgw_tf'))
    # suite.addTest(TestGWSingleTF('test_tf_inference'))

    # ----------------------- Test UniQer RL ---------------------------
    # suite.addTest(TestObjectTargetingNetwork('test_object_targeting_network'))
    # suite.addTest(TestObjectTargetingNetwork('test_extract_top_k_features'))
    # suite.addTest(TestEnvironment('test_environment'))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
