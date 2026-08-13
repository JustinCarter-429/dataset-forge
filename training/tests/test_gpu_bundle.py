from dataset_forge_critic.gpu_bundle import PUBLIC,NATIVE
def test_frozen_bundle_fingerprints_are_pinned():
 assert len(PUBLIC)==64 and len(NATIVE)==64 and PUBLIC!=NATIVE
