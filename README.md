This is simple python re-implementation of the algorithms from papers 
Iscen.et.al "[Fast Spectral Ranking for Similarity Search](https://arxiv.org/abs/1703.06935)", CVPR2018
and Iscen et.al "[Efficient Diffusion on Region Manifolds: Recovering Small Objects with Compact CNN Representations](https://arxiv.org/abs/1611.05113)" CVPR 2017. 

It is NOT authors implementation and some parts, e.g. sparsification, truncation, etc. are missing.

Example of usage: copy files into [python](https://github.com/filipradenovic/revisitop/tree/master/python) directory of the RevisitOP benchmark and run 

    python example_evaluate_with_diff.py

Expected output:
    
    Plain
    >> roxford5k: mAP E: 84.81, M: 64.67, H: 38.47
    >> roxford5k: mP@k[ 1  5 10] E: [ 97.06  85.29  70.59], M: [ 97.14  82.86  64.29], H: [ 81.43  31.43  22.86]
    Conjugate gradient
    >> roxford5k: mAP E: 86.42, M: 72.52, H: 48.56
    >> roxford5k: mP@k[ 1  5 10] E: [ 92.65  91.18  82.35], M: [ 92.86  87.14  75.71], H: [ 87.14  41.43  27.14]
    Spectral K=100, R=2000
    >> roxford5k: mAP E: 86.5, M: 72.0, H: 45.7
    >> roxford5k: mP@k[ 1  5 10] E: [ 94.12  91.18  80.88], M: [ 94.29  82.86  70.  ], H: [ 81.43  41.43  22.86]

    