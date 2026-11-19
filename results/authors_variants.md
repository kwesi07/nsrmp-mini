### Soft evaluation (current): every object blended by its subject weight (mean ± sample std over seeds)

| Model | Split | IOU | IOU-M | Subject acc. | Placement acc. |
|---|---|---|---|---|---|
| Ours (neuro-symbolic) | overall | 0.929 ± 0.002 | 0.776 ± 0.005 | 1.000 ± 0.001 | 0.955 ± 0.005 |
| Ours (neuro-symbolic) | simple | 0.929 ± 0.002 | 0.779 ± 0.006 | 1.000 ± 0.000 | 0.958 ± 0.011 |
| Ours (neuro-symbolic) | complex | 0.929 ± 0.001 | 0.774 ± 0.003 | 0.999 ± 0.001 | 0.951 ± 0.001 |
| Parse-matched neural baseline (NMN+-like) | overall | 0.946 ± 0.002 | 0.786 ± 0.011 | 0.999 ± 0.001 | 0.940 ± 0.013 |
| Parse-matched neural baseline (NMN+-like) | simple | 0.947 ± 0.002 | 0.788 ± 0.010 | 1.000 ± 0.000 | 0.943 ± 0.013 |
| Parse-matched neural baseline (NMN+-like) | complex | 0.944 ± 0.002 | 0.784 ± 0.012 | 0.999 ± 0.001 | 0.937 ± 0.014 |
| End-to-end GRU baseline | overall | 0.744 ± 0.002 | 0.005 ± 0.001 | 0.265 ± 0.005 | 0.007 ± 0.001 |
| End-to-end GRU baseline | simple | 0.743 ± 0.004 | 0.005 ± 0.001 | 0.281 ± 0.018 | 0.007 ± 0.001 |
| End-to-end GRU baseline | complex | 0.744 ± 0.001 | 0.004 ± 0.001 | 0.249 ± 0.009 | 0.007 ± 0.001 |
| NMN+ (authors' scoring) | overall | 0.750 ± 0.002 | 0.005 ± 0.001 | 0.247 ± 0.011 | 0.007 ± 0.001 |
| NMN+ (authors' scoring) | simple | 0.750 ± 0.003 | 0.005 ± 0.001 | 0.256 ± 0.020 | 0.008 ± 0.000 |
| NMN+ (authors' scoring) | complex | 0.750 ± 0.001 | 0.004 ± 0.001 | 0.239 ± 0.003 | 0.007 ± 0.001 |

### Hard evaluation (authors' inference): argmax subject and reference, only the subject moves (mean ± sample std over seeds)

| Model | Split | IOU | IOU-M | Subject acc. | Placement acc. |
|---|---|---|---|---|---|
| Ours (neuro-symbolic) | overall | 0.941 ± 0.002 | 0.764 ± 0.005 | 1.000 ± 0.001 | 0.951 ± 0.008 |
| Ours (neuro-symbolic) | simple | 0.940 ± 0.002 | 0.762 ± 0.006 | 1.000 ± 0.000 | 0.951 ± 0.012 |
| Ours (neuro-symbolic) | complex | 0.942 ± 0.002 | 0.767 ± 0.005 | 0.999 ± 0.001 | 0.952 ± 0.004 |
| Parse-matched neural baseline (NMN+-like) | overall | 0.947 ± 0.003 | 0.789 ± 0.012 | 0.999 ± 0.001 | 0.942 ± 0.015 |
| Parse-matched neural baseline (NMN+-like) | simple | 0.947 ± 0.002 | 0.788 ± 0.010 | 1.000 ± 0.000 | 0.943 ± 0.013 |
| Parse-matched neural baseline (NMN+-like) | complex | 0.947 ± 0.003 | 0.789 ± 0.013 | 0.999 ± 0.001 | 0.941 ± 0.017 |
| End-to-end GRU baseline | overall | 0.751 ± 0.003 | 0.005 ± 0.001 | 0.265 ± 0.005 | 0.007 ± 0.001 |
| End-to-end GRU baseline | simple | 0.750 ± 0.004 | 0.005 ± 0.001 | 0.281 ± 0.018 | 0.008 ± 0.000 |
| End-to-end GRU baseline | complex | 0.751 ± 0.002 | 0.004 ± 0.001 | 0.249 ± 0.009 | 0.007 ± 0.001 |
| NMN+ (authors' scoring) | overall | 0.751 ± 0.003 | 0.005 ± 0.001 | 0.247 ± 0.011 | 0.007 ± 0.001 |
| NMN+ (authors' scoring) | simple | 0.750 ± 0.004 | 0.005 ± 0.001 | 0.256 ± 0.020 | 0.008 ± 0.000 |
| NMN+ (authors' scoring) | complex | 0.751 ± 0.002 | 0.004 ± 0.001 | 0.239 ± 0.003 | 0.007 ± 0.001 |

Total runtime: 311 s.
