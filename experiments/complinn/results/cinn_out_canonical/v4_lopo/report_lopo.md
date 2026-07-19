# CINN v4 — leave-one-POLICY-out (LOPO)

Generalisation to an UNSEEN policy. Supervision = compliant (policy-prescribed action). Policy enters X as a descriptor (5x|A| prescription table); resolved denial mask is used only in loss/eval, never in X. Held-out policy descriptor is 100% novel.

Rows per policy: 1953. in_dim=36 (base 16 + descriptor 20). Seeds: 2. Actions: ['recommend', 'disclose', 'withhold', 'escalate'].

`prescribed_acc` = fraction predicting the policy-prescribed action on the held-out policy (utility). `viol` = mass argmaxed onto a non-prescribed action (= 1 - prescribed_acc, single-allowed case).

## held out: p01  (train on p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0085 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0046 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 80.6% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 80.6%

## held out: p02  (train on p01, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0053 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0033 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 80.1% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 80.1%

## held out: p03  (train on p01, p02, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0092 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0080 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 60.7% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 60.7%

## held out: p04  (train on p01, p02, p03, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0034 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0032 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 79.4% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 79.4%

## held out: p05  (train on p01, p02, p03, p04, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0169 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0088 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 60.0% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 60.0%

## held out: p06  (train on p01, p02, p03, p04, p05, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0113 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0057 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 59.4% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 59.4%

## held out: p07  (train on p01, p02, p03, p04, p05, p06, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0238 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0187 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 59.9% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 59.9%

## held out: p08  (train on p01, p02, p03, p04, p05, p06, p07, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0058 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0024 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 79.7% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 79.7%

## held out: p09  (train on p01, p02, p03, p04, p05, p06, p07, p08, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0133 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0088 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 60.4% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 60.4%

## held out: p10  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0188 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0107 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 59.8% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 59.8%

## held out: p11  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0138 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0044 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 59.5% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 59.5%

## held out: p12  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0129 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0073 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 59.1% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 59.1%

## held out: p13  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0093 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0046 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 60.3% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 60.3%

## held out: p14  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0184 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0081 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 60.8% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 60.8%

## held out: p15  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0035 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0015 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 80.2% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 80.2%

## held out: p16  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0047 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0028 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 80.2% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 80.2%

## held out: p17  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0127 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0062 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 60.8% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 60.8%

## held out: p18  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0249 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0191 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 60.3% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 60.3%

## held out: p19  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0134 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0077 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 59.1% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 59.1%

## held out: p20  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0119 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0056 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 59.5% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 59.5%

## held out: p21  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p22, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0138 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0070 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 59.8% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 59.8%

## held out: p22  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p23, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0230 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0093 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 60.4% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 60.4%

## held out: p23  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p24, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0032 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0015 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 79.7% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 79.7%

## held out: p24  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p25, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0113 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0058 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 59.9% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 59.9%

## held out: p25  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p26, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0128 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0107 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 59.4% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 59.4%

## held out: p26  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p27, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0210 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0135 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 60.0% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 60.0%

## held out: p27  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p28, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0060 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0029 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 79.4% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 79.4%

## held out: p28  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p29, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0079 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0049 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 60.7% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 60.7%

## held out: p29  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p30)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0036 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0016 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 80.1% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 80.1%

## held out: p30  (train on p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29)

| lambda | prescribed_acc | violation | denied_prob |
|---|---|---|---|
| 0.0 | 100.0±0.0% | 0.0±0.0% | 0.0060 |
| 4.0 | 100.0±0.0% | 0.0±0.0% | 0.0059 |

Ablation controls @lambda=4.0 (prescribed_acc):
- full: **100.0%**
- role_data_ablation: 80.6% (zero role+data -> cannot select cell)
- descriptor_ablation: 0.0% (zero descriptor -> cannot read unseen policy)
- majority-class baseline: 80.6%

## reading the result
- Full model transfers AND both ablations collapse toward majority -> genuine policy generalisation (cell selection + descriptor read are both load-bearing).
- Full model transfers BUT an ablation also transfers -> that signal was a shortcut; the claim is NOT supported. Investigate before any patent/report language.
- INTEGRITY: policies here are AUTHORED and SYNTHETIC. Claim is 'generalises to an unseen policy expressed over the action vocabulary', on a toy 4x64 classifier. Not evidence about the real caid_v1.json policy, and not an LLM result (that is Priority 2).