
Implementation of the proposed methods described in [Vectorial total generalized variation for accelerated multi-channel multi-contrast MRI](https://www.sciencedirect.com/science/article/abs/pii/S0730725X16300704)
and my Ph.D. thesis entitled [Acquisition and reconstruction methods for magnetic resonance imaging](https://dspace.mit.edu/handle/1721.1/105570).

Overview
------
**Purpose**
To develop and implement an efficient reconstruction technique to improve accelerated multi-channel multi-contrast MRI.

**Theory and Methods**
The vectorial total generalized variation (TGV) operator is used as a regularizer for the sensitivity encoding (SENSE) technique to improve image quality of multi-channel multi-contrast MRI. The alternating direction method of multipliers (ADMM) is used to efficiently reconstruct the data. The performance of the proposed method (MC-TGV-SENSE) is assessed on two healthy volunteers at several acceleration factors.

**Results**
As demonstrated on the in vivo results, MC-TGV-SENSE had the lowest root-mean-square error (RMSE), highest structural similarity index, and best visual quality at all acceleration factors, compared to other methods under consideration. MC-TGV-SENSE yielded up to 17.3% relative RMSE reduction compared to the widely used total variation regularized SENSE. Furthermore, we observed that the reconstruction time of MC-TGV-SENSE is reduced by approximately a factor of two with comparable RMSEs by using the proposed ADMM-based algorithm as opposed to the more commonly used Chambolle–Pock primal-dual algorithm for the TGV-based reconstruction.

**Conclusion**
MC-TGV-SENSE is a better alternative than the existing reconstruction methods for accelerated multi-channel multi-contrast MRI. The proposed method exploits shared information among the images (MC), mitigates staircasing artifacts (TGV), and uses the encoding power of multiple receiver coils (SENSE).

Main Files
------
* **script_TSE3Contrasts.m** is the main file.

* **MC_TGV_SENSE_SB.m** contains the ADMM implementation of multi-channel MRI with multi-contrast TGV regularization.

* **MC_TV_SENSE_SB.m** contains the ADMM implementation of multi-channel MRI with multi-contrast TV regularization.

The data files are too big to be included in this repository. Please contact me directly for example data.

