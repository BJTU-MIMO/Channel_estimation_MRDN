# Channel_estimation_MRDN

This simulation code package is mainly used to reproduce the results of the following paper [1]:

[1] Y. Jin, J. Zhang, C. Huang, L. Yang, H. Xiao, B. Ai and Z. Wang, "Multiple Residual Dense Networks for Reconfigurable Intelligent Surfaces Cascaded Channel Estimation," in IEEE Transactions on Vehicular Technology, vol. 71, no. 2, pp. 2134-2139, Feb. 2022.

If you use this simulation code package in any way, please cite the original paper [1] above. 

Reference: We highly respect reproducible research, so we try to provide the simulation codes for our published papers. 

## Abstract of the paper: 

Reconfigurable intelligent surface (RIS) constitutes an essential and promising paradigm that relies programmable wireless environment and provides capability for space-intensive communications, due to the use of low-cost massive reflecting elements over the entire surfaces of man-made structures. However, accurate channel estimation is a fundamental technical prerequisite to achieve the huge performance gains from RIS. By leveraging the low rank structure of RIS channels, three practical residual neural networks, named convolutional blind denoising network, convolutional denoising generative adversarial networks and multiple residual dense network, are proposed to obtain accurate channel state information, which can reflect the impact of different methods on the estimation performance. Simulation results reveal the evolution direction of these three methods and reveal their superior performance compared with existing benchmark schemes.

## Content of Code Package

The package generates the simulation results:

- `main_train`: Main function;
- `GAN_model`: Generate GAN-CBD Network;
- `Res_CD_Net`: Generate MRDN Network;;
- `functional`: The used function in the project;;

See each file for further documentation.

Enjoy the reproducible research!








