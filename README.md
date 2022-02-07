# SKIP_Non_Ergodic_Matlab
Matlab code for SKIP applied to non-ergodic seismic hazard prediction.
The code requires the GPML toolbox.

The main file is: Prediction_SKIP_Paper.m

The input example dataset is: 'CS15p4_600_events.csv' and is imported in the file: 'Imports_Data_California_100000.m'

The main file will compute the predicted conditioned median Equation (21) and its epistemic uncertainty from Equation (22) at the same latitude/longitude coordinates of the input dataset. 

The non-ergodic ground-motion model is taken from Landwehr (2016) and is described in Equation (12) of the paper. The hyperparameters of that model are fixed and are given in the main file.

The multiple covariance matrices involved in Equation (21) and (22) are approximated using SKIP.

The singular value decomposition is obtained by SKIP in the function 'get_SVD_K_SKIP_function.m'. First, the one-dimensional covariance matrices are approximated using SKI and their singular value decomposition (SVD) is obtained via the algorithm from Halko (2011) through the function 'Fast_SVD_Improved_function.m'. Then, the SVD of the multidimensional  matrices are obtained using the function 'Delta_mvm_SKIP_faster_function.m' following Equaiton (8).

The SVD of the multimensional covariance matrices can further be used to efficiently compute the median prediction and its epistemic uncertainty.
