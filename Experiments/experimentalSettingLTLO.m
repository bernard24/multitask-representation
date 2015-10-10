clear all
here=pwd;
cd ../../Totana2
addpath(pwd);
cd (here)
cd ../TotanaImpl
addpath(pwd);
cd (here)
warning off


svm_itl=L2SVM_Independent_task_learning_LTL_experiment;
svm_ltl=L2SVM_Learning_to_learn_LTL_experiment;

scoreFunction=RampedLoss_LTL_C

methods={svm_itl, svm_ltl};
dataFunction=@makeMTLClassificationDataFVal_Transfer;
data=[];

dataParameters.m={5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150};%{20};%
dataParameters.d={50};
dataParameters.T={5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150}; %{100};%{160};%{500};%
dataParameters.targetT={50};
dataParameters.K={5}
dataParameters.sigma={0};
dataParameters.l2Norm={1};
dataParameters.l1Norm={1};

methodParameters.K={2};
methodParameters.alpha={ 1 };
storeName=[];
description='Learning to learn';

scriptO (methods, dataFunction, data, dataParameters, methodParameters, scoreFunction, storeName, description);%, seed);
