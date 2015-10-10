clear all
here=pwd;
cd ../../Totana2
addpath(pwd);
cd (here)
cd ../TotanaImpl
addpath(pwd);
cd (here)
warning off

svm_itl=L2SVM_Independent_task_learning_MTL_experiment;
svm_mtl=L2SVM_Multitask_MTL_experiment;

scoreFunction=RampedLossC

methods={svm_itl, svm_mtl};
dataFunction=@makeMTLClassificationDataFVal;
data=[];

dataParameters.m={5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150};
dataParameters.d={50};
dataParameters.T={5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150};
dataParameters.K={2};
dataParameters.sigma={1.5};
dataParameters.l2Norm={1};
dataParameters.l1Norm={1};

methodParameters.K={2};
methodParameters.alpha={ 1 };
storeName=[];
description='Multi-Task Learning';

scriptO (methods, dataFunction, data, dataParameters, methodParameters, scoreFunction, storeName, description);
