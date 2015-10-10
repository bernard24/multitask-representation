
function [dataConf DataParameters] = makeMTLDataF(~, DataParameters) 

nInstances=DataParameters.m;
nAttrs=DataParameters.d;
nTasks=DataParameters.T;
nTargetTasks=DataParameters.targetT;
nAtoms=DataParameters.K;
noise=DataParameters.sigma;
l2Norm=DataParameters.l2Norm;

Aux=randn(nAttrs);
[D, L, V]=svd(Aux);
D=D(:,1:nAtoms);

C=unitalizeColumns(randn(nAtoms, nTasks))*l2Norm;
targetC=unitalizeColumns(randn(nAtoms, nTargetTasks))*l2Norm;
W=D*C;
targetW=D*targetC;

trainYCell=cell(1,nTasks);
trainXCell=cell(1,nTasks);
for i=1:nTasks
    X=unitalizeColumns(randn(nAttrs,nInstances))*sqrt(nAttrs);
    Y=W(:,i)'*X;
    Y=sign(Y+randn(size(Y))*noise);

    trainYCell{i}=Y';
    trainXCell{i}=X;
end

targetTrainYCell=cell(1,nTargetTasks);
targetTrainXCell=cell(1,nTargetTasks);
targetTestYCell=cell(1,nTargetTasks);
targetTestXCell=cell(1,nTargetTasks);
for i=1:nTargetTasks
    X=unitalizeColumns(randn(nAttrs,nInstances))*sqrt(nAttrs);
    XTest=unitalizeColumns(randn(nAttrs,1000))*sqrt(nAttrs);
    Y=targetW(:,i)'*X;
    Y=sign(Y+randn(size(Y))*noise);
    YTest=targetW(:,i)'*XTest;
    YTest=sign(YTest+randn(size(YTest))*noise);

    targetTrainYCell{i}=Y';
    targetTrainXCell{i}=X;
    targetTestYCell{i}=YTest';
    targetTestXCell{i}=XTest;
end



dataConf.trainXCell=trainXCell;
dataConf.trainYCell=trainYCell;
dataConf.targetTrainXCell=targetTrainXCell;
dataConf.targetTrainYCell=targetTrainYCell;
dataConf.targetTestXCell=targetTestXCell;
dataConf.targetTestYCell=targetTestYCell;
dataConf.W=W;
dataConf.D=D;
dataConf.C=C;
dataConf.targetW=targetW;
dataConf.targetC=targetC;

end