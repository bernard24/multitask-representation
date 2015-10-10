
function [dataConf DataParameters] = makeMTLDataF(~, DataParameters) 

nInstances=DataParameters.m;
nAttrs=DataParameters.d;
nTasks=DataParameters.T;
nAtoms=DataParameters.K;
noise=DataParameters.sigma;
l2Norm=DataParameters.l2Norm;

Aux=randn(nAttrs);
[D, L, V]=svd(Aux);
D=D(:,1:nAtoms);

C=unitalizeColumns(randn(nAtoms, nTasks))*l2Norm;
W=D*C;

trainYCell=cell(1,nTasks);
trainXCell=cell(1,nTasks);
testYCell=cell(1,nTasks);
testXCell=cell(1,nTasks);
for i=1:nTasks
    X=unitalizeColumns(randn(nAttrs,nInstances))*sqrt(nAttrs);
    XTest=unitalizeColumns(randn(nAttrs,1000))*sqrt(nAttrs);
    Y=W(:,i)'*X;
    Y=sign(Y+randn(size(Y))*noise);
    YTest=W(:,i)'*XTest;
    YTest=sign(YTest+randn(size(YTest))*noise);

    trainYCell{i}=Y';
    trainXCell{i}=X;
    testYCell{i}=YTest';
    testXCell{i}=XTest;
end

dataConf.trainXCell=trainXCell;
dataConf.trainYCell=trainYCell;
dataConf.testXCell=testXCell;
dataConf.testYCell=testYCell;
dataConf.validation_testXCell=testXCell;
dataConf.validation_testYCell=testYCell;
dataConf.W=W;
dataConf.D=D;
dataConf.C=C;

end