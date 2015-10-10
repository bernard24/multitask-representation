classdef RampedLossC
%GROUNDTRUTHLOSS Summary of this function goes here
%   Detailed explanation goes here

properties
    nOutputs=3;
end

methods
    function [score timeLong] = getScore(obj, method, data)
        actuY=data.testYCell;
        actuTrainY=data.trainYCell;
        nTasks=length(actuY);
        [method, predY timeLong]=test(method, data);
        predW=method.model.allW;

        predD=randn(size(data.D));
        predD=predD./repmat(sqrt(sum(predD.^2)), [size(data.D,1),1]);
        if isfield(method.model, 'D')
            predD=method.model.D;
        end
        
        Derror=sum(svd(predD'*data.D));
        
        testError=zeros(1,nTasks);
        trainError=zeros(1,nTasks);
        for i=1:length(actuY)
            testError(i)=mean( abs(sign(predY{i})-sign(actuY{i})) )/2;
            predTrainingY=data.trainXCell{i}'*predW(:,i);
            trainError(i)=mean( abs(sign(actuTrainY{i})-sign(predTrainingY)))/2;
        end
        
        score=[mean(testError), mean(trainError), Derror];
    end
end

end