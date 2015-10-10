classdef L2SVM_Independent_task_learning_MTL_experiment
    %METHOD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        parameters
        currentParameters
        model
        name='L2 Constrained SVM'
    end
    
    methods
        function obj=Method(parameters, name)
            obj.parameters=parameters;
            if nargin>1
                obj.name=name;
            end
        end
        
        function [obj, timeLong, localOutputs] = train(obj, data, ~)
            localOutputs=[];
              
            alpha=obj.currentParameters.alpha;
            K=obj.currentParameters.K;
              
            trainXCell=data.trainXCell;
            trainYCell=data.trainYCell;    
            
            tic;
            allW = constrained_L2_SVM( trainXCell, trainYCell, alpha, 10000, sqrt(K/length(trainYCell{1})));
            timeLong=toc;
            obj.model.allW=allW;
        end
        
        function [obj, predY, timeLong] = test(obj, data)
           if ~isfield(data, 'testXCell') || isempty(data.testXCell)
                predY=[];
                timeLong=0;
                return
            end
            testXCell=data.testXCell;
            predY=cell(1, length(testXCell));
            W=obj.model.allW;
            nAttrs=size(W,1);
            tic
            for t=1:length(testXCell)
                X=testXCell{t};
                predY{t}=X'*W(:,t);
            end
            timeLong=toc;
        end
        
        function obj=setParameters(obj, pars)
            nameParameters=fieldnames(pars);
            for i=1:length(nameParameters)
                name=nameParameters{i};
                if isfield(obj.parameters, name) && ~isempty(parameters.(name))
                    obj.currentParameters.(name)=obj.parameters.(name);
                else
                    obj.currentParameters.(name)=pars.(name);
                end
            end
        end
    end
    
end

