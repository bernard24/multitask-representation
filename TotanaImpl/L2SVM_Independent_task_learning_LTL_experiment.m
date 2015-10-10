classdef L2SVM_Independent_task_learning_LTL_experiment
    
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
            timeLong=0;
        end
        
        function [obj, predY, timeLong] = test(obj, data)
           if ~isfield(data, 'targetTrainXCell') || isempty(data.targetTrainXCell)
                predY=[];
                timeLong=0;
                return
           end
           targetTrainXCell=data.targetTrainXCell;
           targetTrainYCell=data.targetTrainYCell;
            alpha=obj.currentParameters.alpha;
            K=obj.currentParameters.K;
            
            tic
            W = constrained_L2_SVM( targetTrainXCell, targetTrainYCell, alpha, 10000, sqrt(K/length(targetTrainYCell{1})));

            testXCell=data.targetTestXCell;
            predY=cell(1, length(testXCell));
            
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

