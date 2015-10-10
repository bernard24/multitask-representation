classdef L2SVM_Multitask_MTL_experiment 
 properties
        parameters
        currentParameters
        model
        name='L2 Constrained Hinge MTL'
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
            [allW, C, D]= constrained_L2_SVM_MTL( trainXCell, trainYCell, K, alpha, 10000, 10000, sqrt(K/length(trainYCell{1})));
            timeLong=toc;
            obj.model.allW=allW;
            obj.model.C=C;
            obj.model.D=D;
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
