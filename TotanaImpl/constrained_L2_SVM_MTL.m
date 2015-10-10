function [ W C D ] = constrained_L2_SVM_MTL( X, Y, K, alpha, nInnerIt, nOuterIt, epsilon)
%GMTSINF Summary of this function goes here
%   Detailed explanation goes here

nTasks=length(X);
nTasks2=length(Y);
if nTasks~=nTasks2
    [nTasks, nTasks2]
    error('josebi')
end
auxX=X{1};
nAttrs=size(auxX,1);
D=randn(nAttrs,K);
D=D./repmat(sqrt(sum(D.^2)), [nAttrs,1]);
C=zeros(K,nTasks);
A=eps*eye(K);

for t=1:nTasks
    Xt=D'*X{t};
    Yt=Y{t};
    ct=(Xt*Xt'+A)\Xt*Yt;
    C(:,t)=ct/norm(ct)*alpha;
end
margin=2*epsilon;

XY=cell(nTasks,1);
for i=1:nTasks
    XY{i}=X{i}.*repmat(Y{i}', [nAttrs,1]);
end

val=f( XY, C, D, margin );
for it=1:100000
    oldestVal=val;
    for t=1:nTasks
        XYt=D'*XY{t};

        ct=C(:,t);
        
        indices=XYt'*ct<margin;

        grad=-sum(XYt(:,indices),2)/margin;
        oldVal=fc( XYt, ct, margin );
        stepC=1;
        while true
            auxct=ct - stepC*grad;
            auxct=auxct/norm(auxct)*alpha;
            val=fc( XYt, auxct, margin );
            if val>oldVal
                if stepC<10^-8
                    auxct=ct;
                    break
                end
                stepC=stepC/2;
            else
                break
            end
        end
        C(:,t)=auxct;
    end

    dD=zeros(size(D));
    for t=1:nTasks
        XYt=XY{t};
        indices=XYt'*D*C(:,t)<margin;
        dD=dD-sum(XYt(:,indices),2)*C(:,t)';
    end
    dD=dD/margin;
    
    oldVal=f( XY, C, D, margin );
    stepD=1;
    while true
        auxD=D - stepD*dD;
        auxD=auxD./repmat(sqrt(sum(auxD.^2)), [nAttrs,1]);
        val=f( XY, C, auxD, margin );
        if val>oldVal
            if stepD<10^-8
                auxD=D;
                break
            end 
            stepD=stepD/2;
        else
            break
        end
    end
    D=auxD;
    val=f( XY, C, D, margin );
    if oldestVal<val+10^-9
        break;
    end
end
W=D*C;
end

function val = f( XY, C, D, margin )
nTasks=length(XY);
val=0;
for t=1:nTasks
    w=D*C(:,t);
    val=val+sum(max(0,1-(XY{t}'*w)/margin));
end
end

function val = fc( XYt, Ct, margin )
    val=sum(max(0,1-(XYt'*Ct)/margin));
end
