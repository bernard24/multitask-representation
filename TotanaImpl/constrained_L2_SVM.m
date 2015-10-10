function [ W ] = constrained_L2_SVM( X, Y, alpha, nIt, epsilon)
%GMTSINF Summary of this function goes here
%   Detailed explanation goes here

step=0.0001;

nTasks=length(X);
nTasks2=length(Y);
if nTasks~=nTasks2
    [nTasks, nTasks2]
    error('josebi')
end
auxX=X{1};
nAttrs=size(auxX,1);
W=zeros(nAttrs,nTasks);
A=eps*eye(nAttrs);
for t=1:nTasks
    previous_wt=Inf(nAttrs,1);
    Xt=X{t};
    Yt=Y{t};
    if size(Xt,1)~=nAttrs
        keyboard
    end
    wt=(Xt*Xt'+A)\Xt*Yt;
    wt=wt/norm(wt)*alpha;
        
    it=0;
    margin=2*epsilon;
    while true
        
        indices=find((Xt'*wt).*Yt<margin);
        if isempty(indices)
            break
        end
        grad=-Xt(:,indices)*Yt(indices)/(2*epsilon);
        wt=wt - step*grad;
        wt=wt/norm(wt)*alpha;
        
        if it>nIt || norm(previous_wt-wt)<10^-6*nAttrs
            break
        end
        previous_wt=wt;
        it=it+1;
    end
    W(:,t)=wt;
end

end

function opt = f( Xt, Yt, wt )

opt=norm(Yt-Xt'*wt)^2;

end
