function [ A ] = unitalizeColumns( A )
for i=1:size(A,2)
    v=A(:,i);
    A(:,i)=v/sqrt(sum(v.^2));
end
end
