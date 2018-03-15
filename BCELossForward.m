function y = BCELossForward(x,r)
%r = reshape(r,[1,1,9216]);
t1 = r.*(log(x)./log(10));
t2 = (1.-r).*(log(1.-x)./log(10));
t = (t1 + t2);
y = -sum(t(:));
y = y / (size(x,1) * size(x,2)) ;  % normalize by image size
