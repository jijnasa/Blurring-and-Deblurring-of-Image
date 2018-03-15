function dx = BCELossBackward(x,r,p)
%r = reshape(r,[1,1,9216]);
dx = -(1/log(10))*p * ((r./x)-((1.-r)./(1.-x))) ;
dx = dx / (size(x,1) * size(x,2)) ;  % normalize by image size
