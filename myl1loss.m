function y = myl1loss(x,r)
delta = abs(x-r);
y = sum(delta(:));
y = y/(size(x,1)*size(x,2));