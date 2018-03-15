function dx = myl1lossb(x,r,p)
k = p*(x-r);
dx = single(zeros(size(x)));
if (x-r)>0
    dx = 1;
else
    dx = -1;
end