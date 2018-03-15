function weights = bilinearW(k,d,n)
%filterSize = [varargin{:}] ;
%scale = sqrt(2/prod(filterSize(1:3))) ;
%filters = ones(filterSize, 'single') * scale ;
filters = single(bilinear_u(k, d, n)) ;
biases = zeros(n,1,'single') ;
weights = {filters, biases} ;