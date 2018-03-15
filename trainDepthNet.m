function net = trainDepthNet
imdb = load('.\data\imdb_depth.mat') ;
imdb = imdb.imdb;
net = initializeLargeCNN;
vl_simplenn_display(net) ;


%%add custom loss l,ayers...
net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;
trainOpts.expDir = '.\data\depth-nets' ;
%trainOpts.gpus = [1] ;
trainOpts.batchSize = 64;% 32;
trainOpts.learningRate = 0.002 ;
trainOpts.plotDiagnostics = false ;
%trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics
trainOpts.numEpochs = 500 ;
trainOpts.errorFunction = 'none' ;

net = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Deploy: remove loss
net.layers(end) = [] ;
res = vl_simplenn(net, imdb.images.data(:,:,:,1)) ;
imshow(res(end).x,[])

end
