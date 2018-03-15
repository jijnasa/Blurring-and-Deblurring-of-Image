function net = initializeSmallCNN()
%INITIALIZESMALLCNN  Initialize a small CNN for text deblurring
%   NET = INITIALIZESMALLCNN() returns the SimpleNN model NET.

net.meta.inputSize = [96 96 3 1] ;

net.layers = { } ;

net.layers{end+1} = struct(...
  'name', 'conv1', ...
  'type', 'conv', ...
  'weights', {xavier(5,5,3,32)}, ...
  'pad', 2, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu1', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv2', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,32,64)}, ...
  'pad', 1, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu2', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv3', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,64,64)}, ...
  'pad', 1, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu3', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv4', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,64,128)}, ...
  'pad', 1, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu4', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'lastlayer', ...
  'type', 'conv', ...
  'weights', {xavier(1,1,128,1)}, ...
  'pad', 0, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'sigmoid', ...
  'type', 'sigmoid') ;
% Consolidate the network, fixing any missing option
% in the specification above

net = vl_simplenn_tidy(net) ;