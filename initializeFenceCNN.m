function net = initializeFenceCNN()
%INITIALIZELARGECNN  Initialize a large CNN for text deblurring
%   NET = INITIALIZELARGECNN() returns the SimpleNN model NET.

net.meta.inputSize = [64 64 1 1] ;

net.layers = { } ;

net.layers{end+1} = struct(...
  'name', 'conv1', ...
  'type', 'conv', ...
  'weights', {xavier(5,5,1,32)}, ...
  'pad', 2, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu1', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv2', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,32,32)}, ...
  'pad', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu2', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv3', ...
  'type', 'conv', ...
  'weights', {xavier(1,1,32,32)}, ...
  'pad', [0 0 3 3], ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu3', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'conv4', ...
  'type', 'conv', ...
  'weights', {xavier(3,3,32,32)}, ...
  'pad', [3 3 0 0], ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;

net.layers{end+1} = struct(...
  'name', 'relu4', ...
  'type', 'relu') ;

net.layers{end+1} = struct(...
  'name', 'prediction', ...
  'type', 'conv', ...
  'weights', {xavier(5,5,32,1)}, ...
  'pad', 1, ...
  'stride', 1, ...
  'learningRate', [1 .001], ...
  'weightDecay', [1 0]) ;

% Consolidate the network, fixing any missing option
% in the specification above.

net = vl_simplenn_tidy(net) ;
