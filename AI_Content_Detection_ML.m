% Machine Learning Model for Ai-content Detection in Students’ Assignments
% at Dkut By ERIC MULWA BSc Eng
close all
clc
clear all
% Defining the CNN architecture
layers = [
    imageInputLayer([224 224 3])
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
% Setting training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 0.002, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false);
% Loading and preprocessing the training dataset
imds = imageDatastore('D:\Training Dataset', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
% Resizing images to the desired input size
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
% Spliting the dataset into training and validation sets
[trainSet, valSet] = splitEachLabel(imds, 0.8, 'randomized');
% Training the CNN model
net = trainNetwork(trainSet, layers, options);
% Evaluating the trained model on the validation set
predictions = classify(net, valSet)
accuracy = mean(predictions == valSet.Labels)
% Saving the trained model for future use
save('Dkut_Ai_detection_model.mat', 'net');
% Helper function to preprocess the images
function img = readAndPreprocessImage(filename)
    img = imread(filename);
    img = imresize(img, [224 224]);
end


