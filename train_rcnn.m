%% Train R-CNN Iris and Pupil Detector
% Written by Michael Soskind

%%
% Load training data and network layers

% Removing previous pupil and iris data
clear pupils irises; clc;

% Creating a fileID for the metadata
filename = 'metadata.txt';
fileID = fopen(filename);

% Reading in metadata from the text file
columns = textscan(fileID, '%s %s %s %s %s %s %s', 1);
data = textscan(fileID, '%s %f %f %f %f %f %f');

%%
% Making the new pupil/iris metadata data (for R-CNN)
imageFilenames = data{1};
for i = 1:length(data{1})
    pupils{i} = [data{2}(i)-data{4}(i), data{3}(i)-data{4}(i), 2*data{4}(i), 2*data{4}(i)];
    irises{i} = [data{5}(i)-data{7}(i), data{6}(i)-data{7}(i), 2*data{7}(i), 2*data{7}(i)];
end

% Correcting the dimensions of the pupils and irises
pupils = pupils';
irises = irises';

% Creating the tables for pupils and irises
table_p = table(imageFilenames, pupils);
table_i = table(imageFilenames, irises);

%%
% Defining the neural network design
convLayer = convolution2dLayer(5,10,'Padding',2,'BiasLearnRateFactor',2);
convLayer.Weights = randn([5 5 3 10])*0.0001;
convLayer.Bias = randn([1 1 10])*0.00001+1;
% output = classificationLayer('Name','coutput');
% output.ClassNames = cellstr(['p ';'np']);
objectClasses = {'purpil'};
numClassesPlusBackground = numel(objectClasses) + 1;
layers1 = [ 
    imageInputLayer([250 250 3])
    
    convolution2dLayer(5,64)   % convLayer defined above
    reluLayer()
    convolution2dLayer(5,64)   % convLayer defined above
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32)
    reluLayer()
    convolution2dLayer(3,32)
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(numClassesPlusBackground)
    softmaxLayer()
    classificationLayer()
];
% load('rcnnStopSigns.mat', 'stopSigns', 'layers')
% myLayers = [
%     imageInputLayer([250,250,3])
%     layers(2:end)
%     ];
%%
% Set network training options to use mini-batch size of 32 to reduce GPU
% memory usage. Lower the InitialLearningRate to reduce the rate at which
% network parameters are changed. This is beneficial when fine-tuning a
% pre-trained network and prevents the network from changing too rapidly. 
options = trainingOptions('sgdm', ...
  'MiniBatchSize', 16, ...
  'InitialLearnRate', 1e-6, ...
  'MaxEpochs', 10);

%%
% Train the R-CNN detector. Training can take a few minutes to complete.
rcnn_p = trainRCNNObjectDetector(table_p, layers1, options, 'NegativeOverlapRange', [0 0.3]);

%%
% Test the R-CNN detector on a test image.
img = imread('test_eye.jpg'); 
% img = imread('eye_01.jpg');
img_diff = imresize(img, [250 250]);

[bbox, score, label] = detect(rcnn_p, img, 'MiniBatchSize', 16);
%%
% Display strongest detection result.
[score, idx] = max(score);

bbox = bbox(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);

detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);

figure();
imshow(detectedImg)