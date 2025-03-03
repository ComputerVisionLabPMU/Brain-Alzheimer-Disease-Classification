% load digit datasetdataset
digitDatasetPath = fullfile('D:\Brain\Augment_Data\Training');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
 [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');
%% design CNN
net=load('UpdatedNet1.mat'); %% for vgg16
net=net.net_1;



lgraph=layerGraph(net);



numClasses = numel(categories(imdsTrain.Labels));    
newFCLayer = fullyConnectedLayer(numClasses,'Name','NewFc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'fc_12',newFCLayer);
newClassLayer = softmaxLayer('Name','NewSoftmax');
lgraph = replaceLayer(lgraph,'softmax',newClassLayer);

newClassLayer1 = classificationLayer('Name','classification');
lgraph = addLayers(lgraph,newClassLayer1);
% lgraph = replaceLayer(lgraph,'output',newClassLayer1);
lgraph = connectLayers(lgraph,'NewSoftmax','classification');

  %% Augmenter
    augmenter = imageDataAugmenter( ...
        'RandRotation',[-5 5],'RandXReflection',1,...
        'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
    %% Resize training and testing data according to network
    auimds = augmentedImageDatastore([227 227 3],imdsTrain,'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);
    auimdsVali = augmentedImageDatastore([227 227 3],imdsValidation,'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);
    
 options = trainingOptions('sgdm',...
        'ExecutionEnvironment','gpu',...
        'MaxEpochs',40,'MiniBatchSize',64,...
        'Shuffle','every-epoch', ...
        'ValidationData',auimdsVali,...
        'InitialLearnRate',0.0001, ...
        'ValidationFrequency', 50, ...
        'Verbose',false, ...
        'Plots','training-progress');
% set training options


% training the network
% TrainedModifiedNet16 = trainNetwork(auimds, lgraph, options);
BrainAW_4block_40 = trainNetwork(auimds, lgraph, options);

% save('TrainedModifiedNet16','TrainedModifiedNet16');
save('BrainAW_4block_40','BrainAW_4block_40');