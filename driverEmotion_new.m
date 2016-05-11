% Load training and test data using imageSet.
syntheticDir   = fullfile(toolboxdir('ADS'), 'Train'); 
handwrittenDir = fullfile(toolboxdir('ADS'), 'Test');



% imageSet recursively scans the directory tree containing the images.
trainingSet = imageSet(syntheticDir,   'recursive');
testSet     = imageSet(handwrittenDir, 'recursive');
 
img = read(trainingSet(1), 1);

% Extract HOG features and HOG visualization
[hog_1x1, vis1x1] = extractHOGFeatures(img,'CellSize',[1 1]);
[hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);

% Show the original image
figure;
subplot(2,3,1:3); imshow(img);

cellSize = [2 2];
hogFeatureSize = length(hog_2x2);

% The trainingSet is an array of 10 imageSet objects: one for each digit.
% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.

trainingFeatures = [];
trainingLabels   = [];

for digit = 1:numel(trainingSet)

    %numImages = trainingSet(digit).Count
    numImages = 390;
    features  = zeros(numImages, hogFeatureSize, 'single');
    for i = 1:numImages
       
        
        img = rgb2gray(read(trainingSet(digit), i));
        
        
        % Apply pre-processing steps
        img = im2bw(img);
        %ndims(img)
        %ndims(extractHOGFeatures(img, 'CellSize', cellSize))
       
        s = size(extractHOGFeatures(img, 'CellSize', cellSize));
        image_Size = size(img);
        if (image_Size(1) == 151 && image_Size(2) == 151)
            
        features(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
        end
    end

    % Use the imageSet Description as the training labels. The labels are
    % the digits themselves, e.g. '0', '1', '2', etc.
    labels = repmat(trainingSet(digit).Description, numImages, 1);
    trainingFeatures = [trainingFeatures; features];   %#ok<AGROW>
    trainingLabels   = [trainingLabels;   labels  ];   %#ok<AGROW>

end

%fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
resp = strcmp(cellstr(trainingLabels),'safe_t');
classifier = fitcecoc(trainingFeatures, resp,'FitPosterior',1,'Coding', 'onevsone');
partModel = crossval(classifier, 'KFold', 10);
% 
% [validationPredictions, validationScores] = kfoldPredict(partModel);
% [validationPredictions, validationScores] = predict(partModel);

[label,NegLoss,PBScore,Posterior] = kfoldPredict(partModel);
confmat=confusionmat(resp,label);     

[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(resp,Posterior(:,classifier.ClassNames),'true');

% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);

% Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

helperDisplayConfusionMatrix(confMat)
