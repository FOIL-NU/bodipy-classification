clear;
% Resolve the data directory relative to this script's location
% (repo_root/src/original_single_split/<this file>), so it runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
data_dir = fullfile(repo_root, 'data', 'processed_combined');
% Figures/matrices are likewise written under repo_root, not the
% caller's current directory, so output lands in the same place
% regardless of where this script is invoked from.
output_dir = fullfile(repo_root, 'results', 'original_single_split', 'pseudomix');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

target_bodipy = 1:7;
target_bodipy = [target_bodipy, target_bodipy + 8];

wl_min = 600;
wl_max = 700;

cutoff_val = 1000;

p_value = zeros(length(target_bodipy));
misidentification_sc = nan(length(target_bodipy));
accuracy_matrix_cnn = nan(length(target_bodipy));
precision_matrix_cnn = nan(length(target_bodipy));
recall_matrix_cnn = nan(length(target_bodipy));

n_molecules_total = zeros(length(target_bodipy),1);
mean_spectral_centroid = zeros(length(target_bodipy),1);

% set the random number generator
rng(1);

% plot the n_photons vs. spectral centroids
for i_bodipy = 1:length(target_bodipy)
    i_curr_bodipy = target_bodipy(i_bodipy);
    load(fullfile(data_dir, sprintf('bodipy%d.mat',i_curr_bodipy)));
    i_n_photons = n_photons_combined;
    i_spectral_rois = spectral_rois_combined;
    i_spectral_centroid = spectral_centroids_combined;
    i_spectral_profiles = spectral_profiles_combined;

    % sort the spectral profiles by n_photons, descending
    [i_n_photons, sort_idx] = sort(i_n_photons,'descend');
    i_spectral_rois = i_spectral_rois(:,:,sort_idx);
    i_spectral_centroid = i_spectral_centroid(sort_idx);
    i_spectral_profiles = i_spectral_profiles(:,sort_idx);

    % cutoff the data at n_photons = cutoff_val
    i_spectral_rois = i_spectral_rois(:,:,i_n_photons > cutoff_val);
    i_spectral_profiles = i_spectral_profiles(:,i_n_photons > cutoff_val);
    i_spectral_centroid = i_spectral_centroid(i_n_photons > cutoff_val);
    i_n_photons = i_n_photons(i_n_photons > cutoff_val);

    % remove outliers of the spectral centroids
    [i_spectral_centroid,i_rm_rows] = rmoutliers(i_spectral_centroid,'mean','thresholdfactor',4);
    % update the spectral profiles
    i_spectral_rois = i_spectral_rois(:,:,~i_rm_rows);
    i_spectral_profiles = i_spectral_profiles(:,~i_rm_rows);
    i_n_photons = i_n_photons(~i_rm_rows);
    
    n_molecules_total(i_bodipy) = length(i_n_photons);
    mean_spectral_centroid(i_bodipy) = mean(i_spectral_centroid);
    
    for j_bodipy = (1+i_bodipy):length(target_bodipy)
        j_curr_bodipy = target_bodipy(j_bodipy);
        load(fullfile(data_dir, sprintf('bodipy%d.mat',j_curr_bodipy)));
        j_n_photons = n_photons_combined;
        j_spectral_rois = spectral_rois_combined;
        j_spectral_centroid = spectral_centroids_combined;
        j_spectral_profiles = spectral_profiles_combined;

        % sort the spectral profiles by n_photons, descending
        [j_n_photons, sort_idx] = sort(j_n_photons,'descend');
        j_spectral_rois = j_spectral_rois(:,:,sort_idx);
        j_spectral_centroid = j_spectral_centroid(sort_idx);
        j_spectral_profiles = j_spectral_profiles(:,sort_idx);

        % cutoff the data at n_photons = cutoff_val
        j_spectral_rois = j_spectral_rois(:,:,j_n_photons > cutoff_val);
        j_spectral_profiles = j_spectral_profiles(:,j_n_photons > cutoff_val);
        j_spectral_centroid = j_spectral_centroid(j_n_photons > cutoff_val);
        j_n_photons = j_n_photons(j_n_photons > cutoff_val);

        % remove outliers of the spectral centroids
        [j_spectral_centroid,j_rm_rows] = rmoutliers(j_spectral_centroid,'mean','thresholdfactor',4);
        % update the spectral profiles
        j_spectral_rois = j_spectral_rois(:,:,~j_rm_rows);
        j_spectral_profiles = j_spectral_profiles(:,~j_rm_rows);
        j_n_photons = j_n_photons(~j_rm_rows);
        
        % perform the p-value test (two-sample t-test)
        [~,p] = ttest2(i_spectral_centroid,j_spectral_centroid);
        p_value(i_bodipy,j_bodipy) = p;

        % perform classification test based on spectral centroids
        % we select the number of bodipy molecules in the mean spectral centroid for each dye, plus minus 3
        sc_window = 3;
        % first we check if the mean spectral centroid is within the sc_window of the other dye
        if abs(mean(i_spectral_centroid) - mean(j_spectral_centroid)) < sc_window
            % if it is, we misidentify the dyes
            misidentification_sc(i_bodipy,j_bodipy) = 1;
        else
            % if it isn't, we count the number of dyes with spectral centroid in each window
            i_sc_window = mean(i_spectral_centroid) + [-sc_window, sc_window];
            j_sc_window = mean(j_spectral_centroid) + [-sc_window, sc_window];

            % total number of molecules
            n_molecules = length(i_spectral_centroid) + length(j_spectral_centroid);

            % count the number of molecules in each window
            n_i_sc_window = sum(j_spectral_centroid > i_sc_window(1) & j_spectral_centroid < i_sc_window(2));
            n_j_sc_window = sum(i_spectral_centroid > j_sc_window(1) & i_spectral_centroid < j_sc_window(2));

            % calculate the fraction of misidentified molecules
            misidentification_sc(i_bodipy,j_bodipy) = (n_i_sc_window + n_j_sc_window)/n_molecules;
        end
        
        % perform classification test based on CNN
        % 400 brightest images for training, 100 images for testing        
        
        % check that we have at least 600 spectral rois for each dye
        if size(i_spectral_rois,3) < 600 || size(j_spectral_rois,3) < 600
            accuracy_matrix_cnn(i_bodipy,j_bodipy) = nan;
            precision_matrix_cnn(i_bodipy,j_bodipy) = nan;
            recall_matrix_cnn(i_bodipy,j_bodipy) = nan;

        else
            % % take the brightest 500 images for each dye
            % i_spectral_rois = i_spectral_rois(:,:,1:500);
            % j_spectral_rois = j_spectral_rois(:,:,1:500);

            % scramble the indices for training and testing
            i_idx = randperm(size(i_spectral_rois,3),600);
            j_idx = randperm(size(j_spectral_rois,3),600);

            i_train_rois = i_spectral_rois(:,:,i_idx(1:420));
            i_validate_rois = i_spectral_rois(:,:,i_idx(421:510));
            i_test_rois = i_spectral_rois(:,:,i_idx(511:end));
            j_train_rois = j_spectral_rois(:,:,j_idx(1:420));
            j_validate_rois = j_spectral_rois(:,:,j_idx(421:510));
            j_test_rois = j_spectral_rois(:,:,j_idx(511:end));

            train_labels = [ones(420,1); 2*ones(420,1)];
            validate_labels = [ones(90,1), 2*ones(90,1)];
            test_labels = [ones(90,1); 2*ones(90,1)];

            train_idx = randperm(840);
            validate_idx = randperm(180);
            test_idx = randperm(180);

            X_train = cat(3,i_train_rois,j_train_rois);
            X_validate = cat(3,i_validate_rois,j_validate_rois);
            X_test = cat(3,i_test_rois,j_test_rois);

            X_train = X_train(:,:,train_idx);
            X_validate = X_validate(:,:,validate_idx);
            X_test = X_test(:,:,test_idx);
            
            Y_train = train_labels(train_idx);
            Y_validate = validate_labels(validate_idx);
            Y_test = test_labels(test_idx);

            training_labels = categorical(Y_train);
            validation_labels = categorical(Y_validate);
            testing_labels = categorical(Y_test);
            % permute the dimensions of the training and testing images
            training_images = permute(X_train,[1 2 4 3]);
            validation_images = permute(X_validate,[1 2 4 3]);
            testing_images = permute(X_test,[1 2 4 3]);

            sz = size(training_images);
            n_labels = 2;
            
            % Create the input layer
            inputLayer = imageInputLayer([sz(1),sz(2),sz(3)],'Name','input');

            % Branch 1: Convolutional path to extract spectral features
            convLayers = [
                convolution2dLayer(3,16,'Padding','same','Name','conv1')
                batchNormalizationLayer('Name','bn1')
                reluLayer('Name','relu1')

                maxPooling2dLayer(2,'Stride',2,'Name','pool1')

                convolution2dLayer(3,32,'Padding','same','Name','conv2')
                batchNormalizationLayer('Name','bn2')
                reluLayer('Name','relu2')

                maxPooling2dLayer(2,'Stride',2,'Name','pool2')

                dropoutLayer(0.5,'Name','dropout')

                fullyConnectedLayer(32,'Name','fc_conv')];

            % Branch 2: Direct path to retain positional information
            flattenInputLayer = [
                averagePooling2dLayer([1,5],'Name','gap_input')  % Average pooling along dimension 2 to collapse to 1D.
                fullyConnectedLayer(32,'Name','fc_input')];  % Fully connected layer after pooling

            % Concatenate features from both branches
            concatLayer = concatenationLayer(1,2,'Name','concat');

            % Final classification layers
            finalLayers = [
                reluLayer('Name','relu_concat')
                fullyConnectedLayer(n_labels,'Name','fc_final')
                softmaxLayer('Name','softmax')
                classificationLayer('Name','classOutput')];

            % Assemble the layer graph
            lgraph = layerGraph(inputLayer);
            lgraph = addLayers(lgraph, convLayers);
            lgraph = addLayers(lgraph, flattenInputLayer);
            lgraph = addLayers(lgraph, concatLayer);
            lgraph = addLayers(lgraph, finalLayers);

            % Connect layers
            lgraph = connectLayers(lgraph,'input','conv1');
            lgraph = connectLayers(lgraph,'input','gap_input');
            lgraph = connectLayers(lgraph,'fc_conv','concat/in1');
            lgraph = connectLayers(lgraph,'fc_input','concat/in2');
            lgraph = connectLayers(lgraph,'concat','relu_concat');
            
            augmenter = imageDataAugmenter( ...
                'RandXReflection', true, ...
                'RandYReflection', false);
            
            % Create an augmentedImageDatastore for training data
            augmentedTrainingData = augmentedImageDatastore([sz(1), sz(2), sz(3)], training_images, training_labels, 'DataAugmentation', augmenter);
            validationData = augmentedImageDatastore([sz(1), sz(2), sz(3)], validation_images, validation_labels);
            testData = augmentedImageDatastore([sz(1), sz(2), sz(3)], testing_images, testing_labels);
            
            options = trainingOptions('sgdm', ...
                'InitialLearnRate',0.001, ...
                'MaxEpochs',200, ...
                'Shuffle','every-epoch', ...
                'ValidationData',validationData, ...
                'ValidationFrequency',30, ...
                'Verbose',false);
            
            net = trainNetwork(augmentedTrainingData,lgraph,options);
            % Predict class labels for the validation set (using unaugmented validation images)
            test_predicted = classify(net, testData);  % Using the augmentedImageDatastore for validation data

            % calculate the confusion matrix
            confusion_matrix = confusionmat(testing_labels,test_predicted);

            % calculate the precision and recall
            [accuracy,precision,recall] = calculate_metrics(confusion_matrix);

            accuracy_matrix_cnn(i_bodipy,j_bodipy) = accuracy;
            precision_matrix_cnn(i_bodipy,j_bodipy) = precision;
            recall_matrix_cnn(i_bodipy,j_bodipy) = recall;
        end
    end
    % plot the spectral centroids
    f1 = figure(1); clf;
    f1.Position(3:4) = [600,300];
    h = histogram(i_spectral_centroid,630:680,'facecolor','#E6194B', ...
        'Normalization','probability','displayname',sprintf('bodipy %d',i_curr_bodipy)); % plot the histogram of the spectral centroids
    set(h,'FaceAlpha',1)
    % plot the mean spectral centroids, without displaying the legend
    xline(mean(i_spectral_centroid),'color','#E6194B','LineWidth',2,'HandleVisibility','off');

    % plot the legend
%         legend('location','best');
    xlabel('Spectral centroid (nm)');
    ylabel('Normalized counts');

    formatfig(f1,'presentation',fullfile(output_dir,sprintf('bodipy%d.png',i_curr_bodipy)));
    formatfig(f1,'presentation',fullfile(output_dir,sprintf('bodipy%d.eps',i_curr_bodipy)));
end

% fix the matrices to be symmetric
p_value = p_value + p_value';
misidentification_sc = misidentification_sc';
accuracy_matrix_cnn = accuracy_matrix_cnn';
precision_matrix_cnn = precision_matrix_cnn';
recall_matrix_cnn = recall_matrix_cnn';

misidentification_sc(misidentification_sc > 0.5) = 0.5;
accuracy_matrix_cnn(accuracy_matrix_cnn < 0.5) = 0.5;
precision_matrix_cnn(precision_matrix_cnn < 0.5) = 0.5;
recall_matrix_cnn(recall_matrix_cnn < 0.5) = 0.5;

% set the diagonal to 1
p_value(1:length(target_bodipy)+1:end) = nan;
misidentification_sc(1:length(target_bodipy)+1:end) = 1;
accuracy_matrix_cnn(1:length(target_bodipy)+1:end) = nan;
precision_matrix_cnn(1:length(target_bodipy)+1:end) = nan;
recall_matrix_cnn(1:length(target_bodipy)+1:end) = nan;

% get the f1 score
f1_score_cnn = 2*precision_matrix_cnn.*recall_matrix_cnn./(precision_matrix_cnn + recall_matrix_cnn);

%% plotting the results
% plot the p-values as a heatmap
f2 = figure(2); clf;
p_value_plot = log10(p_value);
p_value_plot(p_value_plot < -3) = -3;
imagesc(p_value_plot);
set(gca,'XTick',1:length(target_bodipy));
set(gca,'XTickLabel',target_bodipy);
set(gca,'YTick',1:length(target_bodipy));
set(gca,'YTickLabel',target_bodipy);
% create a colorbar, with worst values in red and best values in green
cmap_g = [161, 230, 170] ./ 255;
cmap_y = [255, 247, 172] ./ 255;
cmap_r = [255, 153, 179] ./ 255;

intervals1 = linspace(0,1,2);
intervals2 = linspace(0,1,3);
customcmap = [0.5,0.5,0.5; ...
              interp1([0; 1],[cmap_g; cmap_y],intervals1(1:end-1)); ...
              interp1([0; 1],[cmap_y; cmap_r],intervals2)];
colormap(customcmap);            
% cb = colorbar();
caxis([-4,-.8]);

% label the values on the heatmap that are significant with *, **, and ***
% (p < 0.05, 0.01, and 0.001, respectively)
for i_bodipy = 1:length(target_bodipy)
    for j_bodipy = 1:length(target_bodipy)
        if i_bodipy == j_bodipy
            text(j_bodipy,i_bodipy,'-','HorizontalAlignment','center','VerticalAlignment','middle','FontSize',16);
        elseif p_value(i_bodipy,j_bodipy) < 0.001
            text(j_bodipy,i_bodipy,'***','HorizontalAlignment','center','VerticalAlignment','middle','FontSize',16);
        elseif p_value(i_bodipy,j_bodipy) < 0.01
            text(j_bodipy,i_bodipy,'**','HorizontalAlignment','center','VerticalAlignment','middle','FontSize',16);
        elseif p_value(i_bodipy,j_bodipy) < 0.05
            text(j_bodipy,i_bodipy,'*','HorizontalAlignment','center','VerticalAlignment','middle','FontSize',16);
        else
            text(j_bodipy,i_bodipy,'ns','HorizontalAlignment','center','VerticalAlignment','middle','FontSize',12);
        end
    end
end
title('p-values');

% %colorbar ticks
% cb.Ticks = linspace(-4,-.8,11);
% cb.TickLabels = {'','','','***','','**','','*','','ns',''};

formatfig(f2,'presentation',fullfile(output_dir,'ttest_heatmap.eps'));
formatfig(f2,'presentation',fullfile(output_dir,'ttest_heatmap.png'));

% plot the difference in the spectral centroids
f3 = figure(3); clf;
X = abs(mean_spectral_centroid - mean_spectral_centroid');
X = X + triu(nan(length(X)),1);
imagesc(X);
xlim([0.5,length(target_bodipy)-0.5]);
set(gca,'XTick',1:length(target_bodipy)-1);
set(gca,'XTickLabel',target_bodipy(1:end-1));
ylim([1.5,length(target_bodipy)+0.5]);
set(gca,'YTick',2:length(target_bodipy));
set(gca,'YTickLabel',target_bodipy(2:end));
% cb = colorbar();

% create a colorbar, with worst values in red and best values in green
cmap_g = [161, 230, 170] ./ 255;
cmap_y = [255, 247, 172] ./ 255;
cmap_r = [255, 153, 179] ./ 255;

intervals1 = linspace(0,1,128);
intervals2 = linspace(0,1,128);
customcmap = [0.5,0.5,0.5; ...
    interp1([0; 1],[cmap_r; cmap_y],intervals1(1:end-1)); ...
    interp1([0; 1],[cmap_y; cmap_g],intervals2)];
colormap(customcmap);            
cb = colorbar();
cb.Title.String = '\Delta\lambda (nm)';

% title('Spectral centroid difference');

formatfig(f3,'presentation',fullfile(output_dir,'sc_difference.eps'));
formatfig(f3,'presentation',fullfile(output_dir,'sc_difference.png'));

% plot the cnn classification matrix as a heatmap
f4 = figure(4); clf;
imagesc(accuracy_matrix_cnn);
xlim([0.5,length(target_bodipy)-0.5]);
set(gca,'XTick',1:length(target_bodipy)-1);
set(gca,'XTickLabel',target_bodipy(1:end-1));
ylim([1.5,length(target_bodipy)+0.5]);
set(gca,'YTick',2:length(target_bodipy));
set(gca,'YTickLabel',target_bodipy(2:end));
% cb = colorbar();

% create a colorbar, with worst values in red and best values in green
cmap_g = [161, 230, 170] ./ 255;
cmap_y = [255, 247, 172] ./ 255;
cmap_r = [255, 153, 179] ./ 255;

intervals1 = linspace(0,1,6);
intervals2 = linspace(0,1,6);
customcmap = [...
    0.5,0.5,0.5; ...
    interp1([0; 1],[cmap_r; cmap_y],intervals1(1:end-1)); ...
    interp1([0; 1],[cmap_y; cmap_g],intervals2)];
colormap(customcmap);            
cb = colorbar('location','eastoutside');
% set the colorbar limits
caxis([0.45,1]+[-0.025,0.025]);

cb.Title.String = 'Classification Accuracy';

formatfig(f4,'presentation',fullfile(output_dir,'cnn_accuracy.eps'));
formatfig(f4,'presentation',fullfile(output_dir,'cnn_accuracy.png'));

% plot the cnn precision matrix as a heatmap
f5 = figure(5); clf;
imagesc(precision_matrix_cnn);
xlim([0.5,length(target_bodipy)-0.5]);
set(gca,'XTick',1:length(target_bodipy)-1);
set(gca,'XTickLabel',target_bodipy(1:end-1));
ylim([1.5,length(target_bodipy)+0.5]);
set(gca,'YTick',2:length(target_bodipy));
set(gca,'YTickLabel',target_bodipy(2:end));
% cb = colorbar();

% use the previously defined colormap
colormap(customcmap);
cb = colorbar('location','eastoutside');
caxis([0.45,1]+[-0.025,0.025]);

cb.Title.String = 'Classification Precision';

formatfig(f5,'presentation',fullfile(output_dir,'cnn_precision.eps'));
formatfig(f5,'presentation',fullfile(output_dir,'cnn_precision.png'));

% plot the cnn recall matrix as a heatmap
f6 = figure(6); clf;
imagesc(recall_matrix_cnn);
xlim([0.5,length(target_bodipy)-0.5]);
set(gca,'XTick',1:length(target_bodipy)-1);
set(gca,'XTickLabel',target_bodipy(1:end-1));
ylim([1.5,length(target_bodipy)+0.5]);
set(gca,'YTick',2:length(target_bodipy));
set(gca,'YTickLabel',target_bodipy(2:end));
% cb = colorbar();

% use the previously defined colormap
colormap(customcmap);
cb = colorbar('location','eastoutside');
caxis([0.45,1]+[-0.025,0.025]);

cb.Title.String = 'Classification Recall';

formatfig(f6,'presentation',fullfile(output_dir,'cnn_recall.eps'));
formatfig(f6,'presentation',fullfile(output_dir,'cnn_recall.png'));

% plot the spectral centroid misidentification as a heatmap
f7 = figure(7); clf;
imagesc(1-misidentification_sc);
xlim([0.5,length(target_bodipy)-0.5]);
set(gca,'XTick',1:length(target_bodipy)-1);
set(gca,'XTickLabel',target_bodipy(1:end-1));
ylim([1.5,length(target_bodipy)+0.5]);
set(gca,'YTick',2:length(target_bodipy));
set(gca,'YTickLabel',target_bodipy(2:end));

% create a colorbar, with worst values in red and best values in green
cmap_g = [161, 230, 170] ./ 255;
cmap_y = [255, 247, 172] ./ 255;
cmap_r = [255, 153, 179] ./ 255;

intervals1 = linspace(0,1,6);
intervals2 = linspace(0,1,6);
customcmap = [...
    0.5,0.5,0.5; ...
    interp1([0; 1],[cmap_r; cmap_y],intervals1(1:end-1)); ...
    interp1([0; 1],[cmap_y; cmap_g],intervals2)];
colormap(customcmap);            
cb = colorbar('location','eastoutside');
caxis([0.45,1]+[-0.025,0.025]);

title('Spectral Centroid Misidentification');

formatfig(f7,'presentation',fullfile(output_dir,'sc_misidentification.eps'));
formatfig(f7,'presentation',fullfile(output_dir,'sc_misidentification.png'));

% plot the f1 score as a heatmap
f8 = figure(8); clf;
imagesc(f1_score_cnn);
xlim([0.5,length(target_bodipy)-0.5]);
set(gca,'XTick',1:length(target_bodipy)-1);
set(gca,'XTickLabel',target_bodipy(1:end-1));
ylim([1.5,length(target_bodipy)+0.5]);
set(gca,'YTick',2:length(target_bodipy));
set(gca,'YTickLabel',target_bodipy(2:end));
colormap(customcmap);
cb = colorbar();
caxis([0.45,1]+[-0.025,0.025]);

cb.Title.String = 'F1 Score';

formatfig(f8,'presentation',fullfile(output_dir,'cnn_f1.eps'));
formatfig(f8,'presentation',fullfile(output_dir,'cnn_f1.png'));

% save the three matrices
sc_accuracy = 1 - misidentification_sc;
cnn_accuracy = accuracy_matrix_cnn;
cnn_recall = recall_matrix_cnn;
cnn_precision = precision_matrix_cnn;
cnn_f1 = f1_score_cnn;
sc_difference = abs(mean_spectral_centroid - mean_spectral_centroid');
save(fullfile(output_dir,'matrices.mat'),'sc_accuracy','cnn_accuracy','sc_difference','f1_score_cnn','cnn_recall','cnn_precision','cnn_f1');


function [accuracy, precision, recall] = calculate_metrics(confusion_matrix)
    % confusion_matrix: n x n confusion matrix where n is the number of classes

    % Number of classes
    num_classes = size(confusion_matrix, 1);

    % Initialize variables for macro-averaging
    precision_per_class = zeros(1, num_classes);
    recall_per_class = zeros(1, num_classes);
    
    % Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
    for class_idx = 1:num_classes
        % True Positives (TP)
        TP = confusion_matrix(class_idx, class_idx);
        
        % False Positives (FP): Sum of predictions for this class minus true positives
        FP = sum(confusion_matrix(:, class_idx)) - TP;
        
        % False Negatives (FN): Sum of true samples of this class minus true positives
        FN = sum(confusion_matrix(class_idx, :)) - TP;
        
        % True Negatives (TN): Everything that is not FP, FN, or TP
        TN = sum(confusion_matrix(:)) - (TP + FP + FN);
        
        % Precision: TP / (TP + FP)
        if (TP + FP) == 0
            precision_per_class(class_idx) = 0; % Avoid division by zero
        else
            precision_per_class(class_idx) = TP / (TP + FP);
        end
        
        % Recall: TP / (TP + FN)
        if (TP + FN) == 0
            recall_per_class(class_idx) = 0; % Avoid division by zero
        else
            recall_per_class(class_idx) = TP / (TP + FN);
        end
    end
    
    % Macro-Averaged Precision and Recall
    precision = mean(precision_per_class);
    recall = mean(recall_per_class);

    % Compute Accuracy: (sum of diagonal / total samples)
    correct_predictions = trace(confusion_matrix); % Sum of diagonal elements
    total_predictions = sum(confusion_matrix(:));  % Total number of samples
    accuracy = correct_predictions / total_predictions;
end
