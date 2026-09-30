clear;
% Resolve the data directory relative to this script's location
% (repo_root/src/original_single_split/<this file>), so it runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
data_dir = fullfile(repo_root, 'data', 'processed_combined');
% Result .mat files are likewise written under repo_root, not the
% caller's current directory, so output lands in the same place
% regardless of where this script is invoked from.
output_dir = fullfile(repo_root, 'results', 'original_single_split', 'triplet');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

target_bodipy = 1:7;
target_bodipy = [target_bodipy, target_bodipy + 8];

wl_min = 600;
wl_max = 700;

cutoff_val = 1000;

n_bodipy = length(target_bodipy);

n_molecules_total = zeros(n_bodipy,1);
mean_spectral_centroid = zeros(n_bodipy,1);

combination_list = nan(nchoosek(length(target_bodipy),3),3);
confusion_matrixes = nan(nchoosek(length(target_bodipy),3),3,3);
accuracies = nan(nchoosek(length(target_bodipy),3),1);
precisions = nan(nchoosek(length(target_bodipy),3),1);
recalls = nan(nchoosek(length(target_bodipy),3),1);
f1_scores = nan(nchoosek(length(target_bodipy),3),1);

% upd = textprogressbar(100,'updatestep',1);

for ii = 2

rng(ii);

jj = 1;

% plot the n_photons vs. spectral centroids
for i_bodipy = 1:n_bodipy
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
    
    for j_bodipy = (1+i_bodipy):n_bodipy
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
        
        for k_bodipy = (1+j_bodipy):n_bodipy
            k_curr_bodipy = target_bodipy(k_bodipy);
            load(fullfile(data_dir, sprintf('bodipy%d.mat',k_curr_bodipy)));
            k_n_photons = n_photons_combined;
            k_spectral_rois = spectral_rois_combined;
            k_spectral_centroid = spectral_centroids_combined;
            k_spectral_profiles = spectral_profiles_combined;

            % sort the spectral profiles by n_photons, descending
            [k_n_photons, sort_idx] = sort(k_n_photons,'descend');
            k_spectral_rois = k_spectral_rois(:,:,sort_idx);
            k_spectral_centroid = k_spectral_centroid(sort_idx);
            k_spectral_profiles = k_spectral_profiles(:,sort_idx);

            % cutoff the data at n_photons = cutoff_val
            k_spectral_rois = k_spectral_rois(:,:,k_n_photons > cutoff_val);
            k_spectral_profiles = k_spectral_profiles(:,k_n_photons > cutoff_val);
            k_spectral_centroid = k_spectral_centroid(k_n_photons > cutoff_val);
            k_n_photons = k_n_photons(k_n_photons > cutoff_val);

            % remove outliers of the spectral centroids
            [k_spectral_centroid,k_rm_rows] = rmoutliers(k_spectral_centroid,'mean','thresholdfactor',4);
            % update the spectral profiles
            k_spectral_rois = k_spectral_rois(:,:,~k_rm_rows);
            k_spectral_profiles = k_spectral_profiles(:,~k_rm_rows);
            k_n_photons = k_n_photons(~k_rm_rows);
        
            % perform classification test based on CNN
            % 400 brightest images for training, 100 images for testing        
            
            % check that we have at least 500 spectral rois for each dye
            if ~(size(i_spectral_rois,3) < 600 || size(j_spectral_rois,3) < 600 || size(k_spectral_rois,3) < 600)
                % scramble the indices for training and testing
                i_idx = randperm(size(i_spectral_rois,3),600);
                j_idx = randperm(size(j_spectral_rois,3),600);
                k_idx = randperm(size(k_spectral_rois,3),600);

                i_train_rois = i_spectral_rois(:,:,i_idx(1:420));
                i_validate_rois = i_spectral_rois(:,:,i_idx(421:510));
                i_test_rois = i_spectral_rois(:,:,i_idx(511:end));
                j_train_rois = j_spectral_rois(:,:,j_idx(1:420));
                j_validate_rois = j_spectral_rois(:,:,j_idx(421:510));
                j_test_rois = j_spectral_rois(:,:,j_idx(511:end));
                k_train_rois = k_spectral_rois(:,:,k_idx(1:420));
                k_validate_rois = k_spectral_rois(:,:,k_idx(421:510));
                k_test_rois = k_spectral_rois(:,:,k_idx(511:end));

                train_labels = [ones(420,1); 2*ones(420,1); 3*ones(420,1)];
                validate_labels = [ones(90,1); 2*ones(90,1); 3*ones(90,1)];
                test_labels = [ones(90,1); 2*ones(90,1); 3*ones(90,1)];

                train_idx = randperm(420*3);
                validate_idx = randperm(90*3);
                test_idx = randperm(90*3);

                X_train = cat(3,i_train_rois,j_train_rois,k_train_rois);
                X_validate = cat(3,i_validate_rois,j_validate_rois,k_validate_rois);
                X_test = cat(3,i_test_rois,j_test_rois,k_test_rois);

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
                n_labels = 3;
                
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

                % Compute the confusion matrix
                confusion_matrix = confusionmat(testing_labels, test_predicted);

                % Calculate the accuracy, precision, and recall
                [accuracy, precision, recall, f1score] = calculate_metrics(confusion_matrix);

                % Store the results
                combination_list(jj,:) = [i_curr_bodipy,j_curr_bodipy,k_curr_bodipy];

                accuracies(jj) = accuracy;
                precisions(jj) = precision;
                recalls(jj) = recall;
                f1_scores(jj) = f1score;

                confusion_matrixes(jj,:,:) = confusion_matrix;

                jj = jj + 1;
            end
        end
    end
end

sorted_accuracy = sortrows([accuracies,combination_list],1,'descend');
sorted_precision = sortrows([precisions,combination_list],1,'descend');
sorted_recall = sortrows([recalls,combination_list],1,'descend');
sorted_f1_score = sortrows([f1_scores,combination_list],1,'descend');

save(fullfile(output_dir,sprintf('analyze_bodipy_triplets_%d.mat',ii)),'sorted_accuracy','sorted_precision','sorted_recall','sorted_f1_score','confusion_matrixes','combination_list');

end

