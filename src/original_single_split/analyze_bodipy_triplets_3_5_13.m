clear;
% Resolve the data directory relative to this script's location
% (repo_root/src/original_single_split/<this file>), so it runs
% correctly regardless of MATLAB's current working directory.
repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
data_dir = fullfile(repo_root, 'data', 'processed_combined');

target_bodipy = [3,5,13];
% 
% target_bodipy = [1:2,4:8];
% target_bodipy = [target_bodipy, target_bodipy + 9];

wl_min = 600;
wl_max = 700;

cutoff_val = 1500;

n_bodipy = length(target_bodipy);

misidentification_sc = ones(n_bodipy,n_bodipy,n_bodipy);
accuracy_matrix_cnn = nan(n_bodipy,n_bodipy,n_bodipy);

n_molecules_total = zeros(n_bodipy,1);
mean_spectral_centroid = zeros(n_bodipy,1);

rng(1);

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
            
            % perform classification test based on spectral centroids
            % we select the number of bodipy molecules in the mean spectral centroid for each dye, plus minus 3
            sc_window = 3;
            % first we check if the mean spectral centroid is within the sc_window of the other dye
            if ~(abs(mean(i_spectral_centroid) - mean(j_spectral_centroid)) < sc_window) && ...
               ~(abs(mean(i_spectral_centroid) - mean(k_spectral_centroid)) < sc_window) && ...
               ~(abs(mean(j_spectral_centroid) - mean(k_spectral_centroid)) < sc_window)
               
                % if it isn't, we count the number of dyes with spectral centroid in each window
                i_sc_window = mean(i_spectral_centroid) + [-sc_window, sc_window];
                j_sc_window = mean(j_spectral_centroid) + [-sc_window, sc_window];
                k_sc_window = mean(k_spectral_centroid) + [-sc_window, sc_window];

                % total number of molecules
                n_molecules = length(i_spectral_centroid) + length(j_spectral_centroid) + length(k_spectral_centroid);

                % count the number of molecules in each window
                n_i_sc_window = sum(j_spectral_centroid > i_sc_window(1) & j_spectral_centroid < i_sc_window(2)) + ...
                                sum(k_spectral_centroid > i_sc_window(1) & k_spectral_centroid < i_sc_window(2));
                n_j_sc_window = sum(i_spectral_centroid > j_sc_window(1) & i_spectral_centroid < j_sc_window(2)) + ...
                                sum(k_spectral_centroid > j_sc_window(1) & k_spectral_centroid < j_sc_window(2));
                n_k_sc_window = sum(i_spectral_centroid > k_sc_window(1) & i_spectral_centroid < k_sc_window(2)) + ...
                                sum(j_spectral_centroid > k_sc_window(1) & j_spectral_centroid < k_sc_window(2));

                % calculate the fraction of misidentified molecules
                misidentification_sc(i_bodipy,j_bodipy,k_bodipy) = (n_i_sc_window + n_j_sc_window + n_k_sc_window)/n_molecules;
                misidentification_sc(i_bodipy,k_bodipy,j_bodipy) = (n_i_sc_window + n_j_sc_window + n_k_sc_window)/n_molecules;
                misidentification_sc(j_bodipy,i_bodipy,k_bodipy) = (n_i_sc_window + n_j_sc_window + n_k_sc_window)/n_molecules;
                misidentification_sc(j_bodipy,k_bodipy,i_bodipy) = (n_i_sc_window + n_j_sc_window + n_k_sc_window)/n_molecules;
                misidentification_sc(k_bodipy,i_bodipy,j_bodipy) = (n_i_sc_window + n_j_sc_window + n_k_sc_window)/n_molecules;
                misidentification_sc(k_bodipy,j_bodipy,i_bodipy) = (n_i_sc_window + n_j_sc_window + n_k_sc_window)/n_molecules;
            end
        
            % perform classification test based on CNN
            % 400 brightest images for training, 100 images for testing        
            
            disp(size(i_spectral_rois,3));
            disp(size(j_spectral_rois,3));
            disp(size(k_spectral_rois,3));
            
            % check that we have at least 500 spectral rois for each dye
            if ~(size(i_spectral_rois,3) < 700 || size(j_spectral_rois,3) < 700 || size(k_spectral_rois,3) < 700)
                % % take the brightest 500 images for each dye
                % i_spectral_rois = i_spectral_rois(:,:,1:500);
                % j_spectral_rois = j_spectral_rois(:,:,1:500);
                % k_spectral_rois = k_spectral_rois(:,:,1:500);
                % scramble the indices for training and testing
                i_idx = randperm(size(i_spectral_rois,3),700);
                j_idx = randperm(size(j_spectral_rois,3),700);
                k_idx = randperm(size(k_spectral_rois,3),700);

                i_train_rois = i_spectral_rois(:,:,i_idx(1:500));
                i_validate_rois = i_spectral_rois(:,:,i_idx(501:600));
                i_test_rois = i_spectral_rois(:,:,i_idx(601:end));
                j_train_rois = j_spectral_rois(:,:,j_idx(1:500));
                j_validate_rois = j_spectral_rois(:,:,j_idx(501:600));
                j_test_rois = j_spectral_rois(:,:,j_idx(601:end));
                k_train_rois = k_spectral_rois(:,:,k_idx(1:500));
                k_validate_rois = k_spectral_rois(:,:,k_idx(501:600));
                k_test_rois = k_spectral_rois(:,:,k_idx(601:end));

                train_labels = [ones(500,1); 2*ones(500,1); 3*ones(500,1)];
                validate_labels = [ones(100,1); 2*ones(100,1); 3*ones(100,1)];
                test_labels = [ones(100,1); 2*ones(100,1); 3*ones(100,1)];

                train_idx = randperm(500*3);
                validate_idx = randperm(100*3);
                test_idx = randperm(100*3);

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
                    convolution2dLayer(3,8,'Padding','same','Name','conv1')
                    batchNormalizationLayer('Name','bn1')
                    reluLayer('Name','relu1')

                    maxPooling2dLayer(2,'Stride',2,'Name','pool1')

                    convolution2dLayer(3,16,'Padding','same','Name','conv2')
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

            end
        end
    end
end

[accuracy, precision, recall, f1score] = calculate_metrics(confusion_matrix);

disp(round(accuracy*100,1));
disp(round(f1score*100,1));
disp(round(precision*100,1));
disp(round(recall*100,1));

disp(confusion_matrix);