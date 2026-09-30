function [accuracy, precision, recall, f1score] = calculate_metrics(confusion_matrix)
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

    % Compute F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    if (precision + recall) == 0
        f1score = 0; % Avoid division by zero
    else
        f1score = 2 * (precision * recall) / (precision + recall);
    end
end
