from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import precision_recall_curve
import numpy as np

def get_roc_metrics(y_true_arr, y_pred_arr, auc_only = False, final_activation='sigmoid', class_idx=1):
    "Compute all receiver operating characteristic metrics."
    #print('Computing AUC for y_true: ' + str(y_true_arr))
    #print('Computing AUC for y_pred: ' + str(y_pred_arr))
    
    # for netowrks which use sigmoid as final activation --> output shape = (1,1)
    if final_activation == 'sigmoid':
        # compute false positive rate (1 - specificity), true positive rate (sensitivity) and thresholds (sklearn uses all possible thresholds from y_pred)
        falsepositiverate, truepositiverate, thresholds = roc_curve(y_true_arr, y_pred_arr, pos_label = 1)
        #print(thresholds)
        
        # compute the area under the receiver curve
        roc_auc = auc(falsepositiverate, truepositiverate)
        #print(roc_auc)
        
    # for netowrks which use softmax as final activation --> output shape = (1,nr_classes)
    if final_activation == 'softmax':
        # compute false positive rate (1 - specificity), true positive rate (sensitivity) and thresholds (sklearn uses all possible thresholds from y_pred)
        falsepositiverate, truepositiverate, thresholds = roc_curve(y_true_arr[:, class_idx], \
                                                                    y_pred_arr[:,class_idx], pos_label = 1)
        #print(thresholds)
        
        # compute the area under the receiver curve
        roc_auc = auc(falsepositiverate, truepositiverate)
        #print(roc_auc)
        
    if auc_only:
        return roc_auc
    else:
        return falsepositiverate, truepositiverate, thresholds, roc_auc 
    
def get_pr_metrics(y_true_arr, y_pred_arr, auc_only = False, final_activation='sigmoid', class_idx=1):
    "Compute all precision recall metrics."
    #print('Computing AUC for y_true: ' + str(y_true_arr))
    #print('Computing AUC for y_pred: ' + str(y_pred_arr))
    
    # for networks which use sigmoid as final activation --> output shape = (1,1)
    if final_activation == 'sigmoid':
        precision, recall, thresholds = precision_recall_curve(y_true_arr, y_pred_arr)

        # compute the area under the pr curve
        pr_auc = auc(recall, precision)
        
    # for networks which use softmax as final activation --> output shape = (1,nr_classes)
    if final_activation == 'softmax':
        precision, recall, thresholds = precision_recall_curve(y_true_arr[:, class_idx], \
                                                                    y_pred_arr[:,class_idx], pos_label = 1)
        
        # compute the area under the pr curve
        pr_auc = auc(recall, precision)
        
    if auc_only:
        return pr_auc
    else:
        return precision, recall, thresholds, pr_auc 

