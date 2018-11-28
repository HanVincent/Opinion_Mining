from .constant import *

def get_segments(tag_seq, label_map):
    segs = []
    start = -1
    for i, y in enumerate(tag_seq):
        if y == label_map["O"]: 
            if start != -1: segs.append((start, i))
            start = -1
        elif y == label_map["B-DSE"]:
            if start != -1: segs.append((start, i))
            start = i
        elif y == label_map["I-DSE"]:
            if start == -1: start = i
        else:
            print("Bad predicted:", ix_to_tag[y])
    
    if start != -1 and start != len(tag_seq):
        segs.append((start, len(tag_seq)))
        
    return segs


def show(y_predict, y_true):
    ps = [ix_to_tag[ix] for ix in y_predict.cpu().numpy()]
    ts = [ix_to_tag[ix] for ix in y_true.cpu().numpy()]
    
    print("Predict: {}\tTrue: {}".format(''.join(ps), ''.join(ts)))


def evaluate(predicts, trues, label_map):
    assert len(predicts) == len(trues)
    
    precision_prop, recall_prop = .0, .0
    precision_bin, recall_bin = 0, 0
    predict_total, true_total = 0, 0
    
    for y_predict, y_true in zip(predicts, trues):
        assert len(y_predict) == len(y_true)

        predict_segs = get_segments(y_predict, label_map)
        true_segs = get_segments(y_true, label_map)

        predict_count = len(predict_segs)
        true_count = len(true_segs)
        
        predict_total += predict_count
        true_total += true_count
        
        predict_flags = [False for i in range(predict_count)]
        true_flags = [False for i in range(true_count)]

        for t_i, (t_start, t_end) in enumerate(true_segs):
            for p_i, (p_start, p_end) in enumerate(predict_segs):
                assert p_start != p_end

                l_max = t_start if t_start > p_start else p_start
                r_min = t_end   if t_end   < p_end else p_end
                overlap = (r_min - l_max) if r_min > l_max else 0
                
                precision_prop += overlap / (p_end - p_start)
                recall_prop += overlap / (t_end - t_start)

                if not predict_flags[p_i] and overlap > 0:
                    precision_bin += 1
                    predict_flags[p_i] = True
                if not true_flags[t_i] and overlap > 0:
                    recall_bin += 1
                    true_flags[t_i] = True

                    
        # show(y_predict, y_true)
        
    precision = (precision_bin / predict_total) if predict_total != 0 else 1
    recall = recall_bin / true_total
    f1 = (2 * precision * recall) / (precision + recall)    
    binary_overlap = { 'precision': precision, 'recall': recall, 'f1': f1 }
    
    precision = (precision_prop / predict_total) if predict_total != 0 else 1
    recall = recall_prop / true_total
    f1 = (2 * precision * recall) / (precision + recall)
    proportional_overlap = { 'precision': precision, 'recall': recall, 'f1': f1 }
        
    return { 'binary': binary_overlap, 'proportional': proportional_overlap }
