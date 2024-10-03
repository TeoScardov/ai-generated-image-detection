import os
import argparse
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--results_file', default='./results.json', type=str)


class CalculateMetrics:
    def parse_pred_ans(self, pred_ans):
        if pred_ans in ["Real", "Fake"]:
            return pred_ans
        else:
            return "other"

    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "Real": 0,
            "Fake": 1,
            "other": 2,
        }

        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]

        acc = accuracy_score(gts, preds)

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == 4:  # "other"
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        
        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[0, 1])
        precision = precision_score(clean_gts, clean_preds, average='macro')
        recall = recall_score(clean_gts, clean_preds, average='macro')
        f1 = f1_score(clean_gts, clean_preds, average='macro')
        
        metric_dict = {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "confusion_matrix": conf_mat,
            "other_num": other_num,
            "f1": f1
        }
        
        return metric_dict

    def process_result(self, results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)

        gts = []
        preds = []
        task_name = "real/generated prediction"
        task_other_ans_num = 0
        img_num = len(data)

        for item in data:
            gt_ans = item["ground_truth"].strip()
            pred_ans = item["answer"].strip()

            assert gt_ans in ["Real", "Fake"] 

            pred_ans = self.parse_pred_ans(pred_ans)
            assert pred_ans in ["Real", "Fake", "other"]

            gts.append(gt_ans)
            preds.append(pred_ans)

            if pred_ans == "other":
                task_other_ans_num += 1

        metric_dict = self.compute_metric(gts, preds)

        task_score = metric_dict["acc"] * 100

        print("total score:", task_score, "\n")
        print("\t", task_name, " score:", task_score, "\n")
        print(f"\t F1 score: {metric_dict['f1']}\n")
        print("\t invalid responses:", task_other_ans_num, "\n")
        print("\t confusion matrix:\n\t\t\t", metric_dict["confusion_matrix"][0], "\n\t\t\t", metric_dict["confusion_matrix"][1], "\n")
        print("\n")
        
        return 

if __name__ == "__main__":
    cal = CalculateMetrics()

    args = parser.parse_args()
    results_file = args.results_file
    cal.process_result(results_file)
