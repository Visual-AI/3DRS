import os
import json
import argparse


def get_sqa_question_type(question):
    question = question.lstrip()
    if question[:4].lower() == 'what':
        return 'what'
    elif question[:2].lower() == 'is':
        return 'is'
    elif question[:3].lower() == 'how':
        return 'how'
    elif question[:3].lower() == 'can':
        return 'can'
    elif question[:5].lower() == 'which':
        return 'which'
    else:
        return 'others'   # others


def main(args):

    for split in ["train", "val", "test"]:
        qid2ques = {}
        with open(os.path.join(args.sqa3d_dir, 'balanced', f'v1_balanced_questions_{split}_scannetv2.json'), 'r') as f:
            q = json.load(f)
            for item in q['questions']:
                qid2ques[item['question_id']] = item

        transformed_data = []
        with open(os.path.join(args.sqa3d_dir, 'balanced', f'v1_balanced_sqa_annotations_{split}_scannetv2.json')) as f:
            a = json.load(f)

            for item in a['annotations']:
                ques = qid2ques[item['question_id']]
                situations = ques['alternative_situation'] + [ques['situation']] if split=="train" else [ques['situation']]
                assert len(item["answers"])==1, "the length of answers must be equal to 1."
                for situation_txt in situations:
                    transformed_data.append({
                        "id": item["question_id"],
                        "video": f"scannet/{item['scene_id']}",
                        "conversations": [
                            {
                                "value": f"<image> {situation_txt} {ques['question']} Answer the question using a single word or phrase.",
                                "from": "human"
                            },
                            {
                                "value": item["answers"][0]["answer"],
                                "from": "gpt"
                            }
                        ],
                        "metadata": {
                            "dataset": "sqa3d",
                            "question_type": get_sqa_question_type(ques['question'])
                        }
                    })
        
        print(len(transformed_data))
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'sqa3d_{split}_llava_style.json'), 'w') as f:
            json.dump(transformed_data, f)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sqa3d_dir", type=str, default="data/benchmark/sqa_task")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    main(args)    