import json
import os
from fashionpedia.fp import Fashionpedia

annotations_file ='Fashionpedia/instances_attributes_train2020.json'
output_file='Fashionpedia/train_data.jsonl'
image_dir = 'Fashionpedia/train'
print(f"正在加载标注文件: {annotations_file}")
fp=Fashionpedia(annotations_file)
print("加载数据完成")
all_img_ids=fp.getImgIds()
target_img_ids=all_img_ids[:1000]  # 只处理前1000张图片
print(f"处理图片数量: {len(target_img_ids)}")

with open(output_file,'w', encoding='utf-8') as outfile:
    processed_count = 0
    for img_id in target_img_ids:
        img_info=fp.loadImgs(img_id)[0]
        img_path=os.path.join(image_dir,img_info['file_name'])
        ann_ids=fp.getAnnIds(imgIds=img_id)
        anns =fp.loadAnns(ann_ids)
        for anno in anns:
            category_name=fp.loadCats(anno['category_id'])[0]['name']
            attributes=fp.loadAttrs(anno['attribute_ids'])
            attribute_names=[attr['name']for attr in attributes]
            super_cats=[attr['supercategory']for attr in attributes]
            x1,y1,width,height=[int(v) for v in anno['bbox']]
            x2, y2 = x1 + width, y1 + height
            user_prompt = f"请分析图中<box>({x1}, {y1}, {x2}, {y2})</box>的时尚单品，返回JSON格式的类别和所有属性。"
            #助手回答
            target_json={"类别": category_name,
                         "属性": attribute_names,
                         "超类别": super_cats,}
            final_record = {"image": img_path,
                            "conversations":[{"from":"human","value":user_prompt + "\n<image>"},
                                             {"from":"gpt","value":json.dumps(target_json,ensure_ascii=False)}
                                             ]}
            outfile.write(json.dumps(final_record, ensure_ascii=False) + '\n')
            processed_count += 1
print(f"已处理 {processed_count} 条记录，输出到 {output_file}")



