import torch
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor,BitsAndBytesConfig
from PIL import Image
from qwen_vl_utils import process_vision_info
model_path="/home/chenli/weber/Qwen2.5-VL-Fasion/weight"
#qlora配置
quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
print("正在加载处理器和模型")
processor=AutoProcessor.from_pretrained(model_path,trust_remote_code=True)#读取配置文件检测是什么模型
#加载模型权重
model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
    
)
print("模型加载完成！")
image_path="fashion.jpg"
prompt_text="请分析这张图片中的时尚单品，以JSON格式返回各单品的类别、颜色和风格。"
#设定输入内容
messages=[{
    "role":"user",
    "content":[
        {"type":"image","image":image_path},
        {"type":"text","text":prompt_text},
    ],
}]              
#输入内容的规范化，添加特殊标记以及占位符
text=processor.apply_chat_template(
    messages,tokenize=False,add_generation_prompt=True
)
#处理视觉信息，提取图片和视频输入
image_inputs,video_inputs=process_vision_info(messages)
#整合文本、图片和视频输入
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)
print("正在执行前向传播...")
with torch.no_grad():
    generated_ids=model.generate(**inputs,max_new_tokens=512)
    generated_ids=[out_ids[len(in_ids):]for in_ids,out_ids in zip(inputs.input_ids,generated_ids)]
    response=processor.batch_decode(generated_ids,skip_special_tokens=True)[0]
print("向前传播成功！")
print("模型输出：\n",response)

