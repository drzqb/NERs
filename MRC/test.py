from transformers import BertTokenizer,TFBertModel

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

mrc = tokenizer(["找到所有的地点地址了吗",
                        "找到所有的人物名称",
                        "找到所有的机构组织"]
                        , add_special_tokens=True, return_tensors="tf",padding=True)
print(mrc)

bert = TFBertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

print(bert(mrc))