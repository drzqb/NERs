from transformers import BertTokenizer, BertConfig, TFBertModel

config = BertConfig.from_pretrained("e:/tools/chinese-roberta-wwm-ext")
config.num_hidden_layers = 2

bert = TFBertModel.from_pretrained("e:/tools/chinese-roberta-wwm-ext",
                                        config=config
                                        )

tokenizer = BertTokenizer.from_pretrained("e:/tools/chinese-roberta-wwm-ext")
