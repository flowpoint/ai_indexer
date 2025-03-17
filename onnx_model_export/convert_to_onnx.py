from transformers import AutoTokenizer
#from optimum.onnxruntime import ORTModelForQuestionAnswering
#from optimum.onnxruntime import ORTModelForFeatureExtraction

# optimum-cli export onnx --model intfloat/e5-small-v2 intfloat_e5_small_v2

import onnxruntime

#model_name = 'intfloat/e5-small-v2'
model_name = './intfloat_e5_small_v2'
onnx_file = './intfloat_e5_small_v2/model.onnx'

tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = ORTModelForFeatureExtraction.from_pretrained(model_name)
inputs = dict(tokenizer(["What am I using?"]*8192))
inputs.pop('token_type_ids')

sess = onnxruntime.InferenceSession(onnx_file)

ret = sess.run(None, inputs)

from timeit import timeit
tt = timeit(lambda: sess.run(None, inputs), number=20) / 20

#outputs = model(**inputs)
#print(outputs)



