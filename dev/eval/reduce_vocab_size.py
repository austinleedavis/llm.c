import huggingface_hub as hub
from transformers import GPT2LMHeadModel

hub.notebook_login()

input = 'log_chess_gpt_d12_hf'
output = 'austindavis/ChessGPT_d12'

llm = GPT2LMHeadModel.from_pretrained(input)

state_dict = llm.state_dict()
state_dict['transformer.wte.weight'] = state_dict['transformer.wte.weight'][:72]
state_dict['lm_head.weight'] = state_dict['lm_head.weight'][:72]

cfg = llm.config
cfg.bos_token_id = 1
cfg.eos_token_id = 2
cfg.vocab_size = 72
cfg.name_or_path = output

llm2 = GPT2LMHeadModel(cfg)
llm2.load_state_dict(state_dict)

llm2.save_pretrained(output)
llm2.push_to_hub('austindavis')
