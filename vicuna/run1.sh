CUDA_VISIBLE_DEVICES=7 python run_crows_pairs.py --model /home/textbox/align/www/model/contra-02/checkpoint-270
CUDA_VISIBLE_DEVICES=7 python run_winogender.py --model /home/textbox/align/www/model/contra-02/checkpoint-270
CUDA_VISIBLE_DEVICES=7 python run_truthfulqa.py --model /home/textbox/align/www/model/contra-02/checkpoint-270
CUDA_VISIBLE_DEVICES=7 python run_mmlu.py --model /home/textbox/align/www/model/contra-02/checkpoint-270
CUDA_VISIBLE_DEVICES=7 python run_gsm.py --model /home/textbox/align/www/model/contra-02/checkpoint-270
CUDA_VISIBLE_DEVICES=7 python run_hellaswag.py --model /home/textbox/align/www/model/contra-02/checkpoint-270
CUDA_VISIBLE_DEVICES=7 python run_nq.py --model /home/textbox/align/www/model/contra-02/checkpoint-270
CUDA_VISIBLE_DEVICES=7 python run_dialog.py --model /home/textbox/align/www/model/contra-02/checkpoint-270
